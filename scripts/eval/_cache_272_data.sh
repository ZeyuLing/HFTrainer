#!/bin/bash
# Cache MotionStreamer-272 test-set GT + texts (and optionally a pred npz dir)
# from slow CephFS into /dev/shm for fast repeated evaluation.
set -e
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
MS=${MS:-$REPO_ROOT/ref_repo/MotionStreamer/MotionStreamer}
SHM=${SHM:-/dev/shm/ms272_data}
SPLIT=${SPLIT:-$MS/humanml3d_272/split/test.txt}
mkdir -p "$SHM/motion_data" "$SHM/texts"

if [ ! -f "$SPLIT" ]; then
  echo "[cache] missing split file: $SPLIT" >&2
  exit 1
fi

echo "[cache] copying test GT npy + texts ..."
copy_one() {
  cid="$1"
  [ -f "$SHM/motion_data/$cid.npy" ] || cp "$MS/humanml3d_272/motion_data/$cid.npy" "$SHM/motion_data/" 2>/dev/null || true
  [ -f "$SHM/texts/$cid.txt" ]       || cp "$MS/humanml3d_272/texts/$cid.txt"       "$SHM/texts/"       2>/dev/null || true
}
export -f copy_one
export MS SHM
cat "$SPLIT" | xargs -P 16 -I{} bash -c 'copy_one "$@"' _ {}
echo "[cache] GT npy: $(ls $SHM/motion_data | wc -l), texts: $(ls $SHM/texts | wc -l)"

# Optional: cache a prediction npz dir, e.g.
#   bash _cache_272_data.sh PRED /abs/path/to/npz tag
if [ "${1:-}" = "PRED" ]; then
  PRED_SRC="$2"; TAG="$3"
  DST="/dev/shm/ms272_pred_$TAG"
  mkdir -p "$DST"
  echo "[cache] copying pred npz from $PRED_SRC -> $DST"
  ls "$PRED_SRC" | xargs -P 16 -I{} bash -c "[ -f \"$DST/{}\" ] || cp \"$PRED_SRC/{}\" \"$DST/\" 2>/dev/null || true"
  echo "[cache] pred npz: $(ls $DST | wc -l)"
fi
