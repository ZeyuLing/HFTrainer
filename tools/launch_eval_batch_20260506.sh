#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 {e3|e8d} OUT_DIR" >&2
  exit 2
fi

kind="$1"
out_dir="$2"

case "${kind}" in
  e3)
    runner="tools/run_e3_m2m_latest_20260430.sh"
    ;;
  e8d)
    runner="tools/run_e8d_kimodo_fixed_20260430.sh"
    ;;
  *)
    echo "Unknown kind: ${kind}" >&2
    exit 2
    ;;
esac

mkdir -p "${out_dir}"
nohup bash "${runner}" "${out_dir}" > "${out_dir}/driver.log" 2>&1 &
pid="$!"
printf '%s\n' "${pid}" > "${out_dir}/driver.pid"
echo "started ${kind} pid=${pid} out=${out_dir}"
