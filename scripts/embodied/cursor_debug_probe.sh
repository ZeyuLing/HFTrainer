#!/usr/bin/env bash
# Probe the debug machine: free GPUs, IsaacGym env, project, KIMODO model.
set +e
echo "==== HOST ===="; hostname
echo "==== GPUS (idx, used, total, util) ===="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "==== ISAACGYM ENV ===="
for p in /root/physflow_isaacgym_py38_cu118/bin/python /root/*isaac*/bin/python; do
  [ -x "$p" ] && echo "FOUND $p"
done
echo "==== project ===="
ls -d /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer >/dev/null 2>&1 && echo PROJ_OK || echo PROJ_MISSING
echo "==== isaacgym import test (if env found) ===="
PY=/root/physflow_isaacgym_py38_cu118/bin/python
if [ -x "$PY" ]; then
  cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/ProtoMotions 2>/dev/null
  $PY -c "import isaacgym, torch; print('isaacgym OK', torch.__version__, 'cuda', torch.cuda.is_available(), 'ndev', torch.cuda.device_count())" 2>&1 | tail -3
fi
echo "==== KIMODO model availability (quick) ===="
ls -d /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/kimodo 2>/dev/null
echo "==== DONE ===="
