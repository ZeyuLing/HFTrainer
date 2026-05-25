#!/usr/bin/env bash
set -euo pipefail

# Diagnostic wrapper — capture all output to a file
CONFIG="${1:?Usage: $0 <CONFIG> [args...]}"
shift

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJ_ROOT}/work_dirs/fp16_debug.log"

{
    echo "=== DIAGNOSTIC START $(date) ==="
    echo "Kernel: $(uname -r)"
    echo "Python: $(python3 --version)"
    echo "PWD: $(pwd)"
    echo "Config: ${CONFIG}"
    
    # Check tlinux
    KERNEL_VER="$(uname -r)"
    if [[ "${KERNEL_VER}" == *"tlinux3"* ]]; then
        echo "FATAL: tlinux3 detected!"
        exit 1
    fi
    echo "tlinux guard: PASS (kernel=${KERNEL_VER})"
    
    # Check CUDA
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>&1
    
    # Check config parse
    echo "--- Config parse test ---"
    python3 -c "
from mmengine.config import Config
cfg = Config.fromfile('${CONFIG}')
print(f'mixed_precision: {cfg.accelerator.mixed_precision}')
print(f'Config parse: OK')
" 2>&1
    
    # Check accelerate import + Accelerator creation with fp16
    echo "--- Accelerator creation test ---"
    python3 -c "
import torch
from accelerate import Accelerator, FullyShardedDataParallelPlugin
print('Accelerator imported OK')

fsdp_plugin = FullyShardedDataParallelPlugin(
    sharding_strategy='FULL_SHARD',
    backward_prefetch='BACKWARD_PRE',
    auto_wrap_policy='TRANSFORMER_BASED_WRAP',
    transformer_cls_names_to_wrap=['WanTransformerBlockWithMask'],
    state_dict_type='FULL_STATE_DICT',
    sync_module_states=True,
    use_orig_params=True,
    cpu_offload=False,
)
print('FSDP plugin created OK')

accelerator = Accelerator(
    mixed_precision='fp16',
    fsdp_plugin=fsdp_plugin,
)
print(f'Accelerator created OK, mixed_precision={accelerator.mixed_precision}')
print(f'Device: {accelerator.device}')
" 2>&1
    
    echo "--- All checks passed, launching training ---"
    
} 2>&1 | tee "${LOG_FILE}"

# Now actually run training
cd "${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
bash tools/taiji_dist_train.sh "${CONFIG}" "$@" 2>&1 | tee -a "${LOG_FILE}"
