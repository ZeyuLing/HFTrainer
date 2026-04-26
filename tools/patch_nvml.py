"""Patch missing NVML symbols for old NVIDIA drivers (tlinux3-0013 on Taiji).

PyTorch 2.5.0 asserts that nvmlDeviceGetNvLinkRemoteDeviceType exists in
libnvidia-ml.so.1 during CUDA init (c10/cuda/driver_api.cpp:33). Old drivers
are missing this symbol, causing an INTERNAL ASSERT failure.

This module uses ctypes to inject the missing symbol into the already-loaded
libnvidia-ml.so.1 library. Import this BEFORE importing torch.

Usage in start_cmd:
    python3 -c 'import tools.patch_nvml' && python3 tools/train.py ...
  or:
    PYTHONPATH=. python3 -c 'import tools.patch_nvml; import torch; ...'
"""
import ctypes
import ctypes.util
import os
import sys


def _patch_nvml():
    """Attempt to patch missing NVML symbols. Safe to call on any system."""
    try:
        # Try to load libnvidia-ml.so.1
        try:
            nvml = ctypes.CDLL("libnvidia-ml.so.1", mode=ctypes.RTLD_GLOBAL)
        except OSError:
            return  # No NVIDIA driver, nothing to patch

        # Check if the symbol already exists
        try:
            _ = nvml.nvmlDeviceGetNvLinkRemoteDeviceType
            return  # Symbol exists, no patch needed
        except AttributeError:
            pass

        # Symbol missing — create a stub shared library in /tmp and load it
        # to make the symbol available globally
        stub_path = "/tmp/_nvml_stub.so"
        if not os.path.exists(stub_path):
            import subprocess
            import tempfile

            src = """
typedef unsigned int nvmlReturn_t;
typedef void* nvmlDevice_t;
nvmlReturn_t nvmlDeviceGetNvLinkRemoteDeviceType(
    nvmlDevice_t device, unsigned int link, unsigned int *pNvLinkDeviceType) {
    if (pNvLinkDeviceType) *pNvLinkDeviceType = 0;
    return 3;  /* NVML_ERROR_NOT_SUPPORTED */
}
"""
            with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
                f.write(src)
                src_path = f.name

            try:
                subprocess.run(
                    ["gcc", "-shared", "-fPIC", "-o", stub_path, src_path],
                    check=True, capture_output=True, timeout=30,
                )
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                return  # Can't compile, skip patching
            finally:
                os.unlink(src_path)

        # Load the stub globally — this makes the symbol available but
        # dlsym(nvml_handle, ...) still won't find it because it only
        # searches the specific library.
        # So this approach alone doesn't work for PyTorch's dlsym usage.
        # We need a different strategy.
        pass

    except Exception:
        pass  # Best effort, don't break anything


# The Python-level ctypes approach can't solve dlsym(handle, symbol) lookups
# because those only search the specific shared object, not the global table.
# The real fix must be at the LD_PRELOAD/dlsym level.
# Keeping this file as documentation of the problem.
