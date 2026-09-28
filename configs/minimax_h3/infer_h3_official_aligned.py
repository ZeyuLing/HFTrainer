"""Official H3 numerical parity recipe; CUDA FA3 kernel build required.

Validated on H200 with PyTorch 2.11.0+cu130. The portable base recipe uses
SDPA, whose floating-point results are not bit-identical to official FA3.
"""

_base_ = ["infer_h3_base_fl2va.py"]

pipeline = dict(attention_backend="flash_attention_3", noise_mode="independent")
