#!/usr/bin/env python3
"""Smoke-test OpenTrack imports and JAX GPU visibility."""

import importlib


MODULES = [
    "jax",
    "jaxlib",
    "mujoco",
    "mujoco_playground._src.mjx_env",
    "brax.training.agents.ppo.networks",
    "track_mj",
    "torch",
]


def main() -> None:
    for module in MODULES:
        importlib.import_module(module)

    import jax

    devices = jax.devices()
    platforms = [getattr(device, "platform", "") for device in devices]
    print("IMPORT_SMOKE_DEVICES", devices)
    print("IMPORT_SMOKE_PLATFORMS", platforms)
    if not any(platform in {"gpu", "cuda"} for platform in platforms):
        raise SystemExit("NO_GPU_DEVICE")
    print("IMPORT_SMOKE_OK")


if __name__ == "__main__":
    main()
