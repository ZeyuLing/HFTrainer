#!/usr/bin/env python3
"""Patch Brax UInt64 for JAX pmap axis-flattening compatibility."""

from __future__ import annotations

import importlib.util
import argparse
from pathlib import Path


OLD_POST_INIT = '''  def __post_init__(self):
    """Cast post init."""
    object.__setattr__(self, "hi", jnp.uint32(self.hi))
    object.__setattr__(self, "lo", jnp.uint32(self.lo))
'''

NEW_POST_INIT = '''  def __post_init__(self):
    """Cast post init."""
    # JAX pmap builds pytree dummy leaves with object() while flattening axes.
    if type(self.hi) is object or type(self.lo) is object:
      return
    object.__setattr__(self, "hi", jnp.uint32(self.hi))
    object.__setattr__(self, "lo", jnp.uint32(self.lo))
'''

OLD_TO_NUMPY = '''  def to_numpy(self):
    """Convert UInt64 to numpy uint64."""
    hi_np = np.array(self.hi, dtype=np.uint64)
    lo_np = np.array(self.lo, dtype=np.uint64)
    return (hi_np << 32) | lo_np
'''

NEW_TO_NUMPY = '''  def to_numpy(self):
    """Convert UInt64 to numpy uint64."""
    hi_np = np.asarray(self.hi).astype(np.uint64, copy=False)
    lo_np = np.asarray(self.lo).astype(np.uint64, copy=False)
    return np.bitwise_or(np.left_shift(hi_np, np.uint64(32)), lo_np)
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, help="Explicit brax/training/types.py path to patch.")
    args = parser.parse_args()
    if args.path:
        path = Path(args.path)
    else:
        spec = importlib.util.find_spec("brax.training.types")
        if spec is None or spec.origin is None:
            raise SystemExit("brax.training.types is not importable")
        path = Path(spec.origin)
    text = path.read_text()
    changed = False
    if OLD_POST_INIT in text:
        text = text.replace(OLD_POST_INIT, NEW_POST_INIT)
        changed = True
    elif NEW_POST_INIT not in text:
        raise SystemExit(f"expected UInt64.__post_init__ block not found in {path}")

    if OLD_TO_NUMPY in text:
        text = text.replace(OLD_TO_NUMPY, NEW_TO_NUMPY)
        changed = True
    elif NEW_TO_NUMPY not in text:
        raise SystemExit(f"expected UInt64.to_numpy block not found in {path}")

    if not changed:
        print(f"[brax-uint64-patch] already patched: {path}")
        return
    path.write_text(text)
    print(f"[brax-uint64-patch] patched: {path}")


if __name__ == "__main__":
    main()
