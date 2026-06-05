"""Runtime compatibility shims for legacy SMPLify dependencies.

The MBench SMPLify path depends on chumpy, which still expects a few APIs that
were removed from modern Python/numpy.  Keep these shims isolated and opt-in by
adding this directory to PYTHONPATH for Table 3 evaluation jobs only.
"""

import inspect
from collections import namedtuple

import numpy as np


if not hasattr(inspect, "getargspec"):
    ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")

    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = getargspec


_NUMPY_ALIASES = {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "str": str,
    "unicode": str,
}

for _name, _value in _NUMPY_ALIASES.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)
