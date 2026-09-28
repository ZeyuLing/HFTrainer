"""Small tensor assertions used by the local NVIDIA-derived graph."""
import contextlib
import functools
import warnings

def profiled_function(fn):
    return fn

@contextlib.contextmanager
def suppress_tracer_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield

def assert_shape(tensor, reference):
    if tensor.ndim != len(reference):
        raise AssertionError(f'Expected {len(reference)} dimensions, got {tensor.shape}')
    for size, expected in zip(tensor.shape, reference):
        if expected is not None and size != expected:
            raise AssertionError(f'Expected {reference}, got {tuple(tensor.shape)}')
