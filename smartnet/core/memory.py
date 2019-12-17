# coding=utf-8
import numpy as np
try:
    import cupy as cp
    SUPPORT_CUDA = True
except ImportError:
    SUPPORT_CUDA = False


def get_package_by_device(device):
    if device == "cpu":
        return np
    elif device == "cuda":
        assert SUPPORT_CUDA
        return cp
    else:
        raise ValueError("unsupported device")


def to_cpu(data):
    assert SUPPORT_CUDA
    assert isinstance(data, cp.ndarray)
    return cp.asnumpy(data)


def to_gpu(data):
    assert SUPPORT_CUDA
    assert isinstance(data, np.ndarray)
    return cp.asarray(data)

