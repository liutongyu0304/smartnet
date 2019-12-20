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


def get_package_by_data(data):
    if isinstance(data, np.ndarray):
        return np
    elif isinstance(data, cp.ndarray):
        return cp
    else:
        raise ValueError("unsupported data type")


def get_device_by_data(data):
    if isinstance(data, np.ndarray):
        return "cpu"
    elif isinstance(data, cp.ndarray):
        return "cuda"
    else:
        raise ValueError("unsupported data type")


def to_cpu(data):
    # assert SUPPORT_CUDA
    if isinstance(data, np.ndarray):
        return data
    else:
        assert isinstance(data, cp.ndarray)
        return cp.asnumpy(data)


def to_gpu(data):
    assert SUPPORT_CUDA
    if isinstance(data, cp.ndarray):
        return data
    else:
        assert isinstance(data, np.ndarray)
        return cp.asarray(data)


def sigmoid(data):
    pkg = get_package_by_data(data)
    return 1.0 / (1.0 + pkg.exp(-data))


def tanh(data):
    pkg = get_package_by_data(data)
    exp_positive_x = pkg.exp(data)
    exp_negative_x = pkg.exp(-data)
    new_data = (exp_positive_x - exp_negative_x) / (exp_positive_x + exp_negative_x)
    return new_data


def relu(data):
    pkg = get_package_by_data(data)
    return pkg.maximum(data, 0)


def mse(left, right):
    try:
        pkg = get_package_by_data(left)
    except ValueError:
        pkg = get_package_by_data(right)

    return pkg.sum((left - right) ** 2, axis=None, keepdims=True) / left.size