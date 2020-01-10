# coding=utf-8
from __future__ import absolute_import


from .tensor import *
from .function import *
from .graph import no_grad


def zeros(shape, device="cpu", dtype=np.float32, requires_grad=False):
    t = Tensor(shape, device=device, dtype=dtype, requires_grad=requires_grad)
    return t


def zeros_like(tensor):
    return Tensor(tensor.shape, device=tensor.device,
                  dtype=tensor.dtype, requires_grad=tensor.requires_grad)


def ones(shape, device="cpu", dtype=np.float32, requires_grad=False):
    t = Tensor(shape=shape, device=device,
               dtype=dtype, requires_grad=requires_grad)
    t.data[:] = 1.0
    return t


def ones_like(tensor):
    return ones(tensor.shape, device=tensor.device,
                dtype=tensor.dtype, requires_grad=tensor.requires_grad)


def random(shape, device="cpu", dtype=np.float32, requires_grad=False):
    pkg = get_package_by_device(device)
    t = Tensor(data=pkg.random.rand(*shape), device=device,
               dtype=dtype, requires_grad=requires_grad)
    return t


def from_ndarray(data):
    t = Tensor(data=data)
    return t
