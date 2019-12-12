# coding=utf-8
import numpy as np
from .storage import *
from .op import *
import graph


class SmartTensor(object):
    def __init__(self, shape, device="cpu", dtype=np.float32, requires_grad=False):
        self._data = SmartStorage(shape, device=device, dtype=dtype)
        self._grad = None
        self._requires_grad = requires_grad
        self._is_leaf = True
        self._retain_grad = False
        self._op = None
        graph = graph.get_graph()
        graph.add_tensor(self)

    def reshape(self, shape):
        self._data.reshape(shape)
        if self._grad is not None:
            self._grad.reshape(shape)
        return self
    
    def requires_grad(self, requires_grad=True):
        if not self._is_leaf:
            raise Exception("non leaf tensor can not be set requires_grad.")
        self._requires_grad = requires_grad

    def update(self, lr):
        assert self._requires_grad
        if self._data is None or self._grad is None:
            return
        self._data = self._data - lr * self._grad
    
    def make_grad(self):
        if self._grad is not None:
            self._grad = StorageOp.zeros_like(self._data)

    def zero_grad(self):
        if self._grad is not None:
            self._grad.set_zero()

    def clear_grad(self):
        self._grad = None
    
    def set_leaf(self, is_leaf=True):
        self._is_leaf = is_leaf

    def set_retain_grad(self):
        self._retain_grad = True

    def set_op(self, op):
        self._op = op

    def detach(self):
        return self._data

    def reshape(self, shape):
        return ReshapeOperation()(self, shape)
    
    def transpose(self):
        return TransposeOperation()(self)
    
    def __neg__(self)
        return NegativeOperation()(self)

    def __add__(self, right):
        return AddOperation()(self, right)
    
    def __radd__(self, left):
        return AddOperation()(left, self)

    def __minus__(self, right):
        return MinusOperation()(self, right)

    def __rminus__(self, left):
        return MinusOperation()(left, self)

    def __mul__(self, right):
        return MulOperation()(self, right)

    def __rmul__(self, left):
        return MulOperation()(left, self)

    def __truediv__(self, right):
        return DivideOperation()(self, right)
    
    def __rtruediv__(self, left):
        return DivideOperation()(left, self)

    def __pow__(self, right):
        return PowOperation()(self, right)

    def matmul(self, right):
        return MatmulOperation()(self, right)

    def exp(self):
        return ExpOperation()(self)

    def log(self):
        return LogOperation()(self)

    @property
    def requires_grad(self):
        return self._requires_grad

    @property
    def data(self):
        return self._data

    @property
    def grad(self):
        return self._grad

    @property
    def shape(self):
        return self._data.shape

    @property
    def is_leaf(self):
        return self._is_leaf

    @property
    def retain_grad(self):
        return self._retain_grad

    @property
    def op(self):
        return self._op


class TensorOp(object):
    def reshape(tensor, shape):
        return tensor.reshape(shape)

    def transpose(tensor):
        return tensor.transpose()

    def add(left, right):
        return left + right

    def minus(left, right):
        return left - right

    def mul(left, right):
        return left * right

    def divide(left, right):
        return left / right

    def matmul(left, right):
        return left.matmul(right)

    def exp(tensor):
        return tensor.exp()

    def log(tensor):
        return tensor.log()

    def zeros(shape, device="cpu", dtype=np.float32, requires_grad=True):
        tensor = SmartTensor(shape, device=device, dtype=dtype, requires_grad=requires_grad)
        tensor.data._data[:] = 0.0
        return tensor

    def zeros_like(tensor):
        return SmartTensor(tensor.shape, device=tensor.device, dtype=tensor.dtype, requires_grad=tensor.requires_grad)

    def ones(shape, device="cpu", dtype=np.float32, requires_grad=True):
        tensor = SmartTensor(shape, device=device, dtype=dtype, requires_grad=requires_grad)
        tensor.data._data[:] = 1.0
        return tensor

    def ones_like(tensor):
        return self.ones(tensor.shape, device=tensor.device, dtype=tensor.dtype, requires_grad=tensor.requires_grad)

    def random(shape, device="cpu", dtype=np.float32, requires_grad=True):
        tensor = SmartTensor(shape, device=device, dtype=dtype, requires_grad=requires_grad)
        tensor.data._data[:] = random_on_device(shape, device=device, dtype=dtype)
        return tensor
