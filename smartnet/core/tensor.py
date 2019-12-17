# coding=utf-8
import numpy as np
from .storage_op import *


class SmartTensor(object):
    def __init__(self, shape, device="cpu", dtype=np.float32, requires_grad=False):
        self._data = SmartStorage(shape, device=device, dtype=dtype)
        self._grad = None
        self._requires_grad = requires_grad
        self._is_leaf = True
        self._retain_grad = False
        self._op = None
        from .graph import get_graph
        get_graph().add_tensor(self)
    
    def set_requires_grad(self, requires_grad=True):
        if not self._is_leaf:
            raise Exception("non leaf tensor can not be set requires_grad.")
        self._requires_grad = requires_grad

    def update_data(self, lr):
        assert self._requires_grad
        if self._data is None or self._grad is None:
            return
        self._data = self._data - lr * self._grad

    def update_grad(self, grad):
        # grad should be SmartStorage
        if self._grad is not None:
            assert isinstance(grad, SmartStorage)
            # assert grad.shape == self._grad.shape
            self._grad = self._grad + grad
    
    def make_grad(self):
        if self._grad is None:
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

    def set_values(self, data):
        # this will change tensor data without changing grad,
        # do not use it except constucting a new tensor
        if isinstance(data, SmartTensor):
            self._data.set_values(data.data)
        else:
            self._data.set_values(data)

    def detach(self):
        return self._data.data

    def reshape(self, shape):
        from .op import ReshapeOperation
        return ReshapeOperation()(self, shape)
    
    def transpose(self):
        from .op import TransposeOperation
        return TransposeOperation()(self)

    def sum(self, axis=None, keepdims=False):
        from .op import SumOperation
        return SumOperation(axis, keepdims)(self)

    def backward(self, retain_graph=False):
        from .graph import get_graph
        get_graph().auto_grad(self, retain_graph=retain_graph)
    
    def __neg__(self):
        from .op import NegativeOperation
        return NegativeOperation()(self)

    def __add__(self, right):
        from .op import AddOperation
        return AddOperation()(self, right)
    
    def __radd__(self, left):
        from .op import AddOperation
        return AddOperation()(left, self)

    def __sub__(self, right):
        from .op import SubOperation
        return SubOperation()(self, right)

    def __rsub__(self, left):
        from .op import SubOperation
        return SubOperation()(left, self)

    def __mul__(self, right):
        from .op import MulOperation
        return MulOperation()(self, right)

    def __rmul__(self, left):
        from .op import MulOperation
        return MulOperation()(left, self)

    def __truediv__(self, right):
        from .op import DivideOperation
        return DivideOperation()(self, right)
    
    def __rtruediv__(self, left):
        from .op import DivideOperation
        return DivideOperation()(left, self)

    def __pow__(self, right):
        from .op import PowOperation
        return PowOperation()(self, right)

    def __str__(self):
        grad = self._grad.data if self._grad is not None else self._grad
        s = "SmartTensor shape: {}, device: {}, dtype: {},\n" \
            "data: {}\ngrad: {}".format(self.shape, self.device,
                                        self.dtype, self._data.data, grad)
        return s

    def __repr__(self):
        return self.__str__()

    def matmul(self, right):
        from .op import MatmulOperation
        return MatmulOperation()(self, right)

    def exp(self):
        from .op import ExpOperation
        return ExpOperation()(self)

    def log(self):
        from .op import LogOperation
        return LogOperation()(self)

    def to_cpu(self):
        self._data.to_cpu()
        if self._grad is not None:
            self._grad.to_cpu()
        return self

    def to_gpu(self):
        self._data.to_gpu()
        if self._grad is not None:
            self._grad.to_gpu()
        return self

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
    def device(self):
        return self._data.device

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def is_leaf(self):
        return self._is_leaf

    @property
    def retain_grad(self):
        return self._retain_grad

    @property
    def op(self):
        return self._op

    @property
    def size(self):
        return self._data.size
