# coding=utf-8
from .memory import *


class SmartStorage(object):
    def __init__(self, shape, device="cpu", dtype=float):
        self._device = device
        self._sp = get_package_by_device(device)
        self._data = self._sp.zeros(shape, dtype=dtype)
    
    def reshape(self, shape):
        size = 1
        for i in shape:
            size *= i
        if size != self.size:
            raise ValueError("size of shape should equal to origin size")
        
        storage = SmartStorage(shape, self._device, self.dtype)
        storage._data[:] = self._data.copy().reshape(shape)
        return storage

    def transpose(self):
        data = self._data.copy().transpose()
        storage = SmartStorage(data.shape, self._device, self.dtype)
        storage._data[:] = data
        return storage
    
    def __neg__(self):
        data = -1 * self._data
        return self.copy(data)

    def __add__(self, right):
        if isinstance(right, SmartStorage):
            data = self._data + right.data
        else:
            data = self._data + right    
        return self.copy(data)
    
    def __radd__(self, left):
        return self + left

    def __sub__(self, right):
        if isinstance(right, SmartStorage):
            data = self._data - right.data
        else:
            data = self._data - right       
        return self.copy(data)

    def __rsub__(self, left):
        if isinstance(left, SmartStorage):
            data = left.data - self._data
        else:
            data = left - self._data       
        return self.copy(data)

    def __mul__(self, right):
        if isinstance(right, SmartStorage):
            data = self._data * right.data
        else:
            data = self._data * right     
        return self.copy(data)

    def __rmul__(self, left):
        return self * left

    def __truediv__(self, right):
        if isinstance(right, SmartStorage):
            data = self._data / right.data
        else:
            data = self._data / right
        return self.copy(data)
    
    def __rtruediv__(self, left):
        if isinstance(left, SmartStorage):
            data = left._data / self._data
        else:
            data = left / self._data
        return self.copy(data)

    def __pow__(self, right):
        data = self._data**right
        return self.copy(data)

    def __str__(self):
        s = "SmartStorage shape: {}, device: {}, dtype: {}, " \
            "\ndata: {}".format(self.shape, self._device, self.dtype, self._data)
        return s

    def __repr__(self):
        return self.__str__(self)

    def matmul(self, right):
        if isinstance(right, SmartStorage):
            data = self._sp.matmul(self.data, right.data)
        else:
            data = self._sp.matmul(self.data, right)
        return self.copy(data)

    def exp(self):
        data = self._sp.exp(self._data)
        return self.copy(data)

    def log(self):
        data = self._sp.log(self._data)
        return self.copy(data)

    def sum(self, axis=None, keepdims=False):
        data = self._sp.sum(self._data, axis=axis, keepdims=keepdims)
        if len(data.shape) == 0:
            data = data.reshape((1, ))
        return self.copy(data)

    def maximum(self, left):
        data = self._sp.maximum(left, self._data)
        return self.copy(data)

    def minimum(self, left):
        data = self._sp.minimum(left, self._data)
        return self.copy(data)

    def copy(self, data=None):
        if data is None:
            storage = SmartStorage(self.shape, self._device, self.dtype)
            storage._data[:] = self.data
        else:
            storage = SmartStorage(data.shape, self._device, data.dtype)
            storage._data[:] = data
        return storage

    def set_zero(self):
        self._data[:] = 0.0

    def set_values(self, data):
        if isinstance(data, SmartStorage):
            self._data[:] = data.data
        else:
            self._data[:] = data

    def to_cpu(self):
        if self._device == "cpu":
            return
        else:
            self._data = to_cpu(self._data)
            self._device = "cpu"

    def to_gpu(self):
        if self._device == "cuda":
            return
        else:
            self._data = to_gpu(self._data)
            self._device = "cuda"

    @property
    def shape(self):
        return self._data.shape
    
    @property
    def size(self):
        return self._data.size

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def data(self):
        return self._data

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._data.dtype
