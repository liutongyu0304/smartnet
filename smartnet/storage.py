# coding=utf-8
import numpy as np
import numpy as gnp


def create_data_on_device(shape, device, dtype):
    assert isinstance(shape, tuple)
    if device == "cpu":
        return np.zeros(shape, dtype=dtype)
    else:
        return gnp.zeros(shape,dtype=dtype)


def matmul_on_device(left, right, device):
    if device == "cpu":
        return np.matmul(left, right)
    else:
        return gnp.matmul(left, right)


def exp_on_device(data, device):
    if device == "cpu":
        return np.exp(data)
    else:
        return gnp.exp(data)


def log_on_device(data, device):
    if device == "cpu":
        return np.log(data)
    else:
        return gnp.log(data)


def sum_on_device(data, device, axis, keepdims):
    if device == "cpu":
        return np.sum(data, axis=axis, keepdims=keepdims)
    else:
        return gnp.log(data)


def random_on_device(shape, device, dtype):
    assert isinstance(shape, tuple)
    if device == "cpu":
        return np.random.rand(*shape)
    else:
        return gnp.random.rand(*shape)


class SmartStorage(object):
    def __init__(self, shape, device="cpu", dtype=np.float32):
        self._device = device
        self._shape = shape
        self._dtype = dtype
        self._data = create_data_on_device(shape, device, dtype)
        self._size = 1
        for i in self._shape:
            self._size *= i
        self._ndim = len(self._shape)
    
    def reshape(self, shape):
        size = 1
        for i in shape:
            size *= i
        if size != self._size:
            raise ValueError("size of shape should equal to origin size")
        
        newStorage = SmartStorage(shape, self._device, self._data.dtype)
        newStorage._data[:] = self._data.copy().reshape(shape)
        return newStorage

    def transpose(self):
        data = self._data.copy().transpose()
        newStorage = SmartStorage(data.shape, self._device, data.dtype)
        newStorage._data[:] = data
        return newStorage
    
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
            "\ndata: {}".format(self._shape, self._device, self._dtype, self._data)
        return s

    def __repr__(self):
        return self.__str__(self)

    def matmul(self, right):
        assert self._device == right.device
        data = matmul_on_device(self, right, self._device)    
        return self.copy(data)

    def exp(self):
        data = exp_on_device(self._data, self._device)
        return self.copy(data)

    def log(self):
        data = log_on_device(self._data, self._device)
        return self.copy(data)

    def sum(self, axis=None, keepdims=False):
        data = sum_on_device(self._data, self._device, axis, keepdims)
        if len(data.shape) == 0:
            data = data.reshape((1, ))
        return self.copy(data)
    
    def copy(self, data=None):
        if data is None:
            storage = SmartStorage(self._data.shape, self._device, data.dtype)
        else:
            storage = SmartStorage(data.shape, self._device, data.dtype)
        storage._data[:] = data
        return storage

    def set_zeros(self):
        self._data[:] = 0.0

    @property
    def shape(self):
        return self._shape
    
    @property
    def size(self):
        return self._size

    @property
    def ndim(self):
        return self._ndim

    @property
    def data(self):
        return self._data

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype


class StorageOp(object):
    @staticmethod
    def reshape(data, shape):
        return data.reshape(shape)

    @staticmethod
    def transpose(data):
        return data.transpose()

    @staticmethod
    def add(left, right):
        return left + right

    @staticmethod
    def sub(left, right):
        return left - right

    @staticmethod
    def mul(left, right):
        return left * right

    @staticmethod
    def divide(left, right):
        return left / right

    @staticmethod
    def matmul(left, right):
        return left.matmul(right)

    @staticmethod
    def exp(data):
        return data.exp()

    @staticmethod
    def log(data):
        return data.log()

    @staticmethod
    def sum(data, axis, keepdims):
        return data.sum(axis, keepdims)

    @staticmethod
    def zeros(shape, device="cpu", dtype=np.float32):
        return SmartStorage(shape, device=device, dtype=dtype)

    @staticmethod
    def zeros_like(data):
        return SmartStorage(data.shape, device=data.device, dtype=data.dtype)

    @staticmethod
    def ones(shape, device="cpu", dtype=np.float32):
        storage = SmartStorage(shape, device=device, dtype=dtype)
        storage._data[:] = 1
        return storage

    @staticmethod
    def ones_like(data):
        return StorageOp.ones(data.shape, device=data.device, dtype=data.dtype)

    @staticmethod
    def random(shape, device="cpu", dtype=np.float32):
        storage = SmartStorage(shape, device=device, dtype=dtype)
        storage._data[:] = random_on_device(shape, device=device, dtype=dtype)
        return storage
    