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

def exp_on_device(data):
    if device == "cpu":
        return np.exp(data)
    else:
        return gnp.exp(data)

def log_on_device(data):
    if device == "cpu":
        return np.log(data)
    else:
        return gnp.log(data)

def random_on_device(shape, device, dtype):
    assert isinstance(shape, tuple)
    if device == "cpu":
        return np.random.rand(shape, dtype=dtype)
    else:
        return gnp.random.rand(shape, dtype=dtype)


class SmartStorage(object):
    def __init__(shape, device="cpu", dtype=np.float32):
        self._device = device
        self._shape = shape
        self._data = create_data_on_device(shape, device, dtype)
        self._size = 1
        for i in self._shape:
            self._size *= i
        self._ndim = len(self._shape)
    
    def reshape(shape):
        size = 1
        for i in shape:
            size *= i
        if size != self._size:
            raise ValueError("size of shape should equal to origin size")
        
        newStorage = SmartStorage(shape, self._device, data.dtype)
        newStorage._data[:] = self._data.copy().reshape(shape)
        return newStorage

    def transpose(self):
        data = self._data.copy().transpose()
        newStorage = SmartStorage(data.shape, self._device, data.dtype)
        newStorage._data[:] = data
        return newStorage
    
    def __neg__(self)
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

    def __minus__(self, right):
        if isinstance(right, SmartStorage):
            data = self._data - right.data
        else:
            data = self._data - right       
        return self.copy(data)

    def __rminus__(self, left):
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
        if isinstance(right, SmartStorage):
            data = left._data / self._data
        else:
            data = left / self._data
        return self.copy(data)

    def __pow__(self, right):
        data = self._data**right
        return self.copy(data)

    def matmul(self, right):
        assert self._device == right.device
        data = matmul_on_device(self, right, self._device)    
        return self.copy(data)

    def exp(self):
        data = exp_on_device(self._data)    
        return self.copy(data)

    def log(self):
        data = log_on_device(self._data)    
        return self.copy(data)
    
    def copy(self, data=NOne):
        if data is None:
            newStorage = SmartStorage(self._data.shape, self._device, data.dtype)
        else:
            newStorage = SmartStorage(data.shape, self._device, data.dtype)
        newStorage._data[:] = data
        return newStorage

    def set_zeros(self):
        self._data[:] = 0.0
        
    def __rminus__(self, right):
        raise NotImplementedError

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

class StorageOp(object):
    def reshape(data, shape):
        return data.reshape(shape)

    def transpose(data):
        return data.transpose()

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

    def exp(data):
        return data.exp()

    def log(data):
        return data.log()

    def zeros(shape, device="cpu", dtype=np.float32):
        return SmartStorage(shape, device=device, dtype=dtype)

    def zeros_like(data):
        return SmartStorage(data.shape, device=data.device, dtype=data.dtype)

    def ones(shape, device="cpu", dtype=np.float32):
        storage = SmartStorage(shape, device=device, dtype=dtype)
        storage._data[:] = 1
        return storage

    def ones_like(shape, device="cpu", dtype=np.float32):
        return ones(data.shape, device=data.device, dtype=data.dtype)

    def random(shape, device="cpu", dtype=np.float32):
        storage = SmartStorage(shape, device=device, dtype=dtype)
        storage._data[:] = random_on_device(shape, device=device, dtype=dtype)
        return storage
    