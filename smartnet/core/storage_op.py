# coding=utf-8
from .storage import *


class StorageOp(object):
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
        sp = get_package_by_device(device)
        storage.set_values(sp.random.rand(*shape))
        return storage

    @staticmethod
    def reshape(data, shape):
        assert isinstance(data, SmartStorage)
        return data.reshape(shape)

    @staticmethod
    def transpose(data):
        assert isinstance(data, SmartStorage)
        return data.transpose()

    @staticmethod
    def add(left, right):
        assert isinstance(left, SmartStorage) or isinstance(right, SmartStorage)
        return left + right

    @staticmethod
    def sub(left, right):
        assert isinstance(left, SmartStorage) or isinstance(right, SmartStorage)
        return left - right

    @staticmethod
    def mul(left, right):
        assert isinstance(left, SmartStorage) or isinstance(right, SmartStorage)
        return left * right

    @staticmethod
    def divide(left, right):
        assert isinstance(left, SmartStorage) or isinstance(right, SmartStorage)
        return left / right

    @staticmethod
    def matmul(left, right):
        assert isinstance(left, SmartStorage) or isinstance(right, SmartStorage)
        return left.matmul(right)

    @staticmethod
    def maximum(left, right):
        if isinstance(left, SmartStorage):
            left.maximum(right)
        elif isinstance(right, SmartStorage):
            right.maximum(left)
        else:
            raise TypeError("at least one of variables should be SmartStorage")

    @staticmethod
    def minimum(left, right):
        if isinstance(left, SmartStorage):
            left.minimum(right)
        elif isinstance(right, SmartStorage):
            right.minimum(left)
        else:
            raise TypeError("at least one of variables should be SmartStorage")

    @staticmethod
    def exp(data):
        assert isinstance(data, SmartStorage)
        return data.exp()

    @staticmethod
    def log(data):
        assert isinstance(data, SmartStorage)
        return data.log()

    @staticmethod
    def sum(data, axis=None, keepdims=False):
        assert isinstance(data, SmartStorage)
        return data.sum(axis, keepdims)

    @staticmethod
    def sigmoid(data):
        assert isinstance(data, SmartStorage)
        return 1 / (1 + data._sp.exp(-data))

    @staticmethod
    def tanh(data):
        assert isinstance(data, SmartStorage)
        exp_positive_x = StorageOp.exp(data)
        exp_negative_x = StorageOp.exp(-data)
        new_data = (exp_positive_x - exp_negative_x) / (exp_positive_x + exp_negative_x)
        return new_data

    @staticmethod
    def relu(data):
        assert isinstance(data, SmartStorage)
        return StorageOp.maximum(data, 0)

    @staticmethod
    def mse(left, right):
        assert isinstance(left, SmartStorage) or isinstance(right, SmartStorage)
        return StorageOp.sum((left - right)**2, axis=None, keepdims=False) / left.size
