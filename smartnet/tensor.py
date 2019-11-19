# coding=utf-8
import numpy as np


class SmartTensor(object):
    def __init__(self, data, requires_grad=True, copy=False):
        if data is None:
            self._value = None
            self._grad = None
        assert isinstance(data, np.ndarray)
        self._value = data.copy() if copy else data
        self._grad = np.zeros_like(data)
        self._requires_grad = requires_grad

    def reshape(self, shape):
        if self._value is None:
            self._value = np.zeros(shape)
            self._grad = np.zeros(shape)
        else:
            self._value.reshape(shape)
            self._grad.reshape(shape)

    def update(self, lr):
        assert self._requires_grad
        self._value -= lr * self._grad

    def zero_grad(self):
        if self._grad is not None:
            self._grad[:] = 0.0

    @property
    def data(self):
        return self._value

    @property
    def grad(self):
        return self._grad

    @property
    def shape(self):
        return self._value.shape if isinstance(self._value, np.ndarray) else None


class SmartParameter(SmartTensor):
    def __init__(self, *args, **kwargs):
        super(SmartParameter, self).__init__(*args, **kwargs)
        pass
