# coding=utf-8
from ..optim import SmartOptim


class SmartSGDOptim(SmartOptim):
    def __init__(self, name, trainable_parameters, lr=0.01, weight_decay=0):
        super(SmartSGDOptim, self).__init__(name, trainable_parameters)
        self._lr = lr
        self._weight_decay = weight_decay

    def step(self):
        for par in self._trainable_parameters.values():
            if self._weight_decay != 0:
                par.data[:] = par.data * (1 + self._weight_decay) - self._lr * par.grad
            else:
                par.data[:] = par.data - self._lr * par.grad
