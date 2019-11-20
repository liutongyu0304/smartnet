# coding=utf-8
from ..optim import SmartOptim


class SmartSGDOptim(SmartOptim):
    """
    # description:
        sgd optimization algorithm.
        w = w - lr * dw
    """
    def __init__(self, name, trainable_parameters, lr=0.01, weight_decay=0):
        super(SmartSGDOptim, self).__init__(name, trainable_parameters)
        self._lr = lr
        self._weight_decay = weight_decay

    def step(self):
        for value in self._trainable_parameters:
            par = value["parameter"]
            if self._weight_decay != 0:
                par.grad[:] = par.grad + par.data * self._weight_decay

            par.data[:] = par.data - self._lr * par.grad

    def get_property(self):
        return {"lr": self._lr, "weight_decay": self._weight_decay}