from ..optim import SmartOptim
import numpy as np


class SmartRMSPropOptim(SmartOptim):
    """
    # description:
        root mean square prop algorithm.
        sdw = beta * sdw + (1 - beta) * dw
        w = w - lr * dw / (sdw + eps)**0.5
    """
    def __init__(self, name, trainable_parameters, lr=0.01, weight_decay=0,
                 beta=0.9, eps=1e-8):
        super(SmartRMSPropOptim, self).__init__(name, trainable_parameters)
        self._lr = lr
        self._weight_decay = weight_decay
        self._rmsprop_buffs = dict()
        self._beta = beta
        self._eps = eps

    def step(self):
        for value in self._trainable_parameters:
            par = value["parameter"]
            name = value["name"]
            if self._weight_decay != 0:
                par.grad[:] = par.grad + par.data * self._weight_decay

            if name not in self._rmsprop_buffs.keys():
                self._rmsprop_buffs[name] = np.zeros_like(par.grad)
            rmsprop = self._rmsprop_buffs[name]

            rmsprop[:] = self._beta * rmsprop + (1 - self._beta) * par.grad**2
            par.data[:] = par.data - self._lr * par.grad / (rmsprop + self._eps)**0.5

    def get_property(self):
        return {"lr": self._lr,
                "weight_decay": self._weight_decay,
                "beta": self._beta,
                "eps": self._eps}