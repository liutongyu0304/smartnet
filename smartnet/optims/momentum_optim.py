from ..optim import SmartOptim
import numpy as np


class SmartMomentumOptim(SmartOptim):
    """
    # description:
        sgd optimization algorithm with Momentum.
        v = v * momentum + dw
        w = w - lr * dw
    """
    def __init__(self, name, trainable_parameters, lr=0.01, weight_decay=0, momentum=0.9):
        super(SmartMomentumOptim, self).__init__(name, trainable_parameters)
        self._lr = lr
        self._weight_decay = weight_decay
        self._momentum_buffs = dict()
        self._momentum = momentum

    def step(self):
        for value in self._trainable_parameters:
            par = value["parameter"]
            name = value["name"]
            if self._weight_decay != 0:
                par.grad[:] = par.grad + par.data * self._weight_decay

            if name not in self._momentum_buffs.keys():
                self._momentum_buffs[name] = np.zeros_like(par.grad)
            momentum = self._momentum_buffs[name]
            momentum[:] = self._momentum * momentum + par.grad
            par.data[:] = par.data - self._lr * momentum

    def get_property(self):
        return {"lr": self._lr,
                "weight_decay": self._weight_decay,
                "momentum": self._momentum}