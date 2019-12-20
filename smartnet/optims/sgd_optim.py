# coding=utf-8
from ..optim import Optim


class SGDOptim(Optim):
    """
    # description:
        sgd optimization algorithm.
        w = w - lr * dw
    """
    def __init__(self, trainable_parameters, lr=0.01, weight_decay=0):
        super(SGDOptim, self).__init__("sgd", trainable_parameters)
        self._lr = lr
        self._weight_decay = weight_decay

    def step(self):
        for par in self._trainable_parameters.values():
            if not par.requires_grad:
                continue
            data = par.data
            grad = par.grad
            if self._weight_decay != 0:
                par.update_grad(data * self._weight_decay)

            par.set_values(data - self._lr * grad)

    def get_property(self):
        return {"lr": self._lr, "weight_decay": self._weight_decay}