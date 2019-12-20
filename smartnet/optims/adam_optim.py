from ..optim import Optim
from collections import OrderedDict


class AdamOptim(Optim):
    """
    # description:
        adam algorithm.
        vdw = momentum * vdw + (1 - momentum) * dw   # momentum
        sdw = beta * sdw + (1 - beta) * dw**2        # rmsprop
        vdw_correct = vdw / (1 - momentum**t)        # bias correction
        sdw_correct = sdw / (1 - beta**t)

        w = w - lr * vdw_correct / (sdw_correct + eps)**0.5
    """
    def __init__(self, trainable_parameters, lr=0.01, weight_decay=0,
                 momentum=0.9, beta=0.999, eps=1e-8):
        super(AdamOptim, self).__init__("adam", trainable_parameters)
        self._lr = lr
        self._weight_decay = weight_decay
        self._momentum_buffs = OrderedDict()
        self._rmsprop_buffs = OrderedDict()
        self._momentum = momentum
        self._beta = beta
        self._eps = eps
        self._iterations = 1

    def step(self):
        for name, par in self._trainable_parameters.items():
            if not par.requires_grad:
                continue
            data = par.data
            grad = par.grad
            if self._weight_decay != 0:
                par.set_values(data * self._weight_decay)

            if name not in self._momentum_buffs.keys():
                self._momentum_buffs[name] = par.pkg.zeros_like(grad)
            momentum = self._momentum_buffs[name]
            momentum[:] = self._momentum * momentum + (1 - self._momentum) * grad
            momentum_correct = momentum / (1 - self._momentum**self._iterations)

            if name not in self._rmsprop_buffs.keys():
                self._rmsprop_buffs[name] = par.pkg.zeros_like(grad)
            rmsprop = self._rmsprop_buffs[name]
            rmsprop[:] = self._beta * rmsprop + (1 - self._beta) * grad**2
            rmsprop_correct = rmsprop / (1 - self._beta**self._iterations)

            par.set_values(par.data - self._lr * momentum_correct / (rmsprop_correct + self._eps)**0.5)

    def get_property(self):
        return {"lr": self._lr,
                "weight_decay": self._weight_decay,
                "momentum": self._momentum,
                "beta": self._beta,
                "eps": self._eps}