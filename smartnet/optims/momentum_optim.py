from ..optim import Optim
from collections import OrderedDict


class MomentumOptim(Optim):
    """
    # description:
        sgd optimization algorithm with Momentum.
        v = v * momentum + dw
        w = w - lr * dw
    """
    def __init__(self, trainable_parameters, lr=0.01, weight_decay=0, momentum=0.9):
        super(MomentumOptim, self).__init__("momentum", trainable_parameters)
        self._lr = lr
        self._weight_decay = weight_decay
        self._momentum_buffs = OrderedDict()
        self._momentum = momentum

    def step(self):
        for name, par in self._trainable_parameters.items():
            if not par.requires_grad:
                continue
            data = par.data
            grad = par.grad
            if self._weight_decay != 0:
                par.update_grad(data * self._weight_decay)

            if name not in self._momentum_buffs.keys():
                self._momentum_buffs[name] = par.pkg.zeros_like(grad)
            momentum = self._momentum_buffs[name]
            momentum[:] = self._momentum * momentum + grad
            par.set_values(data - self._lr * momentum)

    def get_property(self):
        return {"lr": self._lr,
                "weight_decay": self._weight_decay,
                "momentum": self._momentum}