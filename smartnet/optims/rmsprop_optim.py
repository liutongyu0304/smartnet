from ..optim import Optim


class RMSPropOptim(Optim):
    """
    # description:
        root mean square prop algorithm.
        sdw = beta * sdw + (1 - beta) * dw
        w = w - lr * dw / (sdw + eps)**0.5
    """
    def __init__(self, trainable_parameters, lr=0.01, weight_decay=0,
                 beta=0.9, eps=1e-8):
        super(RMSPropOptim, self).__init__("rmsprop", trainable_parameters)
        self._lr = lr
        self._weight_decay = weight_decay
        self._rmsprop_buffs = dict()
        self._beta = beta
        self._eps = eps

    def step(self):
        for name, par in self._trainable_parameters.items():
            if not par.requires_grad:
                continue
            data = par.data
            grad = par.grad
            if self._weight_decay != 0:
                par.update_grad(data * self._weight_decay)

            if name not in self._rmsprop_buffs.keys():
                self._rmsprop_buffs[name] = par.pkg.zeros_like(grad)
            rmsprop = self._rmsprop_buffs[name]

            rmsprop[:] = self._beta * rmsprop + (1 - self._beta) * grad**2
            par.set_values(data - self._lr * grad / (rmsprop + self._eps)**0.5)

    def get_property(self):
        return {"lr": self._lr,
                "weight_decay": self._weight_decay,
                "beta": self._beta,
                "eps": self._eps}