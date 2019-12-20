# coding=utf-8
from ..module import *
from ..core import function as F


class MSELayer(Module):
    """
    # description:
        mean square error layer.
        loss = (y_hat - y)**2 / m
        where m is number of samples.
        dloss/dy_hat = 2 * (y_hat - y) / m
    """
    def __init__(self):
        super(MSELayer, self).__init__("MSE")

    def forward(self, *inputs, **kwargs):
        layer_input = inputs[0]
        label = inputs[1]

        return F.mse(layer_input, label)
