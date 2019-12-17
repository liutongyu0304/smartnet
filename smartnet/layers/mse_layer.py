# coding=utf-8
from ..module import *
from ..core.tensor_op import TensorOp


class SmartMSELayer(SmartModule):
    """
    # description:
        mean square error layer.
        loss = (y_hat - y)**2 / m
        where m is number of samples.
        dloss/dy_hat = 2 * (y_hat - y) / m
    """
    def __init__(self):
        super(SmartMSELayer, self).__init__("MSE")

    def forward(self, *inputs, **kwargs):
        layer_input = inputs[0]
        label = inputs[1]

        return TensorOp.mse(layer_input, label)
