# coding=utf-8
from ..module import *
from ..core.tensor_op import TensorOp


class SmartTanhLayer(SmartModule):
    def __init__(self):
        super(SmartTanhLayer, self).__init__("TanhLayer")

    def forward(self, *inputs, **kwargs):
        layer_input = inputs[0]
        # a(n) = (exp(z(n)) - exp(-z(n))) / (exp(z(n) + exp(-z(n))))
        return TensorOp.tanh(layer_input)