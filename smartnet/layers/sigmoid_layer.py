# coding=utf-8
from ..module import *
from ..core.tensor_op import TensorOp


class SmartSigmoidLayer(SmartModule):
    def __init__(self):
        super(SmartSigmoidLayer, self).__init__("SigmoidLayer")

    def forward(self, *inputs, **kwargs):
        layer_input = inputs[0]
        # a(n) = 1 / (1 + exp(-z(n)))
        return TensorOp.sigmoid(layer_input)
