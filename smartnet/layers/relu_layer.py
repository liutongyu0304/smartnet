# coding=utf-8
from ..module import *
from ..core.tensor_op import TensorOp


class SmartReluLayer(SmartModule):
    def __init__(self):
        super(SmartReluLayer, self).__init__("ReluLayer")

    def forward(self, *inputs, **kwargs):
        layer_input = inputs[0]
        # a(n) = max(0, z(n))
        return TensorOp.relu(layer_input)
