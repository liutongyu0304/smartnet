# coding=utf-8
from ..module import *
from ..core import function as F


class ReluLayer(Module):
    def __init__(self):
        super(ReluLayer, self).__init__("ReluLayer")

    def forward(self, *inputs, **kwargs):
        layer_input = inputs[0]
        # a(n) = max(0, z(n))
        return F.relu(layer_input)
