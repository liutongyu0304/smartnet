# coding=utf-8
from ..module import *
from ..core import function as F


class SigmoidLayer(Module):
    def __init__(self):
        super(SigmoidLayer, self).__init__("SigmoidLayer")

    def forward(self, *inputs, **kwargs):
        layer_input = inputs[0]
        # a(n) = 1 / (1 + exp(-z(n)))
        return F.sigmoid(layer_input)
