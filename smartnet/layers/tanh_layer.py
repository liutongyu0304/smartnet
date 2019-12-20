# coding=utf-8
from ..module import *
from ..core import function as F


class TanhLayer(Module):
    def __init__(self):
        super(TanhLayer, self).__init__("TanhLayer")

    def forward(self, *inputs, **kwargs):
        layer_input = inputs[0]
        # a(n) = (exp(z(n)) - exp(-z(n))) / (exp(z(n) + exp(-z(n))))
        return F.tanh(layer_input)