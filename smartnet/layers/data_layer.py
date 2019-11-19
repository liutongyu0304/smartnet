# coding=utf-8

from ..layer import SmartLayer
from ..tensor import *


class SmartDataLayer(SmartLayer):
    def __init__(self, name):
        super(SmartDataLayer, self).__init__(name, False)
        self._previous_inputs = 0
        self._outside_inputs = 1

    def set_up_layer(self, inputs):
        if isinstance(inputs, list):
            self._inputs = inputs
        else:
            self._inputs = [inputs]
        self._outputs = self._inputs

    def forward(self):
        pass

    def backward(self):
        pass

    def get_layer_property(self):
        pass
