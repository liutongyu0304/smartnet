# coding=utf-8
from ..layer import SmartLayer
from ..tensor import *


class SmartSigmoidLayer(SmartLayer):
    def __init__(self, name, need_backward=True):
        super(SmartSigmoidLayer, self).__init__(name, need_backward)
        self._previous_inputs = 1
        self._outside_inputs = 0

    def set_up_layer(self, inputs):
        # sigmoid layer gets one input and one output
        self._inputs = inputs
        layer_input = self._inputs[0]

        layer_output = SmartTensor(np.zeros_like(layer_input.data))
        self._outputs = [layer_output]

    def forward(self):
        layer_input = self._inputs[0]
        layer_output = self._outputs[0]
        # a(n) = 1 / (1 + exp(-z(n)))
        layer_output.data[:] = 1 / (1 + np.exp(-layer_input.data))

    def backward(self):
        layer_input = self._inputs[0]
        layer_output = self._outputs[0]
        # dz += a * (1 - a)

        layer_input.grad[:] = layer_input.grad + layer_output.grad * layer_output.data * (1 - layer_output.data)
