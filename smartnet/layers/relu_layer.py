# coding=utf-8
from ..layer import SmartLayer
from ..tensor import *


class SmartReluLayer(SmartLayer):
    def __init__(self, name, need_backward=True):
        super(SmartReluLayer, self).__init__(name, need_backward)
        self._previous_inputs = 1
        self._outside_inputs = 0

    def set_up_layer(self, inputs):
        # relu layer gets one input and one output
        self._inputs = [inputs[0]]
        layer_input = self._inputs[0]

        layer_output = SmartTensor(np.zeros_like(layer_input.data))
        self._outputs = [layer_output]

    def forward(self):
        layer_input = self._inputs[0]
        layer_output = self._outputs[0]
        # a(n) = max(0, z(n))
        layer_output.data[:] = np.maximum(0, layer_input.data)

    def backward(self):
        layer_input = self._inputs[0]
        layer_output = self._outputs[0]
        # dz += 0 if a == 0 else 1

        grad = np.zeros_like(layer_output.data)
        grad[layer_input.data > 0] = 1
        layer_input.grad[:] = layer_input.grad + layer_output.grad * grad