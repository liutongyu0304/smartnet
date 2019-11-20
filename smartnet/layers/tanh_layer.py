# coding=utf-8


from ..layer import SmartLayer
from ..tensor import *


class SmartTanhLayer(SmartLayer):
    def __init__(self, name, need_backward=True):
        super(SmartTanhLayer, self).__init__(name, need_backward)
        self._previous_inputs = 1
        self._outside_inputs = 0

    def set_up_layer(self, inputs):
        # tanh layer gets one input and one output
        self._inputs = [inputs[0]]
        layer_input = self._inputs[0]

        layer_output = SmartTensor(np.zeros_like(layer_input.data))
        self._outputs = [layer_output]

    def forward(self):
        layer_input = self._inputs[0]
        layer_output = self._outputs[0]
        # a(n) = (exp(z(n)) - exp(-z(n))) / (exp(z(n) + exp(-z(n))))
        exp_positive_x = np.exp(layer_input.data)
        exp_negative_x = np.exp(-layer_input.data)
        layer_output.data[:] = (exp_positive_x - exp_negative_x) / (exp_positive_x + exp_negative_x)

    def backward(self):
        layer_input = self._inputs[0]
        layer_output = self._outputs[0]
        # da = dz * (1 - a**2)

        layer_input.grad[:] = layer_input.grad + layer_output.grad * (1 - layer_output.data**2)