# coding=utf-8
from ..layer import SmartLayer
from ..tensor import *


class SmartMSELayer(SmartLayer):
    """
    # description:
        mean square error layer.
        loss = (y_hat - y)**2 / m
        where m is number of samples.
        dloss/dy_hat = 2 * (y_hat - y) / m
    """
    def __init__(self, name, need_backward=True):
        super(SmartMSELayer, self).__init__(name, need_backward)
        self._previous_inputs = 1
        self._outside_inputs = 1

    def set_up_layer(self, inputs):
        self._inputs = [inputs[0], inputs[1]]
        layer_input = self._inputs[0] # shape of (m, 1)
        label = self._inputs[1]       # shape of (m, 1)
        if len(label.shape) != 2 or label.shape != layer_input.shape:
            raise Exception("mse layer {} layer input shape {} and label shape "
                            "{} does not match.".format(self._name, layer_input.shape, label.shape))
        self._outputs = [SmartTensor(np.zeros((1,)))]

    def forward(self):
        layer_input = self._inputs[0]
        label = self._inputs[1]

        loss = np.sum((layer_input.data - label.data)**2) / layer_input.shape[0]
        self._outputs[0].data[0] = loss

    def backward(self):
        layer_input = self._inputs[0]
        label = self._inputs[1]

        if layer_input.requires_grad:
            layer_input.grad[:] = layer_input.grad + 2 * (layer_input.data - label.data) / layer_input.shape[0]
        if label.requires_grad:
            label.grad[:] = label.grad + 2 * (label.data - layer_input.data) / layer_input.shape[0]
