# coding=utf-8
from ..layer import SmartLayer
from ..tensor import *


class SmartCrossEntropyLayer(SmartLayer):
    """
    # description：
        cross entropy with soft max layer.
        for input a with shape of (m, C), where m is the number of samples, C is number of class.
        soft max operation:
            y_hat = exp(a) / sum(exp(a), axis=0)
        with shape of (m, C).

        shape of one hot encoding y_hat is (m, C), and only one entry is 1 for each row.
        cross entropy operation for y and y_hat:
            loss = -sum(y * ln(y_hat))
        finally backward gradient of loss to a:
            dloss / da = y_hat - y.
    # members:
        _inputs: list(SmartTensor)
            _inputs[0] is the output of previous layer
            _inputs[1] is the class label with one hot encoding
            _inputs[0], _inputs[1] shape of (m, C)
        _outputs: list(SmartTensor)
            _outputs[0]: shape of (1, ), loss, only one float entry
            _outputs[1]: shape of (m, C), soft max output y, saved for backward.
    """
    def __init__(self, name, need_backward=True):
        super(SmartCrossEntropyLayer, self).__init__(name, need_backward)
        self._previous_inputs = 1
        self._outside_inputs = 1

    def set_up_layer(self, inputs):
        self._inputs = inputs
        layer_input = self._inputs[0]  # shape of (m, C)
        label = self._inputs[1]        # shape of (m, C)
        if len(label.shape) != 2 or label.shape != layer_input.shape:
            raise Exception("mse layer {} layer input shape {} and label shape "
                            "{} does not match.".format(self._name, layer_input.shape, label.shape))
        self._outputs = [SmartTensor(np.zeros((1,))), SmartTensor(np.zeros_like(layer_input))]

    def forward(self):
        layer_input = self._inputs[0]
        label = self._inputs[1]

        soft_max_output = self._outputs[1]
        loss = self._outputs[0]

        exp_input = np.exp(layer_input.data)
        sum_exp_input = np.sum(exp_input, axis=0)
        soft_max_output.data[:] = exp_input / sum_exp_input

        loss.data[0] = -np.sum(label * np.log(soft_max_output.data))

    def backward(self):
        layer_input = self._inputs[0]
        label = self._inputs[1]

        soft_max_output = self._outputs[1]
        if layer_input.requires_grad:
            layer_input.grad[:] = layer_input.grad + soft_max_output.data - label.data
