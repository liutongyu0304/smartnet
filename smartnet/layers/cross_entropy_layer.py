# coding=utf-8
from ..module import *
from ..core.tensor_op import TensorOp


class SmartCrossEntropyLayer(SmartModule):
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
    def __init__(self):
        super(SmartCrossEntropyLayer, self).__init__("CrossEntropy")

    def forward(self, *inputs, **kwargs):
        layer_input = inputs[0]
        label = inputs[1]

        return TensorOp.cross_entropy(layer_input, label)
