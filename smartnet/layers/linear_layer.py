# coding=utf-8
from ..layer import SmartLayer
from ..tensor import *


class SmartLinearLayer(SmartLayer):
    def __init__(self, name, input_nodes, output_nodes, has_bias=True, need_backward=True):
        super(SmartLinearLayer, self).__init__(name, need_backward)
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._has_bias = has_bias
        self._weight = SmartTensor(None)
        self._bias = SmartTensor(None)
        self._reset_parameters()
        self._reset_trainable_parameters()
        self._previous_inputs = 1
        self._outside_inputs = 0

    def set_up_layer(self, inputs):
        # linear layer gets one input and one output
        layer_input = inputs[0]
        total_nodes = 1
        for i in layer_input.shape:
            total_nodes *= i
        if total_nodes % self._input_nodes != 0:
            raise Exception("linear layer {} total count of inputs[0] {} does not match "
                            "layer input nodes {}".format(self._name, total_nodes, self._input_nodes))
        layer_input.reshape((self._input_nodes, -1))

        layer_output = SmartTensor(np.zeros((self._output_nodes, layer_input)))
        self._weight = SmartTensor(np.random.rand(self._output_nodes, self._input_nodes))
        if self._has_bias:
            self._bias = SmartTensor(np.random.rand(self._output_nodes, 1))
        self._reset_parameters()
        self._reset_trainable_parameters()

        self._inputs = inputs
        self._outputs = [layer_output]

    def forward(self):
        layer_input = self._inputs[0]
        layer_output = self._outputs[0]
        # z(n) = w(n) * a(n-1) + b(n)
        data = layer_output.data
        np.matmul(self._weight.data, layer_input.data, data)
        if self._has_bias:
            data += self._bias.data

    def backward(self):
        # if not self._need_backward:
        #     raise Exception("linear layer {} has property need backward False")
        layer_input = self._inputs[0]
        layer_output = self._outputs[0]
        # da = w.t * dz
        grad = layer_input.grad
        np.matmul(self._weight.data.transpose(), layer_output.grad, grad)

        if self._need_backward:
            np.matmul(layer_output.grad, layer_input.transpose(), self._weight.grad)
            if self._has_bias:
                bias_grad = self._bias.grad
                np.sum(layer_output.grad, axis=1, out=bias_grad)

    def set_need_backward(self, need_backward):
        if need_backward != self._need_backward:
            Warning("you have changed Linear Layer need backward from {} "
                    "to {}".format(self._need_backward, need_backward))
            self._need_backward = need_backward
            self._weight.set_requires_grad(need_backward)
            if self._has_bias:
                self._bias.set_requires_grad(need_backward)
            self._reset_trainable_parameters()
        else:
            pass

    def get_layer_property(self):
        return {"name": self._name,
                "input_nodes": self._input_nodes,
                "output_nodes": self._output_nodes,
                "has_bias": self._has_bias,
                "need_backward": True}
    
    def _reset_parameters(self):
        self._parameters = dict()
        self._parameters[self._name + "_weight"] = self._weight
        if self._has_bias:
            self._parameters[self._name+"_bias"] = self._bias
    
    def _reset_trainable_parameters(self):
        self._trainable_parameters = dict()
        if self._need_backward:
            if self._weight.requires_grad:
                self._trainable_parameters["weight"] = self._weight
            if self._has_bias:
                if self._bias.requires_grad:
                    self._trainable_parameters["bias"] = self._bias