# coding=utf-8
from ..layer import SmartLayer
from ..tensor import *


class SmartLinearLayer(SmartLayer):
    """
    # description
        linear(full connected) layer.
        forward of linear layer.
        for the nth layer:
            z(n) = a(n-1) * w(n) + b(n)
        z(n): the input of the nth layer, shape of (m, l(n)),
              where l(n) is the nodes of the nth layer. m is the number of samples
        a(n-1): the output of the (n-1)th layer, shape of (m, l(n-1)),
              where l(n-1) is the nodes of the (n-1)th layer. if the (n-1) layer is cnn
              layer, then a(n-1) is reshaped from (m, h, w, c) to (m, h*w*c).
        w(n): weight of the linear layer, shape of (l(n-1), l(n)), to be trained.
        b(n): bias of the linear layer, shape of (l(n), ), to be trained if _has_bias is True.
        notice that while doing add operation with b(n), a broadcasting operation of b(n) is done.

        backward of linear layer.
        for simplification, remove subscript of z, a, w, b. dz, da, dw, dx means
        gradient of loss to x. and dz is known from next layer.
            da = dz * w.t
            dw = a.t * dz
            db = sum(dz, axis=0)
    # members
        _input_nodes: int
            nodes of (n-1)th layer
        _output_nodes: int
            nodes of nth layer
        _has_bias: bool
            whether linear layer has bias
        _weight: SmartTensor
            weight of linear layer
        _bias: SmartTensor
            bias of linear layer
    """
    def __init__(self, name, input_nodes, output_nodes, has_bias=True, need_backward=True):
        super(SmartLinearLayer, self).__init__(name, need_backward)
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._has_bias = has_bias

        # initialized with need_backward
        self._weight = SmartTensor(None, requires_grad=need_backward)
        self._bias = SmartTensor(None, requires_grad=need_backward)

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
        # doing reshape operation for cnn layer with four dimensions
        # layer_input_ and layer_input still share the same memory
        layer_input_ = layer_input.reshape((-1, self._input_nodes))

        layer_output = SmartTensor(np.zeros((layer_input_.shape[0], self._output_nodes)))
        self._weight = SmartTensor(np.random.rand(self._input_nodes, self._output_nodes))
        if self._has_bias:
            self._bias = SmartTensor(np.random.rand(self._output_nodes))
        self._reset_parameters()
        self._reset_trainable_parameters()

        self._inputs = [layer_input_]
        self._outputs = [layer_output]

    def forward(self):
        layer_input = self._inputs[0]
        layer_output = self._outputs[0]
        # z(n) = a(n-1) * w(n) + b(n)
        data = layer_output.data
        np.matmul(layer_input.data, self._weight.data, data)
        if self._has_bias:
            data += self._bias.data

    def backward(self):
        # if not self._need_backward:
        #     raise Exception("linear layer {} has property need backward False")
        layer_input = self._inputs[0]
        layer_output = self._outputs[0]
        # da += dz * w.t
        grad = layer_input.grad.copy()
        np.matmul(layer_output.grad, self._weight.data.transpose(), grad)
        layer_input.grad[:] = layer_input.grad + grad

        if self._need_backward:
            # dw += a.t * dz
            weight_grad = self._weight.grad.copy()
            np.matmul(layer_input.data.transpose(), layer_output.grad, weight_grad)
            self._weight.grad[:] = self._weight.grad + weight_grad
            if self._has_bias:
                bias_grad = self._bias.grad.copy()
                np.sum(layer_output.grad, axis=0, out=bias_grad)
                self._bias.grad[:] = self._bias.grad + bias_grad

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
                "need_backward": self._need_backward}
    
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

    @property
    def weight(self):
        return self._weight

    @property
    def bias(self):
        return self._bias