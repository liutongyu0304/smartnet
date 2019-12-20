# coding=utf-8
from ..module import *
from ..import core as Core


class LinearLayer(Module):
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
        b(n): bias of the linear layer, shape of (l(n), 1), to be trained if has_bias is True.
        notice that while doing add operation with b(n), a broadcasting operation of b(n) is done.

        backward of linear layer.
        for simplification, remove subscript of z, a, w, b. dz, da, dw, dx means
        gradient of loss to x. and dz is known from next layer.
            da = dz * w.t
            dw = a.t * dz
            db = sum(dz, axis=0)
    # members
        input_nodes: int
            nodes of (n-1)th layer
        output_nodes: int
            nodes of nth layer
        has_bias: bool
            whether linear layer has bias
        weight: Tensor
            weight of linear layer
        bias: Tensor
            bias of linear layer
    """
    def __init__(self, input_nodes, output_nodes, has_bias=True):
        super(LinearLayer, self).__init__("LinearLayer")
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.has_bias = has_bias

        # initialized with need_backward
        self.weight = Core.random((input_nodes, output_nodes), requires_grad=True)
        if has_bias:
            self.bias = Core.random((1, output_nodes), requires_grad=True)
        else:
            self.bias = None

    def forward(self, *inputs, **kwargs):
        # z(n) = a(n-1) * w(n) + b(n)
        layer_input = inputs[0]
        data = layer_input.matmul(self.weight)
        if self.has_bias:
            data = data + self.bias
        return data

    def get_layer_property(self):
        return {"name": self._name,
                "input_nodes": self.input_nodes,
                "output_nodes": self.output_nodes,
                "has_bias": self.has_bias}
