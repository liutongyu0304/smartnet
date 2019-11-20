# coding=utf-8
from .layer import SmartLayer
from .tensor import *


class SmartNet(object):
    def __init__(self):
        self._layers = list()

    def add_layer(self, layer, inputs=None):
        assert isinstance(layer, SmartLayer)
        # allow the same layer in different position
        # for layer_ in self._layers:
        #     if layer.name == layer_.name:
        #         raise Exception("{} already exists in net".format(layer.name))

        if inputs is not None and not isinstance(inputs, list):
            inputs = [inputs]

        if len(self._layers) == 0:
            self._layers.append(layer)
            layer.set_up_layer(inputs)
        else:
            input_layer = self._layers[-1]
            layer_inputs = list()
            for i in range(layer.previous_inputs):
                layer_inputs.append(input_layer.outputs[i])
            for i in range(layer.outside_inputs):
                layer_inputs.append(inputs[i])

            layer.set_up_layer(layer_inputs)
            self._layers.append(layer)

    def forward(self):
        for layer in self._layers:
            layer.forward()
        return self._layers[-1].outputs[0].data[0]

    def backward(self):
        end_backward = -1
        for i, layer in enumerate(self._layers):
            if layer.need_backward:
                end_backward = i
                break

        n = len(self._layers)
        for i in range(n):
            k = n - i - 1
            if k < end_backward:
                break
            layer = self._layers[k]
            layer.backward()

    def zero_grad(self):
        for layer in self._layers:
            layer.zero_grad()

    def parameters(self):
        # allow duplicate parameters
        pars = list()
        for layer in self._layers:
            for name, par in layer.parameters.items():
                pars.append({"name": name, "parameter": par})
        return pars

    def trainable_parameters(self):
        # allow duplicate parameters
        pars = list()
        for layer in self._layers:
            for name, par in layer.trainable_parameters.items():
                pars.append({"name": name, "parameter": par})
        return pars

    def get_layer(self, name):
        for layer in self._layers:
            if layer.name == name:
                return layer
        return None

    def total_trainable_size(self):
        total = 0
        for layer in self._layers:
            total += layer.total_trainable_size()
        return total

    def total_layers(self):
        return len(self._layers)

    @property
    def layers(self):
        return self._layers
