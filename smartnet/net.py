# coding=utf-8
from .layer import SmartLayer
from .tensor import *


class SmartNet(object):
    def __init__(self, inputs):
        self._layers = list()

    def add_layer(self, layer, inputs=None):
        assert isinstance(layer, SmartLayer)
        for layer_ in self._layers:
            if layer.name == layer_.name:
                raise Exception("{} already exists in net".format(layer.name))

        if inputs is not None and not isinstance(inputs, list):
            inputs = list(inputs)

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
        return self._layers[-1].outputs[0][0]

    def backward(self):
        self._layers.reverse()
        for layer in self._layers:
            layer.backward()
        self._layers.reverse()

    def zero_grad(self):
        for layer in self._layers:
            layer.zero_grad()

    def parameters(self):
        pars = dict()
        for layer in self._layers:
            for name, par in layer.paramerters:
                if name not in pars.keys():
                    pars[name] = par
        return pars

    def trainable_parameters(self):
        pars = dict()
        for layer in self._layers:
            for name, par in layer.trainable_paramerters:
                if name not in pars.keys():
                    pars[name] = par
        return pars