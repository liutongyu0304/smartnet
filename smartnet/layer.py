# coding=utf-8
from .tensor import *


class SmartLayer(object):
    def __init__(self, name, need_backward=True):
        self._name = name
        self._inputs = None
        self._outputs = None
        self._parameters = dict()
        self._trainable_parameters = dict()
        self._need_backward = need_backward

        self._previous_inputs = 1
        self._outside_inputs = 0

    def set_up_layer(self, inputs):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def get_layer_property(self):
        return dict()

    def zero_grad(self):
        """
        # description
            clear all the gradients of parameters,inputs and outputs
        """
        for par in self._trainable_parameters.values():
            par.zero_grad()
        for layer_input in self._inputs:
            layer_input.zero_grad()
        for layer_output in self._outputs:
            layer_output.zero_grad()

    def set_need_backward(self, need_backward):
        if need_backward != self._need_backward:
            Warning("you have changed Layer need backward from {} "
                    "to {}".format(self._need_backward, need_backward))
            self._need_backward = need_backward
        else:
            pass

    @property
    def need_backward(self):
        return self._need_backward

    @property
    def name(self):
        return self._name

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def parameters(self):
        return self._parameters

    @property
    def trainable_parameters(self):
        return self._trainable_parameters

    @property
    def previous_inputs(self):
        return self._previous_inputs

    @property
    def outside_inputs(self):
        return self._outside_inputs
