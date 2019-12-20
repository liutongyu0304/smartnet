# coding=utf-8
from .layers import *
from .core import *


class Optim(object):
    def __init__(self, name, trainable_parameters):
        self._name = name
        self._trainable_parameters = trainable_parameters

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for value in self._trainable_parameters.values():
            value.zero_grad()

    def get_property(self):
        return dict()

    @property
    def name(self):
        return self._name
