# coding=utf-8
from .layer import SmartLayer
from .tensor import *


class SmartOptim(object):
    def __init__(self, name, trainable_parameters):
        self._name = name
        self._trainable_parameters = trainable_parameters

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for par in self._trainable_parameters.values():
            par.zero_grad()
