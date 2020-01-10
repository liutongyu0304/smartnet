# coding=utf-8
from .tensor import *


def reshape(t, shape):
    return t.reshape(shape)


def transpose(t):
    return t.transpose()


def add(left, right):
    return left + right


def minus(left, right):
    return left - right


def mul(left, right):
    return left * right


def divide(left, right):
    return left / right


def matmul(left, right):
    return left.matmul(right)


def exp(t):
    return t.exp()


def log(t):
    return t.log()


def sum(t, axis=None, keepdims=False):
    return t.sum(axis, keepdims)


def pow(t, n):
    return t ** n


def sigmoid(t):
    from .op import SigmoidOperation
    return SigmoidOperation()(t)


def tanh(t):
    from .op import TanhOperation
    return TanhOperation()(t)


def relu(t):
    from .op import ReluOperation
    return ReluOperation()(t)


def mse(left, right):
    from .op import MSEOperation
    return MSEOperation()(left, right)


def cross_entropy(left, right):
    from .op import CrossEntropyOperation
    return CrossEntropyOperation()(left, right)


def drop_out(t, keep_probs=0.5):
    from .op import DropOutOperation
    return DropOutOperation(keep_probs=keep_probs)(t)


def clip(t, min_value, max_value):
    from .op import ClipOperation
    return ClipOperation()(t, min_value, max_value)
