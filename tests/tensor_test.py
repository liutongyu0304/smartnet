# coding=utf-8
from smartnet.tensor import *


def add_test():
    a = TensorOp.zeros((3, 4), requires_grad=True)
    b = TensorOp.ones((3, 1))
    b = b + a
    c = TensorOp.ones((1, 4)) * 2
    c = c + b
    d = TensorOp.random((3, 4))
    d = d + c
    e = c.sum()
    e.backward()
    print(e)


if __name__ == "__main__":
    add_test()