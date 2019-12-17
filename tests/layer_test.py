# coding=utf-8
import numpy as np
from smartnet.layers import *
from smartnet import TensorOp
import unittest


class SmartLayerTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    @classmethod
    def setUpClass(cls):
        print("layer test begins.")

    @classmethod
    def tearDownClass(cls):
        print("layer test finished.")

    @staticmethod
    def test_linear_layer():
        l1 = SmartLinearLayer(input_nodes=3, output_nodes=1)
        x = TensorOp.random((1, 3))
        x = l1(x)
        loss = x.sum()
        loss.backward()

    @staticmethod
    def test_sigmoid_layer():
        s = SmartSigmoidLayer()
        x = TensorOp.random((3, 4), requires_grad=True)
        x = s(x)
        loss = x.sum()
        loss.backward()

    @staticmethod
    def test_tanh_layer():
        t = SmartTanhLayer()
        x = TensorOp.random((3, 4), requires_grad=True)
        x = t(x)
        loss = x.sum()
        loss.backward()

    @staticmethod
    def test_relu_layer():
        r = SmartSigmoidLayer()
        x = TensorOp.random((3, 4), requires_grad=True)
        x = r(x)
        loss = x.sum()
        loss.backward()

    @staticmethod
    def test_mse_layer():
        x = TensorOp.random((10, 3),requires_grad=True)
        y = TensorOp.random((10, 3))
        mse = SmartMSELayer()
        loss = mse(x, y)
        loss.backward()

    @staticmethod
    def test_cross_entropy_layer():
        x = TensorOp.random((3, 4), requires_grad=True)
        y = TensorOp.zeros((3, 4))
        y.set_values(np.array([[0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]]))
        c = SmartCrossEntropyLayer()
        loss = c(x, y)
        loss.backward()


if __name__ == "__main__":
    unittest.main()