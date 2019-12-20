# coding=utf-8
from smartnet.optims import *
from smartnet.layers import *
import smartnet as sn
import unittest


class OptimTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    @classmethod
    def setUpClass(cls):
        print("optim test begins.")

    @classmethod
    def tearDownClass(cls):
        print("optim test finished.")

    @staticmethod
    def test_sgd():
        x = sn.random((30, 3))
        w = sn.ones((3, 1))
        y = sn.matmul(x, w)

        linear = LinearLayer(3, 1, has_bias=False)
        opt = SGDOptim(linear.named_parameters())
        loss = MSELayer()

        for i in range(1000):
            opt.zero_grad()
            y_hat = linear(x)
            l = loss(y_hat, y)
            l.backward()
            opt.step()
        print("sgd:", linear.named_parameters())

    @staticmethod
    def test_momentum():
        x = sn.random((30, 3))
        w = sn.ones((3, 1))
        y = sn.matmul(x, w)

        linear = LinearLayer(3, 1, has_bias=False)
        opt = MomentumOptim(linear.named_parameters(), lr=0.001)
        loss = MSELayer()

        for i in range(1000):
            opt.zero_grad()
            y_hat = linear(x)
            l = loss(y_hat, y)
            l.backward()
            opt.step()
        print("momentum:", linear.named_parameters())

    @staticmethod
    def test_rmsprop():
        x = sn.random((30, 3))
        w = sn.ones((3, 1))
        y = sn.matmul(x, w)

        linear = LinearLayer(3, 1, has_bias=False)
        opt = RMSPropOptim(linear.named_parameters())
        loss = MSELayer()

        for i in range(1000):
            opt.zero_grad()
            y_hat = linear(x)
            l = loss(y_hat, y)
            l.backward()
            opt.step()
        print("rmsprop:", linear.named_parameters())

    @staticmethod
    def test_adam():
        x = sn.random((30, 3))
        w = sn.ones((3, 1))
        y = sn.matmul(x, w)

        linear = LinearLayer(3, 1, has_bias=False)
        opt = AdamOptim(linear.named_parameters())
        loss = MSELayer()

        for i in range(1000):
            opt.zero_grad()
            y_hat = linear(x)
            l = loss(y_hat, y)
            l.backward()
            opt.step()
        print("adam:", linear.named_parameters())


if __name__ == "__main__":
    unittest.main()