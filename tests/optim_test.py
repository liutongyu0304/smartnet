from smartnet.optims import *
from smartnet import *
import unittest


class SmartOptimTest(unittest.TestCase):
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
        x = TensorOp.random((30, 3))
        w = TensorOp.ones((3, 1))
        y = TensorOp.matmul(x, w)

        linear = SmartLinearLayer(3, 1, has_bias=False)
        opt = SmartSGDOptim(linear.named_parameters())
        loss = SmartMSELayer()

        for i in range(1000):
            opt.zero_grad()
            y_hat = linear(x)
            l = loss(y_hat, y)
            l.backward()
            opt.step()
        print("sgd:", linear.named_parameters())

    @staticmethod
    def test_momentum():
        x = TensorOp.random((30, 3))
        w = TensorOp.ones((3, 1))
        y = TensorOp.matmul(x, w)

        linear = SmartLinearLayer(3, 1, has_bias=False)
        opt = SmartMomentumOptim(linear.named_parameters(), lr=0.001)
        loss = SmartMSELayer()

        for i in range(1000):
            opt.zero_grad()
            y_hat = linear(x)
            l = loss(y_hat, y)
            l.backward()
            opt.step()
        print("momentum:", linear.named_parameters())

    @staticmethod
    def test_rmsprop():
        x = TensorOp.random((30, 3))
        w = TensorOp.ones((3, 1))
        y = TensorOp.matmul(x, w)

        linear = SmartLinearLayer(3, 1, has_bias=False)
        opt = SmartRMSPropOptim(linear.named_parameters())
        loss = SmartMSELayer()

        for i in range(1000):
            opt.zero_grad()
            y_hat = linear(x)
            l = loss(y_hat, y)
            l.backward()
            opt.step()
        print("rmsprop:", linear.named_parameters())

    @staticmethod
    def test_adam():
        x = TensorOp.random((30, 3))
        w = TensorOp.ones((3, 1))
        y = TensorOp.matmul(x, w)

        linear = SmartLinearLayer(3, 1, has_bias=False)
        opt = SmartAdamOptim(linear.named_parameters())
        loss = SmartMSELayer()

        for i in range(1000):
            opt.zero_grad()
            y_hat = linear(x)
            l = loss(y_hat, y)
            l.backward()
            opt.step()
        print("adam:", linear.named_parameters())


if __name__ == "__main__":
    unittest.main()