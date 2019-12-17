# coding=utf-8
import smartnet as sn
import unittest
import numpy as np


class SmartTensorTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    @classmethod
    def setUpClass(cls):
        print("tensor test begins.")

    @classmethod
    def tearDownClass(cls):
        print("tensor test finished.")

    @staticmethod
    def test_tensor_add():
        a = sn.TensorOp.zeros((3, 4), requires_grad=True)
        a = a + 1.0
        a = 2.0 + a
        b = sn.TensorOp.ones((3, 1))
        b = b + a
        c = sn.TensorOp.ones((1, 4))
        c = c + b
        d = sn.TensorOp.random((3, 4))
        d = d + c
        e = d.sum()
        e.backward()

    @staticmethod
    def test_tensor_sub():
        a = sn.TensorOp.zeros((3, 4), requires_grad=True)
        a = a - 1.0
        a = 2.0 - a
        b = sn.TensorOp.ones((3, 1))
        b = b - a
        c = sn.TensorOp.ones((1, 4))
        c = c - b
        d = sn.TensorOp.random((3, 4))
        d = d - c
        e = d.sum()
        e.backward()

    @staticmethod
    def test_tensor_mul():
        a = 0.5 * sn.TensorOp.ones((3, 4), requires_grad=True)
        a = a * 2.0
        b = sn.TensorOp.random((3, 1))
        c = b * a
        d = sn.TensorOp.random((1, 4))
        d = d * c
        e = d.sum()
        e.backward()

    @staticmethod
    def test_tensor_div():
        a = 0.5 / sn.TensorOp.ones((3, 4), requires_grad=True)
        a = a / 2.0
        b = sn.TensorOp.random((3, 1))
        c = b / a
        d = sn.TensorOp.random((1, 4))
        d = d / c
        e = d.sum()
        e.backward()

    @staticmethod
    def test_tensor_exp():
        a = sn.TensorOp.ones((3, 4), requires_grad=True)
        b = sn.TensorOp.exp(a)
        c = b.sum()
        c.backward()

    @staticmethod
    def test_tensor_pow():
        a = sn.TensorOp.ones((3, 4), requires_grad=True)
        b = sn.TensorOp.pow(a, 2)
        c = b.sum()
        c.backward()

    @staticmethod
    def test_tensor_log():
        a = sn.TensorOp.ones((3, 4), requires_grad=True)
        b = sn.TensorOp.log(a)
        c = b.sum()
        c.backward()

    @staticmethod
    def test_tensor_matmul():
        a = sn.TensorOp.ones((3, 4), requires_grad=True)
        b = sn.TensorOp.ones((4, 3), requires_grad=True)
        c = sn.TensorOp.matmul(a, b)
        c.set_retain_grad()
        d = c.sum()
        d.backward()

    @staticmethod
    def test_tensor_sigmoid():
        a = sn.TensorOp.random((3, 4), requires_grad=True)
        b = sn.TensorOp.sigmoid(a)
        b.set_retain_grad()
        c = b.sum()
        c.backward()

    @staticmethod
    def test_tensor_tanh():
        a = sn.TensorOp.random((3, 4), requires_grad=True)
        b = sn.TensorOp.tanh(a)
        b.set_retain_grad()
        c = b.sum()
        c.backward()

    @staticmethod
    def test_tensor_relu():
        a = sn.TensorOp.random((3, 4), requires_grad=True)
        b = sn.TensorOp.relu(a)
        b.set_retain_grad()
        c = b.sum()
        c.backward()

    @staticmethod
    def test_tensor_mse():
        a = sn.TensorOp.random((3, 4), requires_grad=True)
        b = sn.TensorOp.random((3, 4), requires_grad=True)
        c = sn.TensorOp.mse(a, b)
        c.backward()

    @staticmethod
    def test_tensor_cross_entropy():
        a = sn.TensorOp.random((3, 4), requires_grad=True)
        b = sn.TensorOp.zeros((3, 4))
        b.set_values(np.array([[0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]]))
        c = sn.TensorOp.cross_entropy(a, b)
        c.backward()

    @staticmethod
    def test_tensor_opt():
        x = sn.TensorOp.ones((2, 1), requires_grad=True)
        b = sn.TensorOp.ones((2, 1))
        data = b.detach()
        data[0] = 0.5

        for i in range(1000):
            y = sn.TensorOp.sum((x * b - 1)**2)
            y.backward()
            x.update_data(0.01)
            x.zero_grad()
        data = x.to_cpu().detach()
        assert np.linalg.norm(data - np.array([[2], [1]])) < 1e-2


if __name__ == "__main__":
    unittest.main()
