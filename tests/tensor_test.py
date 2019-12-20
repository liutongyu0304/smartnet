# coding=utf-8
import smartnet as sn
import unittest
import numpy as np


class TensorTest(unittest.TestCase):
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
        a = sn.zeros((3, 4), requires_grad=True)
        a = a + 1.0
        a = 2.0 + a
        b = sn.ones((3, 1))
        b = b + a
        c = sn.ones((1, 4))
        c = c + b
        d = sn.random((3, 4))
        d = d + c
        e = d.sum()
        e.backward()

    @staticmethod
    def test_tensor_sub():
        a = sn.zeros((3, 4), requires_grad=True)
        a = a - 1.0
        a = 2.0 - a
        b = sn.ones((3, 1))
        b = b - a
        c = sn.ones((1, 4))
        c = c - b
        d = sn.random((3, 4))
        d = d - c
        e = d.sum()
        e.backward()

    @staticmethod
    def test_tensor_mul():
        a = 0.5 * sn.ones((3, 4), requires_grad=True)
        a = a * 2.0
        b = sn.random((3, 1))
        c = b * a
        d = sn.random((1, 4))
        d = d * c
        e = d.sum()
        e.backward()

    @staticmethod
    def test_tensor_div():
        a = 0.5 / sn.ones((3, 4), requires_grad=True)
        a = a / 2.0
        b = sn.random((3, 1))
        c = b / a
        d = sn.random((1, 4))
        d = d / c
        e = d.sum()
        e.backward()

    @staticmethod
    def test_tensor_exp():
        a = sn.ones((3, 4), requires_grad=True)
        b = sn.exp(a)
        c = b.sum()
        c.backward()

    @staticmethod
    def test_tensor_pow():
        a = sn.ones((3, 4), requires_grad=True)
        b = sn.pow(a, 2)
        c = b.sum()
        c.backward()

    @staticmethod
    def test_tensor_log():
        a = sn.ones((3, 4), requires_grad=True)
        b = sn.log(a)
        c = b.sum()
        c.backward()

    @staticmethod
    def test_tensor_matmul():
        a = sn.ones((3, 4), requires_grad=True)
        b = sn.ones((4, 3), requires_grad=True)
        c = sn.matmul(a, b)
        c.set_retain_grad()
        d = c.sum()
        d.backward()

    @staticmethod
    def test_tensor_sigmoid():
        a = sn.random((3, 4), requires_grad=True)
        b = sn.sigmoid(a)
        b.set_retain_grad()
        c = b.sum()
        c.backward()

    @staticmethod
    def test_tensor_tanh():
        a = sn.random((3, 4), requires_grad=True)
        b = sn.tanh(a)
        b.set_retain_grad()
        c = b.sum()
        c.backward()

    @staticmethod
    def test_tensor_relu():
        a = sn.random((3, 4), requires_grad=True)
        b = sn.relu(a)
        b.set_retain_grad()
        c = b.sum()
        c.backward()

    @staticmethod
    def test_tensor_mse():
        a = sn.random((3, 4), requires_grad=True)
        b = sn.random((3, 4), requires_grad=True)
        c = sn.mse(a, b)
        c.backward()

    @staticmethod
    def test_tensor_cross_entropy():
        a = sn.random((3, 4), requires_grad=True)
        b = sn.zeros((3, 4))
        b.set_values(np.array([[0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]]))
        c = sn.cross_entropy(a, b)
        c.backward()

    @staticmethod
    def test_tensor_opt():
        x = sn.ones((2, 1), requires_grad=True)
        b = sn.ones((2, 1))
        data = b.detach()
        data[0] = 0.5

        for i in range(1000):
            y = sn.sum((x * b - 1)**2)
            y.backward()
            x.update_data(0.01)
            x.zero_grad()
        data = x.to_cpu().detach()
        assert np.linalg.norm(data - np.array([[2], [1]])) < 1e-2

    @staticmethod
    def test_tensor_asstride():
        x = sn.ones((3, 4), requires_grad=True)
        b = x[:, 2]
        c = x[:, 2] * 0.33
        d = c + b
        c = d.sum()
        c.backward()
        print(c, x, x.grad)


if __name__ == "__main__":
    unittest.main()
