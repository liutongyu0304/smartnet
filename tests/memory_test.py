# coding=utf-8
import smartnet as sn
import unittest


class MemoryTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    @classmethod
    def setUpClass(cls):
        print("memory test begins.")

    @classmethod
    def tearDownClass(cls):
        print("memory test finished.")

    @staticmethod
    def test_cpu():
        x = sn.zeros((3, 4))
        x.to_gpu()

    @staticmethod
    def test_gpu():
        x = sn.zeros((3, 4))
        x.to_cpu()

    @staticmethod
    def test_run_on_gpu():
        x = sn.random((3, 4), device="cuda", requires_grad=True)
        y = x * 2 + 1
        z = sn.sum(y)
        z = z * 2
        z.backward()


if __name__ == "__main__":
    unittest.main()
