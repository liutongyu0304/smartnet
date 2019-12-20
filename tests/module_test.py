# coding=utf-8
from smartnet.module import *
import unittest


class Module1(Module):
    def __init__(self):
        super(Module1, self).__init__()
        self.seq1 = Tensor((3, 4))
        self.l1 = Tensor((3, 4))

    def forward(self, *inputs, **kwargs):
        pass


class Module2(Module):
    def __init__(self):
        super(Module2, self).__init__()
        self.seq2 = Tensor((3, 4))
        self.l2 = Tensor((3, 4))
        self.module1 = Module1()

    def forward(self, *inputs, **kwargs):
        pass


class Module3(Module):
    def __init__(self):
        super(Module3, self).__init__()
        self.seq3 = Tensor((3, 4))
        self.l3 = Tensor((3, 4))
        self.module2 = Module2()

    def forward(self, *inputs, **kwargs):
        pass


class ModuleTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    @classmethod
    def setUpClass(cls):
        print("module test begins.")

    @classmethod
    def tearDownClass(cls):
        print("module test finished.")

    @staticmethod
    def test_module():
        module = Module3()
        print(module.named_modules())
        print(module.named_parameters())
        print(module.total_trainable_size())


if __name__ == "__main__":
    unittest.main()
