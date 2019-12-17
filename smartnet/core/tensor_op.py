# coding=utf-8
from .tensor import *


class TensorOp(object):
    @staticmethod
    def zeros(shape, device="cpu", dtype=np.float32, requires_grad=False):
        tensor = SmartTensor(shape, device=device, dtype=dtype, requires_grad=requires_grad)
        return tensor

    @staticmethod
    def zeros_like(tensor):
        return SmartTensor(tensor.shape, device=tensor.device,
                           dtype=tensor.dtype, requires_grad=tensor.requires_grad)

    @staticmethod
    def ones(shape, device="cpu", dtype=np.float32, requires_grad=False):
        tensor = SmartTensor(shape, device=device,
                             dtype=dtype, requires_grad=requires_grad)
        tensor.data.set_values(1.0)
        return tensor

    @staticmethod
    def ones_like(tensor):
        return TensorOp.ones(tensor.shape, device=tensor.device,
                             dtype=tensor.dtype, requires_grad=tensor.requires_grad)

    @staticmethod
    def random(shape, device="cpu", dtype=np.float32, requires_grad=False):
        tensor = SmartTensor(shape, device=device,
                             dtype=dtype, requires_grad=requires_grad)
        sp = get_package_by_device(device)
        tensor.data.set_values(sp.random.rand(*shape))
        return tensor

    @staticmethod
    def reshape(tensor, shape):
        return tensor.reshape(shape)

    @staticmethod
    def transpose(tensor):
        return tensor.transpose()

    @staticmethod
    def add(left, right):
        return left + right

    @staticmethod
    def minus(left, right):
        return left - right

    @staticmethod
    def mul(left, right):
        return left * right

    @staticmethod
    def divide(left, right):
        return left / right

    @staticmethod
    def matmul(left, right):
        return left.matmul(right)

    @staticmethod
    def exp(tensor):
        return tensor.exp()

    @staticmethod
    def log(tensor):
        return tensor.log()

    @staticmethod
    def sum(tensor, axis=None, keepdims=False):
        return tensor.sum(axis, keepdims)

    @staticmethod
    def pow(tensor, n):
        return tensor**n

    @staticmethod
    def sigmoid(tensor):
        from .op import SigmoidOperation
        return SigmoidOperation()(tensor)

    @staticmethod
    def tanh(tensor):
        from .op import TanhOperation
        return TanhOperation()(tensor)

    @staticmethod
    def relu(tensor):
        from .op import ReluOperation
        return ReluOperation()(tensor)

    @staticmethod
    def mse(left, right):
        from .op import MSEOperation
        return MSEOperation()(left, right)

    @staticmethod
    def cross_entropy(left, right):
        from .op import CrossEntropyOperation
        return CrossEntropyOperation()(left, right)
