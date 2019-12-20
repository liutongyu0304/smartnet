# coding=utf-8
from .util import *


class Tensor(object):
    """
    # description:
        tensor class based on numpy in cpu and cupy in gpu .
        tensor is the base of auto-grad and deep learning algorithm.
        conditions that need/keep gradient of tensor:
        1.any tensor that requires_grad is true needs gradient to be computed
        2.for non-leaf tensor, if any input that requires_grad is true,
          the non-leaf tensor's gradient should be computed.
        3.non-leaf's gradient does not retain unless its retain_grad(default false) is true.
    # members:
        data: np.ndarray or cp.ndarray
            values of tensor, use detch() or data() function to get data of tensor without tracking,
            but this is dangerous because no gradient is computed.
        grad: np.ndarray or cp.ndarray
            gradient of tensor, dimensions must be the same with data
        is_leaf: bool
            whether tensor is leaaf or not
        requires_grad: bool
            whether tensor requires grad, default false
        retain_grad: bool
            whether retain gradient after backward for non-leaf tensor
        op: Operation
            operation function that make the tensor, op is none for leaf tensor.
            op is very important for backward, any op should remember its inputs,
            doing forward and backward operation.
        pkg: generator package of data, numpy with device cpu or cupy with device gpu
    """
    def __init__(self, shape=None, data=None, device="cpu", dtype=np.float32, requires_grad=False):
        if data is None and shape is None:
            raise ValueError("either shape or data should be not None")
        if data is None:
            self._device = device
            self._data = self.pkg.zeros(shape, dtype=dtype)
        else:
            if isinstance(data, cp.ndarray):
                self._data = data
                self._device = "cuda"
            else:
                self._device = "cpu"
                self._data = self.pkg.asarray(data)
        self._grad = None
        self._requires_grad = requires_grad
        self._is_leaf = True
        self._retain_grad = False
        self._op = None
        from .graph import get_graph
        get_graph().add_tensor(self)
    
    def set_requires_grad(self, requires_grad=True):
        if not self._is_leaf:
            raise Exception("non leaf tensor can not be set requires_grad.")
        self._requires_grad = requires_grad

    def _set_leaf(self, is_leaf=True):
        """
        # description:
            set leaf or non leaf tensor, for generated tensor that op is not None
            users should not use this function.
        """
        self._is_leaf = is_leaf

    def set_retain_grad(self):
        """
        # description:
            set retain grad property for non leaf tensor, this function does not work
            for leaf tensor.
        """
        self._retain_grad = True

    def _set_op(self, op):
        self._op = op

    def set_values(self, data):
        # this will change tensor data without changing grad,
        # do not use it except constucting a new tensor
        if isinstance(data, Tensor):
            self._data[:] = data.data
        else:
            self._data[:] = data

    def update_data(self, lr):
        assert self._requires_grad
        if self._data is None or self._grad is None:
            return
        self._data = self._data - lr * self._grad

    def update_grad(self, grad):
        if self._grad is not None:
            self._grad = self._grad + grad
    
    def make_grad(self):
        if self._grad is None:
            self._grad = self.pkg.zeros_like(self._data)

    def zero_grad(self):
        if self._grad is not None:
            self._grad[:] = 0.0

    def clear_grad(self):
        self._grad = None

    def detach(self):
        return self._data

    def item(self):
        if self.size != 1:
            raise RuntimeError("tensor can only convert to a python scaler when its size=1")
        return self._data.item()

    def to_cpu(self):
        self._data = to_cpu(self._data)
        if self._grad is not None:
            self._grad = to_cpu(self._grad)
        return self

    def to_gpu(self):
        self._data = to_gpu(self._data)
        if self._grad is not None:
            self._grad = to_gpu(self._grad)
        return self

    def backward(self, retain_graph=False):
        from .graph import get_graph
        get_graph().auto_grad(self, retain_graph=retain_graph)

    def __getitem__(self, item):
        from .op import AsStrideOption
        return AsStrideOption()(self, item)

    def __str__(self):
        s = "Tensor shape: {}, device: {}, dtype: {},\n" \
            "data: {}\n".format(self.shape, self.device, self.dtype, self._data)
        return s

    def __repr__(self):
        return self.__str__()

    def __neg__(self):
        from .op import NegativeOperation
        return NegativeOperation()(self)

    def __add__(self, right):
        from .op import AddOperation
        return AddOperation()(self, right)
    
    def __radd__(self, left):
        from .op import AddOperation
        return AddOperation()(left, self)

    def __sub__(self, right):
        from .op import SubOperation
        return SubOperation()(self, right)

    def __rsub__(self, left):
        from .op import SubOperation
        return SubOperation()(left, self)

    def __mul__(self, right):
        from .op import MulOperation
        return MulOperation()(self, right)

    def __rmul__(self, left):
        from .op import MulOperation
        return MulOperation()(left, self)

    def __truediv__(self, right):
        from .op import DivideOperation
        return DivideOperation()(self, right)
    
    def __rtruediv__(self, left):
        from .op import DivideOperation
        return DivideOperation()(left, self)

    def __pow__(self, right):
        from .op import PowOperation
        return PowOperation()(self, right)

    def reshape(self, shape):
        from .op import ReshapeOperation
        return ReshapeOperation()(self, shape)

    def transpose(self):
        from .op import TransposeOperation
        return TransposeOperation()(self)

    def sum(self, axis=None, keepdims=True):
        from .op import SumOperation
        return SumOperation(axis, keepdims)(self)

    def matmul(self, right):
        from .op import MatmulOperation
        return MatmulOperation()(self, right)

    def exp(self):
        from .op import ExpOperation
        return ExpOperation()(self)

    def log(self):
        from .op import LogOperation
        return LogOperation()(self)

    @property
    def requires_grad(self):
        return self._requires_grad

    @property
    def data(self):
        return self._data

    @property
    def grad(self):
        return self._grad

    @property
    def shape(self):
        return self._data.shape

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def is_leaf(self):
        return self._is_leaf

    @property
    def retain_grad(self):
        return self._retain_grad

    @property
    def op(self):
        return self._op

    @property
    def size(self):
        return self._data.size

    @property
    def pkg(self):
        return get_package_by_device(self._device)