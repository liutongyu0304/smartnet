# coding=utf-8
from .tensor import *


class SmartOperation(object):
    def __init__(self, name):
        self._name = name
        self._inputs = list()
        self._output = None
    
    def forward(self, *args):
        raise NotImplementedError()

    def backward(self):
        raise NotImplementedError()

    def __call__(self, *args):
        self._output = self.forward(*args)
        for t in self._inputs:
            if isinstance(t, SmartTensor):
                if t.requires_grad:
                    self._output.set_requires_grad(True)
                    break
        self._output.set_op(self)
        self._output.set_leaf(False)
        

class ReshapeOperation(SmartOperation):
    def __init__(self):
        super(ReshapeOperation, self).__init__("ReshapeOperation")
    
    def forward(self, *args):
        assert len(args) == 2
        self._inputs = args
        assert isinstance(self._inputs[0], SmartTensor)
        return self._inputs[0].reshape(self._inputs[1])
    
    def backward(self):
        self._inputs[0].make_grad()
        storage = self._output.grad.reshape(self._inputs[0].shape)
        data = self._inputs[0].grad.data
        data[:] = storage.data


class TransposeOperation(SmartOperation):
    def __init__(self):
        super(TransposeOperation, self).__init__("TransposeOperation")
    
    def forward(self, *args):
        assert len(args) == 1
        self._inputs = args
        assert isinstance(self._inputs[0], SmartTensor)
        return self._inputs[0].transpose()
    
    def backward(self):
        self._inputs[0].make_grad()
        storage = self._output.grad.transpose()
        data = self._inputs[0].grad.data
        data[:] = storage.data                              


class NegativeOperation(SmartOperation):
    def __init__(self):
        super(NegativeOperation, self).__init__("NegativeOperation")
    
    def forward(self, *args):
        assert len(args) == 1
        self._inputs = args
        assert isinstance(self._inputs[0], SmartTensor)
        return self._inputs[0] * -1
    
    def backward(self):
        self._inputs[0].make_grad()
        data = self._inputs[0].grad.data
        data[:] = self._output.grad.data


class AddOperation(SmartOperation):
    def __init__(self):
        super(AddOperation, self).__init__("AddOpeartion")
    
    def forward(self, *args):
        assert len(args) == 2
        self._inputs = args
        assert isinstance(self._inputs[0], SmartTensor) or isinstance(self._inputs[1], SmartTensor)

        return self._inputs[0] + self._inputs[1]

    def backward(self):
        self._backward(self._inputs[0])
        self._backward(self._inputs[1])

    def _backward(self, input):
        if isinstance(input, SmartTensor):
            input.make_grad()
            data = input.grad.data
            if input.shape == self._output.shape:
                if input.grad is not None:
                    data[:] = self._output.grad.data
            else:
                for i in range(self._output.shape):
                    if input.shape[i] != self._output.shape[i]:
                        data[:] = self._output.grad.data.sum(axis=i, keepdims=True)
                        break
        

class SubOperation(SmartOperation):
    def __init__(self):
        super(SubOperation, self).__init__("MinusOpeartion")
    
    def forward(self, *args):
        assert len(args) == 2
        self._inputs = args
        assert isinstance(self._inputs[0], SmartTensor) or isinstance(self._inputs[1], SmartTensor)

        return self._inputs[0] + self._inputs[1]

    def backward(self):
        self._backward(self._inputs[0])
        self._backward(self._inputs[1], left=False)

    def _backward(self, input, left=True):
        if isinstance(input, SmartTensor):
            input.make_grad()
            data = input.grad.data
            if input.shape == self._output.shape:
                if input.grad is not None:
                    if left:
                        data[:] = self._output.grad.data
                    else:
                        data[:] = -self._output.grad.data
            else:
                for i in range(self._output.shape):
                    if input.shape[i] != self._output.shape[i]:
                        if left:
                            data[:] = self._output.grad.data.sum(axis=i, keepdims=True)
                        else:
                            data[:] = -self._output.grad.data.sum(axis=i, keepdims=True)
                        break    


class MulOperation(SmartOperation):
    def __init__(self):
        super(MulOperation, self).__init__("MulOpeartion")
    
    def forward(self, *args):
        assert len(args) == 2
        self._inputs = args
        assert isinstance(self._inputs[0], SmartTensor) or isinstance(self._inputs[1], SmartTensor)

        return self._inputs[0] * self._inputs[1]

    def backward(self):
        self._backward(self._inputs[0], self._inputs[1])
        self._backward(self._inputs[1], self._inputs[0])

    def _backward(self, input, input1):
        # input * input1
        # input.grad
        if isinstance(input, SmartTensor):
            input.make_grad()
            data = input.grad.data
            if isinstance(input1, SmartTensor):
                storage = self._output.grad * input1.data
            else:
                storage = self._output.grad * input1
            data[:] = storage.data


class DivideOperation(SmartOperation):
    def __init__(self):
        super(DivideOperation, self).__init__("DivideOperation")
    
    def forward(self, *args):
        assert len(args) == 2
        self._inputs = args
        assert isinstance(self._inputs[0], SmartTensor) or isinstance(self._inputs[1], SmartTensor)

        return self._inputs[0] / self._inputs[1]

    def backward(self):
        self._backward_left(self._inputs[0], self._inputs[1])
        self._backward_right(self._inputs[0], self._inputs[1])

    def _backward_left(self, input, input1):
        # input / input1
        # input.grad
        if isinstance(input, SmartTensor):
            input.make_grad()
            data = input.grad.data
            if isinstance(input1, SmartTensor):
                storage = self._output.grad / input1.data
            else:
                storage = self._output.grad / input1
            data[:] = storage.data

    def _backward_right(self, input, input1):
        # input / input1
        # input1.grad
        if isinstance(input1, SmartTensor):
            input1.make_grad()
            data = input1.grad.data
            if isinstance(input1, SmartTensor):
                storage = self._output.grad * input / input1.data**2
            else:
                storage = self._output.grad * input / input1**2
            data[:] = -storage.data


class MatmulOperation(SmartOperation):
    def __init__(self):
        super(MatmulOperation, self).__init__("MatmulOperation")
    
    def forward(self, *args):
        assert len(args) == 2
        self._inputs = args
        assert isinstance(self._inputs[0], SmartTensor) and isinstance(self._inputs[1], SmartTensor)

        return self._inputs[0].matmul(self._inputs[1])

    def backward(self):
        self._backward_left(self._inputs[0], self._inputs[1])
        self._backward_right(self._inputs[0], self._inputs[1])

    def _backward_left(self, input, input1):
        # input.matmul(input1)
        # input.grad
        if isinstance(input, SmartTensor):
            input.make_grad()
            data = input.grad.data
            if isinstance(input1, SmartTensor):
                storage = self._output.grad.matmul(input1.data.transpose())   
            else:
                storage = self._output.grad.matmul(input1.data.transpose())
            data[:] = storage.data

    def _backward_right(self, input, input1):
        # input.matmul(input1)
        # input1.grad
        if isinstance(input1, SmartTensor):
            input1.make_grad()
            data = input1.grad.data
            if isinstance(input, SmartTensor):
                storage = input.data.transpose().matmul(self._output.grad)   
            else:
                storage = input.transpose().matmul(self._output.grad)
            data[:] = storage.data 


class ExpOperation(SmartOperation):
    def __init__(self):
        super(ExpOperation, self).__init__("ExpOperation")

    def forward(self, *args):
        return

    def backward(self):
        input = self._inputs[0]
        assert isinstance(input, SmartTensor)
        input.make_grad()
        data = input.grad.data
        storage = self._output.grad * self._output.data
        data[:] = storage.data


class LogOperation(SmartOperation):
    def __init__(self):
        super(LogOperation, self).__init__("LogOperation")

    def backward(self):
        input = self._inputs[0]
        assert isinstance(input, SmartTensor)
        input.make_grad()
        data = input.grad.data
        storage = self._output.grad / input.data
        data[:] = storage.data
