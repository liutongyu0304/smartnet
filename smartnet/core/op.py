# coding=utf-8
from .tensor import *


class SmartOperation(object):
    def __init__(self, name):
        self._name = name
        self._inputs = list()
        self._output = None
    
    def forward(self, *args):
        result, message = self._check_inputs(*args)
        if not result:
            raise ValueError(message)
        return None

    def backward(self):
        raise NotImplementedError()

    def _check_inputs(self, *args):
        return True, ""

    def __call__(self, *args):
        self._output = self.forward(*args)
        for t in self._inputs:
            if isinstance(t, SmartTensor):
                if t.requires_grad:
                    self._output.set_requires_grad(True)
                    break
        self._output.set_op(self)
        self._output.set_leaf(False)
        return self._output

    @property
    def name(self):
        return self._name

    @property
    def inputs(self):
        return self._inputs

    @property
    def output(self):
        return self._output


class ReshapeOperation(SmartOperation):
    def __init__(self):
        super(ReshapeOperation, self).__init__("ReshapeOperation")
    
    def forward(self, *args):
        super(ReshapeOperation, self).forward(*args)
        shape = self._inputs[0]
        output = SmartTensor(shape, device=self._inputs[0].device,
                             dtype=self._inputs[0].dtype,
                             requires_grad=self._inputs[0].requires_grad)
        storage = self._inputs[0].data.reshape(shape)
        output.set_values(storage)
        return output
    
    def backward(self):
        if not self._inputs[0].requires_grad:
            return
        self._inputs[0].make_grad()
        storage = self._output.grad.reshape(self._inputs[0].shape)
        self._inputs[0].update_grad(storage)

    def _check_inputs(self, *args):
        self._inputs = args
        message = ""
        if len(args) != 2:
            message = "{} should have 2 inputs".format(self._name)
            return False, message
        if not isinstance(self._inputs[0], SmartTensor):
            message = "{} first input's type should be SmartTensor".format(self._name)
            return False, message
        if not isinstance(self._inputs[1], tuple):
            message = "{} shape should be tuple".format(self._name)
            return False, message
        return True, message


class TransposeOperation(SmartOperation):
    def __init__(self):
        super(TransposeOperation, self).__init__("TransposeOperation")
    
    def forward(self, *args):
        super(TransposeOperation, self).forward(*args)
        storage = self._inputs[0].data.transpose()
        output = SmartTensor(storage.shape, device=self._inputs[0].device,
                             dtype=self._inputs[0].dtype,
                             requires_grad=self._inputs[0].requires_grad)
        output.set_values(storage)
        return output
    
    def backward(self):
        if not self._inputs[0].requires_grad:
            return
        self._inputs[0].make_grad()
        storage = self._output.grad.transpose()
        self._inputs[0].update_grad(storage)

    def _check_inputs(self, *args):
        self._inputs = args
        message = ""
        if len(args) != 1:
            message = "{} should have 1 inputs".format(self._name)
            return False, message
        if not isinstance(self._inputs[0], SmartTensor):
            message = "{} to be transposed should be SmartTensor".format(self._name)
            return False, message
        return True, message


class NegativeOperation(SmartOperation):
    def __init__(self):
        super(NegativeOperation, self).__init__("NegativeOperation")
    
    def forward(self, *args):
        super(NegativeOperation, self).forward(*args)
        output = SmartTensor(self._inputs[0].shape, device=self._inputs[0].device,
                             dtype=self._inputs[0].dtype,
                             requires_grad=self._inputs[0].requires_grad)
        output.set_values(-self._inputs[0].data)
        return output
    
    def backward(self):
        if not self._inputs[0].requires_grad:
            return
        self._inputs[0].make_grad()
        self._inputs[0].update_grad(-self._output.grad)

    def _check_inputs(self, *args):
        self._inputs = args
        message = ""
        if len(args) != 1:
            message = "{} should have 1 inputs".format(self._name)
            return False, message
        if not isinstance(self._inputs[0], SmartTensor):
            message = "{} to be transposed should be SmartTensor".format(self._name)
            return False, message
        return True, message


class TwoInputsOperation(SmartOperation):
    def __init__(self, name):
        super(TwoInputsOperation, self).__init__(name)

    def forward(self, *args):
        super(TwoInputsOperation, self).forward(*args)
        requires_grad = False
        if isinstance(self._inputs[0], SmartTensor):
            input0 = self._inputs[0].data
            requires_grad = requires_grad and self._inputs[0].requires_grad
        else:
            input0 = self._inputs[0]

        if isinstance(self._inputs[1], SmartTensor):
            input1 = self._inputs[1].data
            requires_grad = requires_grad and self._inputs[1].requires_grad
        else:
            input1 = self._inputs[1]

        storage = self._operation(input0, input1)
        output = SmartTensor(storage.shape, device=storage.device,
                             dtype=storage.dtype,
                             requires_grad=requires_grad)
        output.set_values(storage)
        return output

    def _operation(self, input0, input1):
        # obviously either input0 or input1 must be SmartStorage
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def _check_inputs(self, *args):
        self._inputs = args
        message = ""
        if len(args) != 2:
            message = "{} should have 2 inputs".format(self._name)
            return False, message
        if not (isinstance(self._inputs[0], SmartTensor) or isinstance(self._inputs[1], SmartTensor)):
            message = "{} should have at least one SmartTensor".format(self._name)
            return False, message
        return True, message


class AddOperation(TwoInputsOperation):
    def __init__(self):
        super(AddOperation, self).__init__("AddOperation")

    def _operation(self, input0, input1):
        return input0 + input1

    def backward(self):
        self._backward(self._inputs[0])
        self._backward(self._inputs[1])

    def _backward(self, input0):
        if isinstance(input0, SmartTensor) and input0.requires_grad:
            input0.make_grad()
            if input0.shape == self._output.shape:
                input0.update_grad(self._output.grad)
            else:
                for i in range(len(self._output.shape)):
                    if input0.shape[i] != self._output.shape[i]:
                        input0.update_grad(self._output.grad.sum(axis=i, keepdims=True))
                        break
        

class SubOperation(TwoInputsOperation):
    def __init__(self):
        super(SubOperation, self).__init__("SubOperation")
    
    def _operation(self, input0, input1):
        return input0 - input1

    def backward(self):
            self._backward(self._inputs[0])
            self._backward(self._inputs[1], left=False)

    def _backward(self, input0, left=True):
        if isinstance(input0, SmartTensor) and input0.requires_grad:
            input0.make_grad()
            if input0.shape == self._output.shape:
                if input0.grad is not None:
                    if left:
                        input0.update_grad(self._output.grad)
                    else:
                        input0.update_grad(-self._output.grad)
            else:
                for i in range(len(self._output.shape)):
                    if input0.shape[i] != self._output.shape[i]:
                        if left:
                            input0.update_grad(self._output.grad.sum(axis=i, keepdims=True))
                        else:
                            input0.update_grad(-self._output.grad.sum(axis=i, keepdims=True))
                        break    


class MulOperation(TwoInputsOperation):
    def __init__(self):
        super(MulOperation, self).__init__("MulOperation")
    
    def _operation(self, input0, input1):
        return input0 * input1

    def backward(self):
        self._backward(self._inputs[0], self._inputs[1])
        self._backward(self._inputs[1], self._inputs[0])

    def _backward(self, input0, input1):
        # input0 * input1
        # input0.grad
        if isinstance(input0, SmartTensor) and input0.requires_grad:
            input0.make_grad()
            data = input0.grad.data
            if isinstance(input1, SmartTensor):
                storage = self._output.grad * input1.data
            else:
                storage = self._output.grad * input1
            if data.shape != storage.data.shape:
                # for broadcast
                reduce_ind = 0
                for i in range(len(data.shape)):
                    if data.shape[i] != storage.data.shape[i]:
                        reduce_ind = i
                        break
                input0.update_grad(storage.sum(axis=reduce_ind, keepdims=True))
            else:
                input0.update_grad(storage)


class DivideOperation(TwoInputsOperation):
    def __init__(self):
        super(DivideOperation, self).__init__("DivideOperation")
    
    def _operation(self, input0, input1):
        return input0 / input1

    def backward(self):
        self._backward_left(self._inputs[0], self._inputs[1])
        self._backward_right(self._inputs[0], self._inputs[1])

    def _backward_left(self, input0, input1):
        # input0 / input1
        # input0.grad
        if isinstance(input0, SmartTensor) and input0.requires_grad:
            input0.make_grad()
            data = input0.grad.data
            if isinstance(input1, SmartTensor):
                storage = self._output.grad / input1.data
            else:
                storage = self._output.grad / input1
            if data.shape != storage.data.shape:
                # for broadcast
                reduce_ind = 0
                for i in range(len(data.shape)):
                    if data.shape[i] != storage.data.shape[i]:
                        reduce_ind = i
                        break
                input0.update_grad(storage.sum(axis=reduce_ind, keepdims=True))
            else:
                input0.update_grad(storage)

    def _backward_right(self, input0, input1):
        # input0 / input1
        # input1.grad
        if isinstance(input1, SmartTensor) and input1.requires_grad:
            input1.make_grad()
            data = input1.grad.data
            if isinstance(input0, SmartTensor):
                storage = self._output.grad * input0.data / input1.data ** 2
            else:
                storage = self._output.grad * input0 / input1.data ** 2
            if data.shape != storage.data.shape:
                # for broadcast
                reduce_ind = 0
                for i in range(len(data.shape)):
                    if data.shape[i] != storage.data.shape[i]:
                        reduce_ind = i
                        break
                input1.update_grad(storage.sum(axis=reduce_ind, keepdims=True))
            else:
                input1.update_grad(storage)


class MatmulOperation(TwoInputsOperation):
    def __init__(self):
        super(MatmulOperation, self).__init__("MatmulOperation")
    
    def _operation(self, input0, input1):
        return input0.matmul(input1)

    def backward(self):
        self._backward_left(self._inputs[0], self._inputs[1])
        self._backward_right(self._inputs[0], self._inputs[1])

    def _backward_left(self, input0, input1):
        # input.matmul(input1)
        # input.grad
        if isinstance(input0, SmartTensor) and input0.requires_grad:
            input0.make_grad()
            if isinstance(input1, SmartTensor):
                storage = self._output.grad.matmul(input1.data.transpose())   
            else:
                storage = self._output.grad.matmul(input1.data.transpose())
            input0.update_grad(storage)

    def _backward_right(self, input0, input1):
        # input.matmul(input1)
        # input1.grad
        if isinstance(input1, SmartTensor) and input1.requires_grad:
            input1.make_grad()
            if isinstance(input0, SmartTensor):
                storage = input0.data.transpose().matmul(self._output.grad)
            else:
                storage = input0.transpose().matmul(self._output.grad)
            input1.update_grad(storage)


class PowOperation(SmartOperation):
    def __init__(self):
        super(PowOperation, self).__init__("PowOperation")

    def forward(self, *args):
        super(PowOperation, self).forward(*args)
        output = SmartTensor(self._inputs[0].shape, device=self._inputs[0].device,
                             dtype=self._inputs[0].dtype,
                             requires_grad=self._inputs[0].requires_grad)
        storage = self._inputs[0].data**self._inputs[1]
        output.set_values(storage)
        return output

    def backward(self):
        if not self._inputs[0].requires_grad:
            return
        input0 = self._inputs[0]
        input0.make_grad()
        storage = self._output.grad * self._inputs[1] * self._inputs[0].data**(self._inputs[1] - 1)
        input0.update_grad(storage)

    def _check_inputs(self, *args):
        self._inputs = args
        message = ""
        if len(args) != 2:
            message = "{} should have 2 inputs".format(self._name)
            return False, message
        if not isinstance(self._inputs[0], SmartTensor):
            message = "{} first input's type should be SmartTensor".format(self._name)
            return False, message
        if "int" not in str(type(self._inputs[1])) and "float" not in str(type(self._inputs[1])):
            message = "{} second input's type should be int/float".format(self._name)
            return False, message
        return True, message


class ExpOperation(SmartOperation):
    def __init__(self):
        super(ExpOperation, self).__init__("ExpOperation")

    def forward(self, *args):
        super(ExpOperation, self).forward(*args)
        output = SmartTensor(self._inputs[0].shape, device=self._inputs[0].device,
                             dtype=self._inputs[0].dtype,
                             requires_grad=self._inputs[0].requires_grad)
        storage = self._inputs[0].data.exp()
        output.set_values(storage)
        return output

    def backward(self):
        if not self._inputs[0].requires_grad:
            return
        input0 = self._inputs[0]
        input0.make_grad()
        storage = self._output.grad * self._output.data
        input0.update_grad(storage)

    def _check_inputs(self, *args):
        self._inputs = args
        message = ""
        if len(args) != 1:
            message = "{} should have 1 inputs".format(self._name)
            return False, message
        if not isinstance(self._inputs[0], SmartTensor):
            message = "{} input should be SmartTensor".format(self._name)
            return False, message
        return True, message


class LogOperation(SmartOperation):
    def __init__(self):
        super(LogOperation, self).__init__("LogOperation")

    def forward(self, *args):
        super(LogOperation, self).forward(*args)
        output = SmartTensor(self._inputs[0].shape, device=self._inputs[0].device,
                             dtype=self._inputs[0].dtype,
                             requires_grad=self._inputs[0].requires_grad)
        storage = self._inputs[0].data.log()
        output.set_values(storage)
        return output

    def backward(self):
        if not self._inputs[0].requires_grad:
            return
        input0 = self._inputs[0]
        input0.make_grad()
        storage = self._output.grad / input0.data
        input0.update_grad(storage)

    def _check_inputs(self, *args):
        self._inputs = args
        message = ""
        if len(args) != 1:
            message = "{} should have 1 inputs".format(self._name)
            return False, message
        if not isinstance(self._inputs[0], SmartTensor):
            message = "{} input should be SmartTensor".format(self._name)
            return False, message
        return True, message


class SumOperation(SmartOperation):
    def __init__(self, axis=None, keepdims=True):
        super(SumOperation, self).__init__("SumOperation")
        self._axis = axis
        self._keepdims = keepdims

    def forward(self, *args):
        super(SumOperation, self).forward(*args)
        storage = StorageOp.sum(self._inputs[0].data, self._axis, self._keepdims)
        output = SmartTensor(storage.shape, device=self._inputs[0].device,
                             dtype=self._inputs[0].dtype,
                             requires_grad=self._inputs[0].requires_grad)
        output.set_values(storage)
        return output

    def backward(self):
        if not self._inputs[0].requires_grad:
            return
        input0 = self._inputs[0]
        input0.make_grad()
        if self._output.size > 1:
            storage = self._output.grad * StorageOp.ones_like(input0.data)
        else:
            storage = self._output.grad.data[0] * StorageOp.ones_like(input0.data)
        input0.update_grad(storage)

    def _check_inputs(self, *args):
        self._inputs = args
        message = ""
        if len(args) != 1:
            message = "{} should have 1 inputs".format(self._name)
            return False, message
        if not isinstance(self._inputs[0], SmartTensor):
            message = "{} input should be SmartTensor".format(self._name)
            return False, message
        return True, message


class SigmoidOperation(SmartOperation):
    def __init__(self):
        super(SigmoidOperation, self).__init__("SigmoidOperation")

    def forward(self, *args):
        super(SigmoidOperation, self).forward(*args)
        output = SmartTensor(self._inputs[0].shape, device=self._inputs[0].device,
                             dtype=self._inputs[0].dtype,
                             requires_grad=self._inputs[0].requires_grad)
        storage = StorageOp.sigmoid(self._inputs[0].data)
        output.set_values(storage)
        return output

    def backward(self):
        if not self._inputs[0].requires_grad:
            return
        input0 = self._inputs[0]
        input0.make_grad()
        storage = self._output.grad * self._output.data * (1 - self._output.data)
        input0.update_grad(storage)

    def _check_inputs(self, *args):
        self._inputs = args
        message = ""
        if len(args) != 1:
            message = "{} should have 1 inputs".format(self._name)
            return False, message
        if not isinstance(self._inputs[0], SmartTensor):
            message = "{} input should be SmartTensor".format(self._name)
            return False, message
        return True, message


class TanhOperation(SmartOperation):
    def __init__(self):
        super(TanhOperation, self).__init__("TanhOperation")

    def forward(self, *args):
        super(TanhOperation, self).forward(*args)
        output = SmartTensor(self._inputs[0].shape, device=self._inputs[0].device,
                             dtype=self._inputs[0].dtype,
                             requires_grad=self._inputs[0].requires_grad)
        storage = StorageOp.tanh(self._inputs[0].data)
        output.set_values(storage)
        return output

    def backward(self):
        if not self._inputs[0].requires_grad:
            return
        input0 = self._inputs[0]
        input0.make_grad()
        storage = self._output.grad * (1 - self._output.data**2)
        input0.update_grad(storage)

    def _check_inputs(self, *args):
        self._inputs = args
        message = ""
        if len(args) != 1:
            message = "{} should have 1 inputs".format(self._name)
            return False, message
        if not isinstance(self._inputs[0], SmartTensor):
            message = "{} input should be SmartTensor".format(self._name)
            return False, message
        return True, message


class ReluOperation(SmartOperation):
    def __init__(self):
        super(ReluOperation, self).__init__("ReluOperation")

    def forward(self, *args):
        super(ReluOperation, self).forward(*args)
        output = SmartTensor(self._inputs[0].shape, device=self._inputs[0].device,
                             dtype=self._inputs[0].dtype,
                             requires_grad=self._inputs[0].requires_grad)
        storage = StorageOp.relu(self._inputs[0].data)
        output.set_values(storage)
        return output

    def backward(self):
        if not self._inputs[0].requires_grad:
            return
        input0 = self._inputs[0]
        input0.make_grad()
        storage = StorageOp.zeros_like(input0.grad)
        storage.set_values(self._output.grad)

        data = storage.data
        data[input0.data.data < 0] = 0.0
        input0.update_grad(storage)

    def _check_inputs(self, *args):
        self._inputs = args
        message = ""
        if len(args) != 1:
            message = "{} should have 1 inputs".format(self._name)
            return False, message
        if not isinstance(self._inputs[0], SmartTensor):
            message = "{} input should be SmartTensor".format(self._name)
            return False, message
        return True, message


class MSEOperation(TwoInputsOperation):
    def __init__(self):
        super(MSEOperation, self).__init__("MSEOperation")

    def _operation(self, input0, input1):
        return StorageOp.mse(input0, input1)

    def backward(self):
        self._backward(self._inputs[0], self._inputs[1])
        self._backward(self._inputs[1], self._inputs[0])

    def _backward(self, input0, input1):
        if isinstance(input0, SmartTensor) and input0.requires_grad:
            if isinstance(input1, SmartTensor):
                storage = input0.data - input1.data
            else:
                storage = input0.data - input1
            storage = self._output.grad * 2.0 * storage / storage.size
            input0.update_grad(storage)


class CrossEntropyOperation(TwoInputsOperation):
    def __init__(self):
        super(CrossEntropyOperation, self).__init__("CrossEntropyOperation")
        self._soft_max_output = None

    def _operation(self, input0, input1):
        layer_input = input0
        label = input1

        self._soft_max_output = StorageOp.zeros_like(input0)

        exp_input = layer_input.exp()
        sum_exp_input = exp_input.sum(axis=0)
        self._soft_max_output.set_values(exp_input / sum_exp_input)

        loss = label * self._soft_max_output.log()
        loss = -loss.sum()
        return loss

    def backward(self):
        self._backward_soft_max(self._inputs[0], self._inputs[1])
        self._backward_label(self._inputs[0], self._inputs[1])
        del self._soft_max_output
        self._soft_max_output = None

    def _backward_soft_max(self, input0, input1):
        if isinstance(input0, SmartTensor) and input0.requires_grad:
            if isinstance(input1, SmartTensor):
                storage = self._soft_max_output - input1.data
            else:
                storage = self._soft_max_output - input1
            storage = self._output.grad * storage
            input0.update_grad(storage)

    def _backward_label(self, input0, input1):
        if isinstance(input1, SmartTensor) and input1.requires_grad:
            storage = -self._output.grad * self._soft_max_output.log()
            input1.set_values(storage)

    def _check_inputs(self, *args):
        self._inputs = args
        message = ""
        if len(args) != 2:
            message = "{} should have 2 inputs".format(self._name)
            return False, message
        if not isinstance(self._inputs[0], SmartTensor):
            message = "input0 should be SmartTensor"
            return False, message
        if self._inputs[0].shape != self._inputs[1].shape:
            raise ValueError("shape of input0 and input1 should be equal")
        return True, message