# coding=utf-8
from .util import *
from .tensor import Tensor


class Operation(object):
    def __init__(self, name):
        self._name = name
        self._inputs = list()
        self._output = None
        self._pkg = None
    
    def forward(self, *args):
        result, message = self._check_inputs(*args)
        if not result:
            raise ValueError(message)
        return None

    def backward(self):
        raise NotImplementedError()

    def _check_inputs(self, *args):
        return True, ""

    def _check_input_device(self, *args):
        def get_tensors():
            tensors = []
            for arg in args:
                if isinstance(arg, Tensor):
                    tensors.append(arg)
                elif isinstance(arg, (list, tuple)):
                    for t in arg:
                        if isinstance(t, Tensor):
                            tensors.append(t)
                elif isinstance(arg, dict):
                    for t in arg.values():
                        if isinstance(t, Tensor):
                            tensors.append(t)
            return tensors

        tensors = get_tensors()
        for t in tensors:
                if self._pkg is None:
                    self._pkg = t.pkg
                else:
                    if self._pkg != t.pkg:
                        raise RuntimeError("all inputs should have same device")

    def __call__(self, *args):
        self._output = self.forward(*args)
        for t in self._inputs:
            if isinstance(t, Tensor):
                if t.requires_grad:
                    self._output.set_requires_grad(True)
                    break
        self._output._set_op(self)
        self._output._set_leaf(False)
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


class ReshapeOperation(Operation):
    def __init__(self):
        super(ReshapeOperation, self).__init__("ReshapeOperation")
    
    def forward(self, *args):
        super(ReshapeOperation, self).forward(*args)
        shape = self._inputs[1]
        data = self._inputs[0].data.reshape(shape)
        output = Tensor(data=data, requires_grad=self._inputs[0].requires_grad)
        return output
    
    def backward(self):
        if not self._inputs[0].requires_grad:
            return
        self._inputs[0].make_grad()
        grad = self._output.grad.reshape(self._inputs[0].shape)
        self._inputs[0].update_grad(grad)

    def _check_inputs(self, *args):
        self._check_input_device(*args)
        self._inputs = args
        message = ""
        if len(args) != 2:
            message = "{} should have 2 inputs".format(self._name)
            return False, message
        if not isinstance(self._inputs[0], Tensor):
            message = "{} first input's type should be Tensor".format(self._name)
            return False, message
        if not isinstance(self._inputs[1], tuple):
            message = "{} shape should be tuple".format(self._name)
            return False, message
        return True, message


class TransposeOperation(Operation):
    def __init__(self):
        super(TransposeOperation, self).__init__("TransposeOperation")
    
    def forward(self, *args):
        super(TransposeOperation, self).forward(*args)
        data = self._inputs[0].data.transpose()
        output = Tensor(data=data, requires_grad=self._inputs[0].requires_grad)
        return output
    
    def backward(self):
        if not self._inputs[0].requires_grad:
            return
        self._inputs[0].make_grad()
        storage = self._output.grad.transpose()
        self._inputs[0].update_grad(storage)

    def _check_inputs(self, *args):
        self._check_input_device(*args)
        self._inputs = args
        message = ""
        if len(args) != 1:
            message = "{} should have 1 inputs".format(self._name)
            return False, message
        if not isinstance(self._inputs[0], Tensor):
            message = "{} to be transposed should be Tensor".format(self._name)
            return False, message
        return True, message


class NegativeOperation(Operation):
    def __init__(self):
        super(NegativeOperation, self).__init__("NegativeOperation")
    
    def forward(self, *args):
        super(NegativeOperation, self).forward(*args)
        output = Tensor(data=-self._inputs[0].data,
                        requires_grad=self._inputs[0].requires_grad)
        return output
    
    def backward(self):
        if not self._inputs[0].requires_grad:
            return
        self._inputs[0].make_grad()
        self._inputs[0].update_grad(-self._output.grad)

    def _check_inputs(self, *args):
        self._check_input_device(*args)
        self._inputs = args
        message = ""
        if len(args) != 1:
            message = "{} should have 1 inputs".format(self._name)
            return False, message
        if not isinstance(self._inputs[0], Tensor):
            message = "{} to be transposed should be Tensor".format(self._name)
            return False, message
        return True, message


class TwoInputsOperation(Operation):
    def __init__(self, name):
        super(TwoInputsOperation, self).__init__(name)

    def forward(self, *args):
        super(TwoInputsOperation, self).forward(*args)
        requires_grad = False
        if isinstance(self._inputs[0], Tensor):
            input0 = self._inputs[0].data
            requires_grad = requires_grad and self._inputs[0].requires_grad
        else:
            input0 = self._inputs[0]

        if isinstance(self._inputs[1], Tensor):
            input1 = self._inputs[1].data
            requires_grad = requires_grad and self._inputs[1].requires_grad
        else:
            input1 = self._inputs[1]

        data = self._operation(input0, input1)
        output = Tensor(data=data, requires_grad=requires_grad)
        return output

    def _operation(self, input0, input1):
        # obviously either input0 or input1 must be ndarray
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def _check_inputs(self, *args):
        self._check_input_device(*args)
        self._inputs = args
        message = ""
        if len(args) != 2:
            message = "{} should have 2 inputs".format(self._name)
            return False, message
        if not (isinstance(self._inputs[0], Tensor) or isinstance(self._inputs[1], Tensor)):
            message = "{} should have at least one Tensor".format(self._name)
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
        if isinstance(input0, Tensor) and input0.requires_grad:
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
        if isinstance(input0, Tensor) and input0.requires_grad:
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
        if isinstance(input0, Tensor) and input0.requires_grad:
            input0.make_grad()
            grad = input0.grad
            if isinstance(input1, Tensor):
                data = self._output.grad * input1.data
            else:
                data = self._output.grad * input1
            if grad.shape != data.shape:
                # for broadcast
                reduce_ind = 0
                for i in range(len(grad.shape)):
                    if grad.shape[i] != data.shape[i]:
                        reduce_ind = i
                        break
                input0.update_grad(self._pkg.sum(data, axis=reduce_ind, keepdims=True))
            else:
                input0.update_grad(data)


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
        if isinstance(input0, Tensor) and input0.requires_grad:
            input0.make_grad()
            grad = input0.grad
            if isinstance(input1, Tensor):
                data = self._output.grad / input1.data
            else:
                data = self._output.grad / input1
            if grad.shape != data.shape:
                # for broadcast
                reduce_ind = 0
                for i in range(len(grad.shape)):
                    if grad.shape[i] != data.shape[i]:
                        reduce_ind = i
                        break
                input0.update_grad(self._pkg.sum(data, axis=reduce_ind, keepdims=True))
            else:
                input0.update_grad(data)

    def _backward_right(self, input0, input1):
        # input0 / input1
        # input1.grad
        if isinstance(input1, Tensor) and input1.requires_grad:
            input1.make_grad()
            grad = input1.grad
            if isinstance(input0, Tensor):
                data = self._output.grad * input0.data / input1.data ** 2
            else:
                data = self._output.grad * input0 / input1.data ** 2
            if grad.shape != data.shape:
                # for broadcast
                reduce_ind = 0
                for i in range(len(grad.shape)):
                    if grad.shape[i] != data.shape[i]:
                        reduce_ind = i
                        break
                input1.update_grad(data.sum(axis=reduce_ind, keepdims=True))
            else:
                input1.update_grad(data)


class MatmulOperation(TwoInputsOperation):
    def __init__(self):
        super(MatmulOperation, self).__init__("MatmulOperation")
    
    def _operation(self, input0, input1):
        return self._pkg.matmul(input0, input1)

    def backward(self):
        self._backward_left(self._inputs[0], self._inputs[1])
        self._backward_right(self._inputs[0], self._inputs[1])

    def _backward_left(self, input0, input1):
        # input.matmul(input1)
        # input.grad
        if isinstance(input0, Tensor) and input0.requires_grad:
            input0.make_grad()
            if isinstance(input1, Tensor):
                data = self._pkg.matmul(self._output.grad, input1.data.transpose())
            else:
                data = self._pkg.matmul(self._output.grad, input1.transpose())
            input0.update_grad(data)

    def _backward_right(self, input0, input1):
        # input.matmul(input1)
        # input1.grad
        if isinstance(input1, Tensor) and input1.requires_grad:
            input1.make_grad()
            if isinstance(input0, Tensor):
                data = self._pkg.matmul(input0.data.transpose(), self._output.grad)
            else:
                data = self._pkg.matmul(input0.transpose(), self._output.grad)
            input1.update_grad(data)


class PowOperation(Operation):
    def __init__(self):
        super(PowOperation, self).__init__("PowOperation")

    def forward(self, *args):
        super(PowOperation, self).forward(*args)
        data = self._inputs[0].data ** self._inputs[1]
        output = Tensor(data=data, requires_grad=self._inputs[0].requires_grad)
        return output

    def backward(self):
        if not self._inputs[0].requires_grad:
            return
        input0 = self._inputs[0]
        input0.make_grad()
        grad = self._output.grad * self._inputs[1] * self._inputs[0].data**(self._inputs[1] - 1)
        input0.update_grad(grad)

    def _check_inputs(self, *args):
        self._check_input_device(*args)
        self._inputs = args
        message = ""
        if len(args) != 2:
            message = "{} should have 2 inputs".format(self._name)
            return False, message
        if not isinstance(self._inputs[0], Tensor):
            message = "{} first input's type should be Tensor".format(self._name)
            return False, message
        if "int" not in str(type(self._inputs[1])) and "float" not in str(type(self._inputs[1])):
            message = "{} second input's type should be int/float".format(self._name)
            return False, message
        return True, message


class ExpOperation(Operation):
    def __init__(self):
        super(ExpOperation, self).__init__("ExpOperation")

    def forward(self, *args):
        super(ExpOperation, self).forward(*args)
        data = self._pkg.exp(self._inputs[0].data)
        output = Tensor(data=data, requires_grad=self._inputs[0].requires_grad)
        return output

    def backward(self):
        if not self._inputs[0].requires_grad:
            return
        input0 = self._inputs[0]
        input0.make_grad()
        data = self._output.grad * self._output.data
        input0.update_grad(data)

    def _check_inputs(self, *args):
        self._check_input_device(*args)
        self._inputs = args
        message = ""
        if len(args) != 1:
            message = "{} should have 1 inputs".format(self._name)
            return False, message
        if not isinstance(self._inputs[0], Tensor):
            message = "{} input should be Tensor".format(self._name)
            return False, message
        return True, message


class LogOperation(Operation):
    def __init__(self):
        super(LogOperation, self).__init__("LogOperation")

    def forward(self, *args):
        super(LogOperation, self).forward(*args)
        data = self._pkg.log(self._inputs[0].data)
        output = Tensor(data=data, requires_grad=self._inputs[0].requires_grad)
        return output

    def backward(self):
        if not self._inputs[0].requires_grad:
            return
        input0 = self._inputs[0]
        input0.make_grad()
        grad = self._output.grad / input0.data
        input0.update_grad(grad)

    def _check_inputs(self, *args):
        self._check_input_device(*args)
        self._inputs = args
        message = ""
        if len(args) != 1:
            message = "{} should have 1 inputs".format(self._name)
            return False, message
        if not isinstance(self._inputs[0], Tensor):
            message = "{} input should be Tensor".format(self._name)
            return False, message
        return True, message


class SumOperation(Operation):
    def __init__(self, axis=None, keepdims=True):
        super(SumOperation, self).__init__("SumOperation")
        self._axis = axis
        self._keepdims = keepdims

    def forward(self, *args):
        super(SumOperation, self).forward(*args)
        data = self._pkg.sum(self._inputs[0].data, axis=self._axis, keepdims=self._keepdims)
        output = Tensor(data=data, requires_grad=self._inputs[0].requires_grad)
        return output

    def backward(self):
        if not self._inputs[0].requires_grad:
            return
        input0 = self._inputs[0]
        input0.make_grad()
        if self._axis is not None:
            grad = self._output.grad * input0.pkg.ones_like(input0.data)
        else:
            grad = self._output.grad.item() * self._pkg.ones_like(input0.data)
        input0.update_grad(grad)

    def _check_inputs(self, *args):
        self._check_input_device(*args)
        self._inputs = args
        message = ""
        if len(args) != 1:
            message = "{} should have 1 inputs".format(self._name)
            return False, message
        if not isinstance(self._inputs[0], Tensor):
            message = "{} input should be Tensor".format(self._name)
            return False, message
        return True, message


class SigmoidOperation(Operation):
    def __init__(self):
        super(SigmoidOperation, self).__init__("SigmoidOperation")

    def forward(self, *args):
        super(SigmoidOperation, self).forward(*args)
        data = sigmoid(self._inputs[0].data)
        output = Tensor(data=data, requires_grad=self._inputs[0].requires_grad)
        return output

    def backward(self):
        if not self._inputs[0].requires_grad:
            return
        input0 = self._inputs[0]
        input0.make_grad()
        grad = self._output.grad * self._output.data * (1 - self._output.data)
        input0.update_grad(grad)

    def _check_inputs(self, *args):
        self._check_input_device(*args)
        self._inputs = args
        message = ""
        if len(args) != 1:
            message = "{} should have 1 inputs".format(self._name)
            return False, message
        if not isinstance(self._inputs[0], Tensor):
            message = "{} input should be Tensor".format(self._name)
            return False, message
        return True, message


class TanhOperation(Operation):
    def __init__(self):
        super(TanhOperation, self).__init__("TanhOperation")

    def forward(self, *args):
        super(TanhOperation, self).forward(*args)
        data = tanh(self._inputs[0].data)
        output = Tensor(data=data, requires_grad=self._inputs[0].requires_grad)
        return output

    def backward(self):
        if not self._inputs[0].requires_grad:
            return
        input0 = self._inputs[0]
        input0.make_grad()
        grad = self._output.grad * (1 - self._output.data**2)
        input0.update_grad(grad)

    def _check_inputs(self, *args):
        self._check_input_device(*args)
        self._inputs = args
        message = ""
        if len(args) != 1:
            message = "{} should have 1 inputs".format(self._name)
            return False, message
        if not isinstance(self._inputs[0], Tensor):
            message = "{} input should be Tensor".format(self._name)
            return False, message
        return True, message


class ReluOperation(Operation):
    def __init__(self):
        super(ReluOperation, self).__init__("ReluOperation")

    def forward(self, *args):
        super(ReluOperation, self).forward(*args)
        data = relu(self._inputs[0].data)
        output = Tensor(data=data, requires_grad=self._inputs[0].requires_grad)
        return output

    def backward(self):
        if not self._inputs[0].requires_grad:
            return
        input0 = self._inputs[0]
        input0.make_grad()
        grad = self._pkg.zeros_like(input0.grad)
        grad[:] = self._output.grad

        grad[input0.data < 0] = 0.0
        input0.update_grad(grad)

    def _check_inputs(self, *args):
        self._check_input_device(*args)
        self._inputs = args
        message = ""
        if len(args) != 1:
            message = "{} should have 1 inputs".format(self._name)
            return False, message
        if not isinstance(self._inputs[0], Tensor):
            message = "{} input should be Tensor".format(self._name)
            return False, message
        return True, message


class MSEOperation(TwoInputsOperation):
    def __init__(self):
        super(MSEOperation, self).__init__("MSEOperation")

    def _operation(self, input0, input1):
        return mse(input0, input1)

    def backward(self):
        self._backward(self._inputs[0], self._inputs[1])
        self._backward(self._inputs[1], self._inputs[0])

    def _backward(self, input0, input1):
        if isinstance(input0, Tensor) and input0.requires_grad:
            input0.make_grad()
            if isinstance(input1, Tensor):
                grad = input0.data - input1.data
            else:
                grad = input0.data - input1
            grad = self._output.grad * 2.0 * grad / grad.size
            input0.update_grad(grad)


class CrossEntropyOperation(TwoInputsOperation):
    def __init__(self):
        super(CrossEntropyOperation, self).__init__("CrossEntropyOperation")
        self._soft_max_output = None

    def _operation(self, input0, input1):
        layer_input = input0
        label = input1

        exp_input = self._pkg.exp(layer_input)
        sum_exp_input = exp_input.sum(axis=1, keepdims=True)
        self._soft_max_output = exp_input / sum_exp_input

        loss = label * self._pkg.log(self._soft_max_output)
        loss = -loss.sum(keepdims=True)
        return loss

    def backward(self):
        self._backward_soft_max(self._inputs[0], self._inputs[1])
        self._backward_label(self._inputs[0], self._inputs[1])
        del self._soft_max_output
        self._soft_max_output = None

    def _backward_soft_max(self, input0, input1):
        if isinstance(input0, Tensor) and input0.requires_grad:
            input0.make_grad()
            if isinstance(input1, Tensor):
                grad = self._soft_max_output - input1.data
            else:
                grad = self._soft_max_output - input1
            grad = self._output.grad * grad
            input0.update_grad(grad)

    def _backward_label(self, input0, input1):
        if isinstance(input1, Tensor) and input1.requires_grad:
            grad = -self._output.grad * self._soft_max_output.log()
            input1.set_values(grad)

    def _check_inputs(self, *args):
        self._check_input_device(*args)
        self._inputs = args
        message = ""
        if len(args) != 2:
            message = "{} should have 2 inputs".format(self._name)
            return False, message
        if not isinstance(self._inputs[0], Tensor):
            message = "input0 should be Tensor"
            return False, message
        if self._inputs[0].shape != self._inputs[1].shape:
            raise ValueError("shape of input0 and input1 should be equal")
        return True, message


class AsStrideOption(Operation):
    def __init__(self):
        super(AsStrideOption, self).__init__("AsStrideOption")

    def forward(self, *args):
        super(AsStrideOption, self).forward(*args)
        self._inputs = args
        output = Tensor(data=self._inputs[0].data[self._inputs[1]])
        return output

    def backward(self):
        self._inputs[0].make_grad()
        self._inputs[0].grad[self._inputs[1]] = self._inputs[0].grad[self._inputs[1]] + self._output.grad

    def _check_inputs(self, *args):
        self._check_input_device(*args)
        self._inputs = args
        message = ""
        if len(args) != 2:
            message = "{} should have 2 inputs".format(self._name)
            return False, message
        if not isinstance(self._inputs[0], Tensor):
            message = "{} first input's type should be Tensor".format(self._name)
            return False, message
        return True, message


class CatOperation(Operation):
    def __init__(self):
        super(CatOperation, self).__init__("CatOperation")

    def forward(self, *args):
        super(CatOperation, self).forward(*args)
        data = self._pkg.concatenate(self._inputs[0], axis=self._inputs[1])
        output = Tensor(data=data)
        return output

    def backward(self):
        inputs = self._inputs[0]
        axis = self._inputs[1]
        items = [None] * inputs[0].ndim
        ind = 0
        for t in inputs:
            items_t = items.copy()
            items_t[axis] = range(t.shape[axis]) + ind
            ind += t.shape[axis]
            t.update_grad(self._output.grad[items_t])

    def _check_inputs(self, *args):
        self._check_input_device(*args)
        self._inputs = args
        message = ""
        if len(args) != 2:
            message = "{} should have 2 inputs".format(self._name)
            return False, message
        if not isinstance(self._inputs[0], (tuple, list)):
            message = "{} first input's type should be tuple or list".format(self._name)
            return False, message
        for t in self._inputs[0]:
            if not isinstance(t, Tensor):
                message = "array to be cat should be all tensors."
                return False, message

        if "int" not in str(type(self._inputs[1])):
            message = "axis along which to cat should be int type."
            return False, message

        ndim = -1
        for t in self.inputs[0]:
            if ndim == -1:
                ndim = t.ndim
            else:
                if ndim != t.ndim:
                    message = "all the tensors to be cat should have same dimensions."
                    return False, message

        shape_ = None
        for t in self._inputs[0]:
            if shape_ is None:
                shape_ = t.shape
            else:
                for i in range(ndim):
                    if i != self._inputs[1]:
                        if shape_[i] != t.shape[i]:
                            message = "length of each dimensions except axis of tensors should be the same."
                            return False, message

        return True, message


class DropOutOperation(Operation):
    def __init__(self, keep_probs=0.5):
        super(DropOutOperation, self).__init__("DropOutOperation")
        assert 0 < keep_probs <= 1
        self._keep_probs = keep_probs
        self._prob_array = None

    def forward(self, *args):
        super(DropOutOperation, self).forward(*args)
        self._inputs = args
        layer_input = self._inputs[0]
        self._prob_array = self._pkg.random.rand(*layer_input.shape)
        output = Tensor(data=layer_input.data / self._keep_probs)
        output.data[self._prob_array >= self._keep_probs] = 0.0
        return output

    def backward(self):
        layer_input = self._inputs[0]
        layer_input.make_grad()
        grad = layer_input.grad
        ind = self._prob_array < self._keep_probs
        grad[ind] = grad[ind] + self._output.grad[ind] / self._keep_probs

    def _check_inputs(self, *args):
        self._check_input_device(*args)
        self._inputs = args
        message = ""
        if len(args) != 1:
            message = "{} should have 1 inputs".format(self._name)
            return False, message
        if not isinstance(self._inputs[0], Tensor):
            message = "{} first input's type should be Tensor".format(self._name)
            return False, message
        return True, message


class ClipOperation(Operation):
    """
    # description:
        clip gradient in case of gradient explode
        t = clip(t, min, max)
    """
    def __init__(self):
        super(ClipOperation, self).__init__("ClipOperation")
        self._ind = None

    def forward(self, *args):
        self._inputs = args
        layer_input = self._inputs[0]
        min_value = self._inputs[1]
        max_value = self._inputs[2]
        self._ind = min_value < layer_input < max_value
        output = Tensor(self._pkg.clip(layer_input, min_value, max_value))
        return output

    def backward(self):
        layer_input = self._inputs[0]
        layer_input.make_grad()
        grad = layer_input.grad
        grad[self._ind] = grad[self._ind] + self._output.grad[self._ind]

    def _check_inputs(self, *args):
        self._check_input_device(*args)
        self._inputs = args
        message = ""
        if len(args) != 3:
            message = "{} should have 3 inputs".format(self._name)
            return False, message
        if not isinstance(self._inputs[0], Tensor):
            message = "{} first input's type should be Tensor".format(self._name)
            return False, message
        return True, message
