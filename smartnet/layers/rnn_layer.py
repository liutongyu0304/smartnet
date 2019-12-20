# coding=utf-8
from ..module import *
from .. import core as Core
from ..core import function as F


class RnnCell(Module):
    """
    # description:
        rnncell with tanh or relu nonlinearity, base of rnn.
            h_o = tanh(x * w_ih + b_ih + h_i * w_hh + b_hh)
        with samples = m.
        h_i: shape of (m, hidden_size), output of previos rnncell, or user defined tensor.
        x: shape of (m, input_size), the nth input of rnn
        w_ith: shape of (input_size, hidden_size), weight of x, to be trained tensor.
        w_hh: shape of (hidden_size, hidden_size), weight of h_i, to be trained tensor.
        b_ih: shape of (1, hidden_size), bias of x, to be trained tensor.
        b_hh: shape of (1, hidden_size), bias of h_i, to bt trained tensor.
    """
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh'):
        super(RnnCell, self).__init__("RnnCell")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity

        self.weight_ih = Core.random((input_size, hidden_size), requires_grad=True)
        self.weight_hh = Core.random((hidden_size, hidden_size), requires_grad=True)

        if bias:
            self.bias_ih = Core.random((1, hidden_size), requires_grad=True)
            self.bias_hh = Core.random((1, hidden_size), requires_grad=True)

    def forward(self, input0, h0=None):
        assert isinstance(input0, Tensor)
        self.check_forward_input(input0)
        if h0 is None:
            h0 = Core.zeros((input0.shape[0], self.hidden_size),
                            dtype=input0.dtype, device=input0.device)
        self.check_forward_hidden(input0, h0)

        s = input0 * self.weight_ih + h0 * self.weight_hh
        if self.has_bias:
            s = s + self.bias_ih + self.bias_hh
        if self.nonlinearity == "tanh":
            s = F.tanh(s)
        elif self.nonlinearity == "relu":
            s = F.relu(s)
        else:
            raise ValueError("nonlinearity should be either tanh or relu")
        return s

    def reset_parameters(self):
        pass

    def check_forward_input(self, input0):
        if input0.shape[1] != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input0.shape[1], self.input_size))

    def check_forward_hidden(self, input0, h0):
        if input0.shape[0] != h0.shape[0]:
            raise RuntimeError(
                "Input batch size {} doesn't match hidden batch size {}".format(
                    input0.shape[0], h0.shape[0]))

        if h0.shape[1] != self.hidden_size:
            raise RuntimeError(
                "hidden has inconsistent hidden_size: got {}, expected {}".format(
                    h0.shape[1], self.hidden_size))


class Rnn(Module):
    """
    # description:
        rnn modified from pytorch RNNBase and RNN.
        for an n input sequences, the (t)th output can be computed by the
        (t-1)th output and the (t)th input sequence.
            h_t = tanh(x * w_ih + b_ih + h_t_1 * w_hh + b_hh)
    # arguments:
        inputs: shape of (seq_len, batch_size, input_size)
        h_0: shape of (num_layers * num_directions, batch, hidden_size)
    # return:
        outputs: shape of (seq_len, batch, num_directions * hidden_size)
        h_n: shape of  (num_layers * num_directions, batch, hidden_size)
    """
    def __init__(self, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0., bidirectional=False):
        super(Rnn, self).__init__("Rnn")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        gate_size = hidden_size
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions

                w_ih = Core.random(gate_size, layer_input_size)
                w_hh = Core.random(gate_size, hidden_size)

                layer_params = [w_ih, w_hh]
                suffix = '_reverse' if direction == 1 else ''
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                    b_ih = Core.random(gate_size)
                    b_hh = Core.random(gate_size)
                    layer_params += [b_ih, b_hh]
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._all_weights.append(param_names)
        self.reset_parameters()

    def forward(self, input, h0=None):
        """
            1.out tensor does not slicing, indexing backward, so input and h0's gradient can not be
              transferring backward
            2.currently bi-direction is not supported.
        """
        batch_size = input.shape[1]
        if h0 is None:
            num_directions = 2 if self.bidirectional else 1
            h0 = Core.zeros((self.num_layers * num_directions,
                                batch_size, self.hidden_size),
                                dtype=input.dtype, device=input.device)
        self.check_forward_args(input, h0)
        len_seq = input.shape[0]
        input_size = self.input_size
        outputs = []
        hx = []
        input, h0 = self.preprocess_args(input, h0)
        for t in range(len_seq):
            for i in range(self.num_layers):
                w_ih = self.__getattr__("weight_ih_l{}1".format(i))

    def preprocess_args(self, input, h0):
        # change three dimension tensor to list of two dimension tensor
        input_data = input.data
        h0_data = h0.data
        output = list()
        h0_out = list()

        for i in range(input.shape[0]):
            out = Core.zeros(input.shape[1:], device=input.device, dtype=input.dtype)
            out.set_values(input_data.data[i, :, :])
            output.append(out)

        for i in range(h0.shape[0]):
            out = Core.zeros(h0.shape[1:], device=h0.device, dtype=h0.dtype)
            out.set_values(h0.data[i, :, :])
            h0_out.append(out)
        return output, h0_out

    def forward_layer(self, t_layer, input, h0=None):
        """
        # description:
            compute output and hx of t_th hidden layer
        # arguments:
            t_layer: int, the t_th layer
            input: list(Tensor), shape of ()
        # return:
            output: list(Tensor), shape of (batch, num_directions * hidden_size)
            hx: list(Tensor), shape of (batch, hidden_size)
        """
        output = []
        hx = None
        w_ih = self.__getattr__("weight_ih_l{}".format(t_layer))
        w_hh = self.__getattr__("weight_hh_l{}".format(t_layer))
        if self.bias:
            b_ih = self.__getattr__("bias_ih_l{}".format(t_layer))
            b_hh = self.__getattr__("bias_hh_l{}".format(t_layer))
        for t in range(len(input)):
             pass

        return output, hx

    def check_input(self, input):
        # accept three dimension tensor as input
        expected_input_dim = 3
        if input.ndim != expected_input_dim:
            raise RuntimeError('input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.ndim))
        if self.input_size != input.shape[-1]:
            raise RuntimeError('input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.shape[-1]))

    def check_hidden_size(self, h0):
        mini_batch = h0.shape[1]
        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)
        if expected_hidden_size != h0.shape:
            raise RuntimeError("Expected hidden size {}, got {}".
                               format(expected_hidden_size, h0.shape))

    def check_forward_args(self, input, h0):
        self.check_input(input)
        self.check_hidden_size(h0)

    def reset_parameters(self):
        pass
