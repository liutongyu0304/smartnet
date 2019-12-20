# coding=utf-8
from .core import Tensor
from collections import OrderedDict


class Module(object):
    def __init__(self, name=""):
        self._parameters = OrderedDict()
        self._modules = OrderedDict()
        self._training = True
        self._name = name

    def __setattr__(self, key, value):
        if isinstance(value, Module):
            if key in self._modules:
                del self._modules[key]
            self._modules[key] = value
        elif isinstance(value, Tensor):
            if key in self._parameters:
                del self._parameters[key]
            self._parameters[key] = value
        else:
            object.__setattr__(self, key, value)

    def __getattr__(self, item):
        if item in self._parameters:
            return self._parameters[item]
        elif item in self._modules:
            return self._modules[item]
        else:
            raise AttributeError("module has no attribution {}".format(item))

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)

    def named_parameters(self, recurse=True):
        parameters = OrderedDict()
        for name, module in self.named_modules(recurse=recurse, root=True).items():
            for key, parameter in module._parameters.items():
                parameters[name + key] = parameter
        return parameters

    def parameters(self, recurse=True):
        return list(self.named_parameters(recurse=recurse).values())

    def named_modules(self, recurse=True, root=True):
        modules = OrderedDict()
        if root:
            modules["self->"] = self
        for name, module in self._modules.items():
            modules[name + "->"] = module

        if recurse:
            for name, module in self._modules.items():
                for name_, module_ in module.named_modules(recurse=recurse, root=False).items():
                    modules[name + "->" + name_] = module
        return modules

    def modules(self, recurse=True):
        return list(self.named_modules(recurse=recurse).values())

    def named_children(self):
        return self._modules

    def children(self):
        return list(self._modules.values())

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def train(self):
        self._training = True
        for module in self._modules.values():
            module.train()

    def eval(self):
        self._training = False
        for module in self._modules.values():
            module.eval()

    def total_trainable_size(self, root=True):
        trainable_size = 0
        for parameter in self.named_parameters().values():
            if parameter is not None:
                trainable_size += parameter.size
        return trainable_size

    @property
    def name(self):
        return self._name
