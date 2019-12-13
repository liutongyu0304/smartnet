# coding=utf-8
from .tensor import *
import queue


class SmartAutoGrad(object):
    def __init__(self):
        self._tensors = list()
        self._dag = list()

    def add_tensor(self, tensor):
        assert isinstance(tensor, SmartTensor)
        self._tensors.append(tensor)

    def create_dag(self, tensor):
        assert tensor in self._tensors
        if tensor.size != 1:
            raise Exception("backward should begin from scalar.")
        self._dag = list()
        self._dag.append(tensor)

        q = queue.Queue()
        q.put(tensor)
        while not q.empty():
            p = q.get()
            if p.op is None:
                break
            for t in p.op.inputs:
                if isinstance(t, SmartTensor):
                    assert t in self._tensors
                    if not t.is_leaf:
                        self._dag.append(t)
        return self._dag

    def backward(self):
        origin = self._dag[0]
        origin.make_grad()
        data = origin.grad.data
        data[:] = 1.0
        for tensor in self._dag:
            if tensor.op is not None:
                tensor.op.backward()
        for tensor in self._tensors:
            if not tensor.is_leaf and not tensor.retain_grad:
                tensor.clear_grad()
    
    def clear_dag(self):
        del self._dag

    def clear_tensors(self):
        del self._tensors

    def clear(self):
        self.clear_dag()
        self.clear_tensors()
