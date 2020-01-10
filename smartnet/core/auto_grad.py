# coding=utf-8
from .tensor import *
import queue


class SmartAutoGrad(object):
    def __init__(self):
        self._tensors = list()
        self._dag = list()

    def add_tensor(self, tensor):
        assert isinstance(tensor, Tensor)
        if tensor not in self._tensors:
            self._tensors.append(tensor)

    def create_dag(self, tensor):
        """
        # description:
            non-leaf tensor is created by operation, and operation's inputs point to the next tensor.
            tenors and operations make a directed acyclic graph.
            this function do broad first search of dag.
            begin from needed backward tensor, and put all the tensors that requires gradient to a queue.
        """
        assert tensor in self._tensors
        if tensor.size != 1:
            raise Exception("backward should begin from scalar.")
        if not tensor.requires_grad:
            raise Exception("all tensors are set not requires_grad.")
        self._dag = list()
        self._dag.append(tensor)

        q = queue.Queue()
        q.put(tensor)
        while not q.empty():
            p = q.get()
            if p.op is None:
                break
            for t in p.op.inputs:
                if isinstance(t, Tensor):
                    assert t in self._tensors
                    if not t.is_leaf and t.requires_grad:
                        if t in self._dag:
                            # if t already exists in dag, if putting it to queue again,
                            # it will backwards twice, which make a wrong result of next tensors.
                            # if do not put it, the previous t does not have information
                            # between previous t and current t.
                            # so the best idea is remove previous t and then put t to dag.
                            self._dag.remove(t)
                        self._dag.append(t)
                        q.put(t)
        return self._dag

    def backward(self):
        origin = self._dag[0]
        origin.make_grad()
        origin._grad = 0 * origin.grad + 1.0
        for tensor in self._dag:
            if tensor.op is not None:
                if tensor.is_leaf:
                    raise RuntimeError("leaf variable has been moved into the graph interior")
                else:
                    tensor.op.backward()
        for tensor in self._tensors:
            if not tensor.is_leaf and not tensor.retain_grad:
                tensor.clear_grad()
    
    def clear_dag(self):
        del self._dag
        self._dag = list()

    def clear_tensors(self):
        for tensor in self._tensors:
            if not tensor.is_leaf and tensor.requires_grad:
                self._tensors.remove(tensor)
                del tensor

    def clear(self):
        self.clear_dag()
        self.clear_tensors()
