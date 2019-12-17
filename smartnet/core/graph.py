# coding=utf-8
from .auto_grad import SmartAutoGrad


class SmartGraph(object):
    def __init__(self):
        self._no_grad = False
        self._auto_grad = SmartAutoGrad()

    def add_tensor(self, tensor):
        if not self._no_grad:
            self._auto_grad.add_tensor(tensor)

    def clear_graph(self):
        self._auto_grad.clear()

    def auto_grad(self, tensor, grad=None, retain_graph=False):
        self._auto_grad.create_dag(tensor)
        self._auto_grad.backward()
        if not retain_graph:
            self.clear_graph()

    def set_no_grad(self, mode=True):
        self._no_grad = mode


Graph = SmartGraph()


def get_graph():
    return Graph


class no_grad(object):
    def __enter__(self):
        get_graph().set_no_grad(True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        get_graph().set_no_grad(False)
