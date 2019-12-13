# coding=utf-8
from .auto_grad import SmartAutoGrad


class SmartGraph(object):
    def __init__(self):
        self._auto_grad = SmartAutoGrad()

    def add_tensor(self, tensor):
        self._auto_grad.add_tensor(tensor)

    def clear_graph(self):
        self._auto_grad.clear()

    def auto_grad(self, tensor, grad=None, retain_graph=False):
        self._auto_grad.create_dag(tensor)
        self._auto_grad.backward()
        if not retain_graph:
            self.clear_graph()


Graph = SmartGraph()


def get_graph():
    return Graph
