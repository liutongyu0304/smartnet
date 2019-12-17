# coding=utf-8
from ..module import *


class SmartRnnCell(SmartModule):
    def __init__(self):
        super(SmartRnnCell, self).__init__("RnnCell")

    def forward(self, *inputs, **kwargs):
        pass