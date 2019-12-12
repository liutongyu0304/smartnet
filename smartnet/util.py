# coding=utf-8
import numpy as np
import numpy as gnp


def create_data_on_device(shape, device, dtype):
    assert isinstance(shape, tuple)
    if device == "cpu":
        return np.zeros(shape, dtype=dtype)
    else:
        return gnp.zeros(shape,dtype=dtype)