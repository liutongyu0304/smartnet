import sys
sys.path.append("/Home/kisen/liuxm/smartnet/smartnet/")

from smartnet.tensor import *

a = TensorOp.zeros((3,4))
b = a + 1
c = b * 2