import torch
import numpy as np

a = np.ones((3,)) * 0.5
b = np.ones((3,))
c = a + b
d = c.sum(axis=0, keepdims=True)

a = torch.randn(2,3,requires_grad=True)
b = a * 3+ 2
b.retain_grad()
e = torch.reshape(b, (3, 2))
e.retain_grad()
c = torch.sum(b)

c.backward(retain_graph=True)
d = torch.sum(e)
d.backward()
print(a)