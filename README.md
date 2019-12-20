# smartnet
a smart(simple) implentation of dynamic compute graph and neural net just like pytorch based on numpy and cupy.  
#how to use  
it is much alike pytorch. for a simple optimization problem:

```
min f(x) = (x1 - 1)^2 + (x2 - 1)^2
x0 = [0.0, 0.0]
```
obviously x_opt = [1.0, 1.0]. we can solve it with smartnet
```
import smartnet as sn


x = sn.zeros((2, 1), requires_grad=True)
for i in range(1000):
    x.zero_grad()
    y = sn.sum((x - 1)**2)
    y.backward()
    x.update_data(0.01)
print(x)
```
for more usage see examples and tests.
