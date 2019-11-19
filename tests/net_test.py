from smartnet import *
import numpy as np

x = SmartTensor(np.random.rand(100, 10))
y = SmartTensor(np.random.rand(100, 1))
net = SmartNet()

net.add_layer(SmartDataLayer("input"), x)
net.add_layer(SmartLinearLayer("linear1", 10, 5))
net.add_layer(SmartLinearLayer("linear2", 5, 1))
net.add_layer(SmartMSELayer("mse"), y)

optim = SmartSGDOptim("sgd", net.trainable_parameters())

for i in range(100):
    optim.zero_grad()
    loss = net.forward()
    net.backward()
    optim.step()