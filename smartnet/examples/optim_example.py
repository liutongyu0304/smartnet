from smartnet import *


"""
# description：
    compare results of different optimization algorithms in regression.
    SGD, RMSProp, Momentum, Adam.
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


optim_options = {"lr": 0.01, "weight_decay": 0,
                 "momentum": 0.9, "beta": 0.999, "eps": 1e-8}


class DataFactory(object):
    def __init__(self, features, samples):
        self.initial_x = SmartTensor(np.random.rand(samples, features), requires_grad=False)
        self.initial_y = SmartTensor(np.zeros((samples, 1)), requires_grad=False)

    def create_y_by_linear(self, gain, bias):
        self.initial_y.data[:] = np.matmul(self.initial_x.data, gain * np.ones((self.initial_x.shape[1], 1))) + bias


def create_net(net_nodes, x, y):
    net_nodes = [x.shape[1]] + net_nodes + [1]
    net = SmartNet()
    net.add_layer(SmartDataLayer("x"), x)
    for i in range(len(net_nodes)-1):
        layer = SmartSigmoidLayer("sigmoid" + str(i))
        net.add_layer(layer)
        layer = SmartLinearLayer("linear" + str(i), net_nodes[i], net_nodes[i+1])
        net.add_layer(layer)
    net.add_layer(SmartMSELayer("mse"), y)
    return net


def create_optims(name, net, options):
    if name == "sgd":
        return SmartSGDOptim(name, net.trainable_parameters(),
                             lr=options["lr"], weight_decay=options["weight_decay"])
    elif name == "momentum":
        return SmartMomentumOptim(name, net.trainable_parameters(),
                                  lr=options["lr"], weight_decay=options["weight_decay"],
                                  momentum=options["momentum"])
    elif name == "rmsprop":
        return SmartRMSPropOptim(name, net.trainable_parameters(),
                                  lr=options["lr"], weight_decay=options["weight_decay"],
                                  beta=options["beta"])
    elif name == "adam":
        return SmartAdamOptim(name, net.trainable_parameters(),
                                  lr=options["lr"], weight_decay=options["weight_decay"],
                                  beta=options["beta"], momentum=options["momentum"])


def linear_regression_example():
    """
    # description：
        one hidden layer neural to fit linear function.
    """
    nsamples = 100
    nfeatures = 4
    periods = 100
    hidden_nodes = [5, 10]
    loss = np.zeros((periods, 4))

    df = DataFactory(nfeatures, nsamples)
    df.create_y_by_linear(0.1, 0.05)

    for i, name in enumerate(["sgd", "momentum", "rmsprop", "adam"]):
        net = create_net(hidden_nodes, df.initial_x, df.initial_y)
        opt = create_optims(name, net, optim_options)
        for j in range(periods):
            net.zero_grad()
            loss[j, i] = net.forward()
            net.backward()
            opt.step()

    d = pd.DataFrame(loss, columns=["sgd", "momentum", "rmsprop", "adam"])
    d.plot()
    plt.show()


if __name__ == "__main__":
    linear_regression_example()
