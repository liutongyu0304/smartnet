import smartnet as sn
import smartnet.optims as optims
import smartnet.layers as layers


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
        self.initial_x = sn.random((samples, features), requires_grad=False)
        self.initial_y = sn.random((samples, 1), requires_grad=False)

    def create_y_by_linear(self, gain, bias):
        self.initial_y.data[:] = np.matmul(self.initial_x.data, gain * np.ones((self.initial_x.shape[1], 1))) + bias


def create_net(features):
    class Net(sn.Module):
        def __init__(self, features):
            super(Net, self).__init__()
            self.l1 = layers.LinearLayer(features, 2*features)
            self.a1 = layers.ReluLayer()
            self.l2 = layers.LinearLayer(2*features, 1)

        def forward(self, x):
            x = self.l1(x)
            x = self.a1(x)
            x = self.l2(x)
            return x
    return Net(features)


def create_optims(name, net, options):
    if name == "sgd":
        return optims.SGDOptim(net.named_parameters(),
                              lr=options["lr"], weight_decay=options["weight_decay"])
    elif name == "momentum":
        return optims.MomentumOptim(net.named_parameters(),
                                   lr=options["lr"], weight_decay=options["weight_decay"],
                                   momentum=options["momentum"])
    elif name == "rmsprop":
        return optims.RMSPropOptim(net.named_parameters(),
                                  lr=options["lr"], weight_decay=options["weight_decay"],
                                  beta=options["beta"])
    elif name == "adam":
        return optims.AdamOptim(net.named_parameters(),
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
    loss = np.zeros((periods, 4))

    df = DataFactory(nfeatures, nsamples)
    df.create_y_by_linear(0.1, 0.05)

    for i, name in enumerate(["sgd", "momentum", "rmsprop", "adam"]):
        net = create_net(nfeatures)
        opt = create_optims(name, net, optim_options)
        net_loss = layers.MSELayer()

        for j in range(periods):
            opt.zero_grad()
            y = net(df.initial_x)
            l = net_loss(y, df.initial_y)
            l.backward()
            opt.step()
            loss[j, i] = l.item()

    d = pd.DataFrame(loss, columns=["sgd", "momentum", "rmsprop", "adam"])
    d.plot()
    plt.show()


if __name__ == "__main__":
    linear_regression_example()
