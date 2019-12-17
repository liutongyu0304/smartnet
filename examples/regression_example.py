from smartnet import *
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score


"""
# description：
    do some regressions by smartnet.
"""


options = {"lr": 0.01, "weight_decay": 0.01,
           "momentum": 0.9, "beta": 0.999, "eps": 1e-8}


class DataFactory(object):
    def __init__(self):
        self.x = SmartTensor(None, requires_grad=False)
        self.y = SmartTensor(None, requires_grad=False)
        self.nsamples = 0

    def create_y_by_fun1(self, nsamples):
        self.x = SmartTensor(np.random.rand(nsamples, 3), requires_grad=False)
        self.y = SmartTensor(np.zeros((nsamples, 1)), requires_grad=False)
        self.y.data[:, 0] = 0.5*self.x.data[:, 0] * self.x.data[:, 1] + self.x.data[:, 2]**2 + 0.1
        self.nsamples = nsamples

    def create_y_by_fun2(self, nsamples):
        self.x = SmartTensor(np.random.rand(nsamples, 3), requires_grad=False)
        self.y = SmartTensor(np.zeros((nsamples, )), requires_grad=False)
        self.y.data[:, 0] = np.exp(self.x.data[:, 0]) * self.x.data[:, 1] + self.x.data[:, 2]

    def get_train_samples(self):
        n = int(self.nsamples*0.5)
        train_x = SmartTensor(self.x.data[0:n, :], requires_grad=False)
        train_y = SmartTensor(self.y.data[0:n, :],requires_grad=False)
        return train_x, train_y

    def get_test_samples(self):
        n = int(self.nsamples * 0.5)
        test_x = SmartTensor(self.x.data[n:, :], requires_grad=False)
        test_y = SmartTensor(self.y.data[n:, :], requires_grad=False)
        return test_x, test_y


def create_net(net_nodes, x, y):
    net_nodes = [x.shape[1]] + net_nodes + [1]
    net = SmartNet()
    net.add_layer(SmartDataLayer("x"), x)
    for i in range(len(net_nodes)-1):
        layer = SmartReluLayer("sigmoid" + str(i))
        net.add_layer(layer)
        layer = SmartLinearLayer("linear" + str(i), net_nodes[i], net_nodes[i+1], has_bias=False)
        net.add_layer(layer)
    net.add_layer(SmartMSELayer("mse"), y)
    return net


def regression_example():
    nsamples = 500
    periods = 100
    hidden_nodes = [5,10]
    loss = np.zeros((periods,))

    df = DataFactory()
    df.create_y_by_fun1(nsamples)

    train_x, train_y = df.get_train_samples()
    net = create_net(hidden_nodes, train_x, train_y)
    opt = SmartRMSPropOptim("rmsprop", net.trainable_parameters(),
                            lr=options["lr"], weight_decay=options["weight_decay"],
                            beta=options["beta"])

    # begin to train.
    for j in range(periods):
        net.zero_grad()
        loss[j] = net.forward()
        net.backward()
        opt.step()
    # plot train loss
    plt.plot(loss)

    # plot train result
    l, predict_y = net.predict(train_x)
    print("trian set loss:", l)
    print("train set correlation", np.corrcoef(train_y.data[:, 0], predict_y[:, 0])[0, 1])
    print("train set r2 score", r2_score(train_y.data[:, 0], predict_y[:, 0]))
    plt.figure(2)
    plt.scatter(train_y.data[:, 0], predict_y[:, 0])

    # plot test result
    test_x, test_y = df.get_test_samples()
    l, predict_y = net.predict(test_x)
    print("\ntest set loss:", l)
    print("test set correlation", np.corrcoef(test_y.data[:, 0], predict_y[:, 0])[0, 1])
    print("train set r2 score", r2_score(test_y.data[:, 0], predict_y[:, 0]))
    plt.figure(3)
    plt.scatter(test_y.data[:,0], predict_y[:,0])
    plt.show()


if __name__ == "__main__":
    regression_example()