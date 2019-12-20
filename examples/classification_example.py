import smartnet as sn
import smartnet.optims as optims
import smartnet.layers as layers
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.datasets import make_circles, make_moons
from sklearn.preprocessing import OneHotEncoder


"""
# description：
    do classification by smartnet.
"""


options = {"lr": 0.01, "weight_decay": 0.00,
           "momentum": 0.9, "beta": 0.999, "eps": 1e-8}


class DataFactory(object):
    def __init__(self):
        self.x = None
        self.y = None
        self.nsamples = 0

    def create_circles(self, nsamples):
        x, y = make_circles(n_samples=nsamples, factor=0.5, noise=0.1)
        h = OneHotEncoder()
        y = h.fit_transform(y.reshape((-1, 1))).todense()
        self.x = sn.Tensor(data=x, requires_grad=False)
        self.y = sn.Tensor(data=y, requires_grad=False)
        self.nsamples = nsamples

    def create_moons(self, nsamples):
        x, y = make_moons(n_samples=self.nsamples, noise=0.1)
        h = OneHotEncoder()
        y = h.fit_transform(y.reshape((-1, 1))).todense()
        self.x = sn.Tensor(data=x, requires_grad=False)
        self.y = sn.Tensor(data=y, requires_grad=False)
        self.nsamples = nsamples

    def get_train_samples(self):
        n = int(self.nsamples*0.5)
        train_x = sn.Tensor(data=self.x.data[0:n, :], requires_grad=False)
        train_y = sn.Tensor(data=self.y.data[0:n, :], requires_grad=False)
        return train_x, train_y

    def get_test_samples(self):
        n = int(self.nsamples * 0.5)
        test_x = sn.Tensor(data=self.x.data[n:, :], requires_grad=False)
        test_y = sn.Tensor(data=self.y.data[n:, :], requires_grad=False)
        return test_x, test_y


def create_net(features, labels):
    class Net(sn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.l1 = layers.LinearLayer(features, 2*features)
            self.a1 = layers.SigmoidLayer()
            self.l2 = layers.LinearLayer(2*features, labels)
            self.a2 = layers.SigmoidLayer()

        def forward(self, x):
            x = self.l1(x)
            x = self.a1(x)
            x = self.l2(x)
            y = self.a2(x)
            return y

    return Net()


def regression_example():
    nsamples = 500
    periods = 100
    loss = np.zeros((periods,))

    df = DataFactory()
    df.create_circles(nsamples)
    # df.create_moons(nsamples)

    train_x, train_y = df.get_train_samples()
    net = create_net(2, 2)
    net_loss = layers.CrossEntropyLayer()
    opt = optims.RMSPropOptim(net.named_parameters(),
                              lr=options["lr"], weight_decay=options["weight_decay"],
                              beta=options["beta"])

    # begin to train.
    for j in range(periods):
        opt.zero_grad()
        y = net.forward(train_x)
        l = net_loss(y, train_y)
        loss[j] = l.data[0, 0]
        l.backward()
        opt.step()
    # plot train loss
    plt.plot(loss)
    plt.show()

    # plot train result
    with sn.no_grad():
        predict_y = net(train_x)
        l = net_loss(predict_y, train_y)
    print("trian set loss:", l.data[0, 0])
    print("train set correlation", np.corrcoef(train_y.data[:, 0], predict_y.data[:, 0])[0, 1])
    print("train set r2 score", r2_score(train_y.data[:, 0], predict_y.data[:, 0]))
    plt.figure(2)
    plt.scatter(train_y.data[:, 0], predict_y.data[:, 0])

    # plot test result
    test_x, test_y = df.get_test_samples()
    with sn.no_grad():
        predict_y = net(test_x)
        l = net_loss(predict_y, test_y)
    print("\ntest set loss:", l.data[0, 0])
    print("test set correlation", np.corrcoef(test_y.data[:, 0], predict_y.data[:, 0])[0, 1])
    print("test set r2 score", r2_score(test_y.data[:, 0], predict_y.data[:, 0]))
    plt.figure(3)
    plt.scatter(test_y.data[:,0], predict_y.data[:,0])
    plt.show()


if __name__ == "__main__":
    regression_example()
