import smartnet as sn
import smartnet.layers as layer
import smartnet.optims as optim
import numpy as np
import unittest


class SmartNetTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    @classmethod
    def setUpClass(cls):
        nsamples = 100
        cls.x = sn.Tensor(data=np.random.rand(nsamples, 10), requires_grad=False)
        cls.y = sn.Tensor(data=np.matmul(cls.x.data, 0.2 * np.ones((10, 1))) + 0.1, requires_grad=False)
        print("net test begins.")

    @classmethod
    def tearDownClass(cls):
        print("net test finished.")

    def create_net(self):
        class Net(sn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.l1 = layer.LinearLayer(10, 20)
                self.a1 = layer.ReluLayer()
                self.l2 = layer.LinearLayer(20, 1)
                self.a2 = layer.SigmoidLayer()
                self.l3 = layer.LinearLayer(1, 1)

            def forward(self, x):
                x = self.l1(x)
                x = self.a1(x)
                x = self.l2(x)
                x = self.a2(x)
                y = self.l3(x)
                return y

        return Net()

    def create_opt(self, net):
        opt = optim.SGDOptim(net.named_parameters(), lr=0.01, weight_decay=0.0)
        return opt

    def test_smartnet(self):
        net = self.create_net()
        opt = self.create_opt(net)
        loss = layer.MSELayer()

        for i in range(200):
            opt.zero_grad()
            y = net(self.x)
            l = loss(y, self.y)
            l.backward()
            if i % 10 == 0:
                print("the {}th period, loss = {}".format(i, l.data[0]))
            opt.step()

    def test_with_torch(self):
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            print("torch module is not installed, exit function compare_with_torch.")
            return

        class TorchNet(nn.Module):
            def __init__(self):
                super(TorchNet, self).__init__()
                self.l1 = nn.Linear(10, 20)
                self.a1 = nn.ReLU()
                self.l2 = nn.Linear(20, 1)
                self.a2 = nn.Sigmoid()
                self.l3 = nn.Linear(1, 1)

            def forward(self, x):
                x = self.l1(x)
                x = self.a1(x)
                x = self.l2(x)
                x = self.a2(x)
                y = self.l3(x)
                return y

        torch_net = TorchNet()
        torch_mse = nn.MSELoss()
        torch_yhat = torch_net(torch.Tensor(self.x.data))
        torch_loss = torch_mse(torch_yhat, torch.Tensor(self.y.data))
        torch_loss.backward()

        torch_w1 = torch_net.l1.weight
        torch_b1 = torch_net.l1.bias
        torch_w2 = torch_net.l2.weight
        torch_b2 = torch_net.l2.bias
        torch_w3 = torch_net.l3.weight
        torch_b3 = torch_net.l3.bias

        mynet = self.create_net()
        # initial parameters with torch parameters to compare with torch results.
        w1 = mynet.l1.weight
        w1.set_values(torch_w1.data.numpy().transpose())
        b1 = mynet.l1.bias
        b1.set_values(torch_b1.data.numpy().reshape(1, -1))
        w2 = mynet.l2.weight
        w2.set_values(torch_w2.data.numpy().transpose())
        b2 = mynet.l2.bias
        b2.set_values(torch_b2.data.numpy().reshape(1, -1))
        w3 = mynet.l3.weight
        w3.set_values(torch_w3.data.numpy())
        b3 = mynet.l3.bias
        b3.set_values(torch_b3.data.numpy().reshape(1, -1))

        loss = layer.MSELayer()
        l = loss(mynet(self.x), self.y)
        l.backward()

        assert abs(l.data[0, 0] - float(torch_loss)) < 1e-6
        assert abs(np.linalg.norm(torch_w1.grad.data.numpy()) - np.linalg.norm(w1.grad)) < 1e-6
        assert abs(np.linalg.norm(torch_b1.grad.data.numpy()) - np.linalg.norm(b1.grad)) < 1e-6
        assert abs(np.linalg.norm(torch_w2.grad.data.numpy()) - np.linalg.norm(w2.grad)) < 1e-6
        assert abs(np.linalg.norm(torch_b2.grad.data.numpy()) - np.linalg.norm(b2.grad)) < 1e-6


if __name__ == '__main__':
    unittest.main()