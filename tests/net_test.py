from smartnet import *
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
        cls.x = SmartTensor(np.random.rand(nsamples, 10), requires_grad=False)
        cls.y = SmartTensor(np.random.rand(nsamples, 1), requires_grad=False)
        cls.y.data[:] = np.matmul(cls.x.data, 0.2 * np.ones((10, 1))) + 0.1
        cls.label = SmartTensor(np.random.rand(nsamples, 3), requires_grad=False)
        print("begin to test:")
        pass

    @classmethod
    def tearDownClass(cls):
        print("end of test.")

    def create_net(self):
        net = SmartNet()
        net.add_layer(SmartDataLayer("input"), self.x)
        net.add_layer(SmartLinearLayer("linear1", 10, 5))
        net.add_layer(SmartSigmoidLayer("sigmoid1"))
        net.add_layer(SmartLinearLayer("linear2", 5, 1))
        net.add_layer(SmartMSELayer("mse"), self.y)
        return net

    def create_opt(self, net):
        opt = SmartSGDOptim("sgd", net.trainable_parameters(), lr=0.01, weight_decay=0.0001)
        opt.zero_grad()
        loss = net.forward()
        net.backward()
        opt.step()
        return opt

    def test_smartnet(self):
        net = self.create_net()
        opt = self.create_opt(net)

        for i in range(1000):
            net.zero_grad()
            loss = net.forward()
            if i % 10 == 0:
                print("the {}th period, loss = {}".format(i, loss))
            net.backward()
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
                self.l1 = nn.Linear(10, 5)
                self.a1 = nn.Sigmoid()
                self.l2 = nn.Linear(5, 1)
                pass

            def forward(self, x):
                x = self.l1(x)
                x = self.a1(x)
                x = self.l2(x)
                return x

        torch_net = TorchNet()
        torch_mse = nn.MSELoss()
        torch_yhat = torch_net(torch.Tensor(self.x.data))
        torch_loss = torch_mse(torch_yhat, torch.Tensor(self.y.data))
        torch_loss.backward()

        torch_w1 = torch_net.l1.weight
        torch_b1 = torch_net.l1.bias
        torch_w2 = torch_net.l2.weight
        torch_b2 = torch_net.l2.bias

        mynet = self.create_net()
        # initial parameters with torch parameters to compare with torch results.
        w1 = mynet.get_layer("linear1").weight
        w1.data[:] = torch_w1.data.numpy().transpose()
        b1 = mynet.get_layer("linear1").bias
        b1.data[:] = torch_b1.data.numpy()
        w2 = mynet.get_layer("linear2").weight
        w2.data[:] = torch_w2.data.numpy().transpose()
        b2 = mynet.get_layer("linear2").bias
        b2.data[:] = torch_b2.data.numpy()

        loss = mynet.forward()
        mynet.backward()

        assert abs(loss - float(torch_loss)) < 1e-6
        assert abs(np.linalg.norm(torch_w1.grad.data.numpy()) - np.linalg.norm(w1.grad)) < 1e-6
        assert abs(np.linalg.norm(torch_b1.grad.data.numpy()) - np.linalg.norm(b1.grad)) < 1e-6
        assert abs(np.linalg.norm(torch_w2.grad.data.numpy()) - np.linalg.norm(w2.grad)) < 1e-6
        assert abs(np.linalg.norm(torch_b2.grad.data.numpy()) - np.linalg.norm(b2.grad)) < 1e-6


if __name__ == '__main__':
    unittest.main()