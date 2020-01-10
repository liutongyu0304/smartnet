# coding=utf-8
import smartnet as sn
import smartnet.optims as optims
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


"""
# description:
    for unconstrained optimization:
    y = x1 ** 2 + 2 * x2 ** 2 - 2 * x1 * x2 - 4 * x1
    optimal point:[4, 2]
    
    optimize this problem by smartnet and draw trajectory of middle points.
"""


optim_options = {"lr": 0.02, "weight_decay": 0,
                 "momentum": 0.9, "beta": 0.999, "eps": 1e-8}


def create_optimizer(name, parameters, options):
    if name == "sgd":
        return optims.SGDOptim(parameters,
                               lr=options["lr"], weight_decay=options["weight_decay"])
    elif name == "momentum":
        return optims.MomentumOptim(parameters,
                                    lr=options["lr"], weight_decay=options["weight_decay"],
                                    momentum=options["momentum"])
    elif name == "rmsprop":
        return optims.RMSPropOptim(parameters, lr=options["lr"], weight_decay=options["weight_decay"],
                                   beta=options["beta"])
    elif name == "adam":
        return optims.AdamOptim(parameters, lr=options["lr"], weight_decay=options["weight_decay"],
                                beta=options["beta"], momentum=options["momentum"])


n = 50


def optimize_and_store(name="sgd"):
    storage = np.zeros((n, 2))

    x = sn.ones((2, 1), requires_grad=True)
    opt = create_optimizer(name, {"x": x}, optim_options)
    for i in range(n):
        storage[i] = x.data.reshape((-1,))
        x1, x2 = x[0], x[1]
        y = x1 ** 2 + 2 * x2 ** 2 - 2 * x1 * x2 - 4 * x1
        y.backward()
        opt.step()
        opt.zero_grad()
    return storage


def draw_contour():
    m = 100
    x, y = np.meshgrid(np.linspace(0, 6, m),
                       np.linspace(0, 4, m))
    z = x ** 2 + 2 * y ** 2 - 2 * x * y - 4 * x

    contour = plt.contour(x, y, z, 15, linewidths=0.5)
    plt.clabel(contour, inline_spacing=1.0, fmt='%.1f', fontsize=8)
    ax = plt.scatter([4], [2])
    return ax


def draw_opt(storage, ax):
    def update(i):
        for st in storage:
            data = st["data"]
            x1, y1 = data[i, 0], data[i, 1]
            delta_x, delta_y = data[i + 1, 0] - x1, data[i + 1, 1] - y1
            plt.arrow(x1, y1, delta_x, delta_y, head_width=0.05, head_length=0.1, color=st["color"])

    anim = FuncAnimation(ax.figure, update, frames=np.arange(0, n-1), interval=200)
    # anim.save('line.gif', dpi=80, writer='imagemagick')
    plt.show()


def optimize_and_show():
    ax = draw_contour()
    names = ["sgd", "momentum", "rmsprop", "adam"]
    colors = ["red", "blue", "black", "green"]
    storage = list()
    for i in range(len(names)):
        storage.append({"name": names[i], "color": colors[i], "data": optimize_and_store(names[i])})

    draw_opt(storage, ax)


if __name__ == "__main__":
    optimize_and_show()
