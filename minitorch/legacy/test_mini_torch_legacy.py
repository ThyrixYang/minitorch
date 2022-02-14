import numpy as np
import jax.numpy as jnp
from jax import grad
import jax.nn as jnn
import matplotlib.pyplot as plt

import mini_torch as mtorch


def jax_loss(x, ws, bs, y):
    x1 = jnp.add(jnp.matmul(x, ws[0]), bs[0])
    x1 = jnn.relu(x1)
    x2 = jnp.add(jnp.matmul(x1, ws[1]), bs[1])
    x2 = jnn.relu(x2)
    x3 = jnp.add(jnp.matmul(x2, ws[2]), bs[2])
    x3 = jnp.add(jnn.relu(x3), x1)
    x4 = jnp.add(jnp.matmul(x3, ws[3]), bs[3])
    loss = jnp.mean(jnp.pow(x4 - y, 2))
    return loss


class MLP:

    def __init__(self, input_size, output_size):
        self.W1 = mtorch.Tensor(np.random.uniform(-1, 1,
                                                  size=(input_size, 32)), requires_grad=True)
        self.b1 = mtorch.Tensor(np.zeros((1, 32)), requires_grad=True)
        self.W2 = mtorch.Tensor(
            np.random.uniform(-1, 1, size=(32, 32)), requires_grad=True)
        self.b2 = mtorch.Tensor(np.zeros((1, 32)), requires_grad=True)
        self.W3 = mtorch.Tensor(
            np.random.uniform(-1, 1, size=(32, 32)), requires_grad=True)
        self.b3 = mtorch.Tensor(np.zeros((1, 32)), requires_grad=True)
        self.W4 = mtorch.Tensor(np.random.uniform(-1, 1,
                                                  size=(32, output_size)), requires_grad=True)
        self.b4 = mtorch.Tensor(np.zeros((1, output_size)), requires_grad=True)
        self.params = [self.W1, self.W2, self.W3, self.W4,
                       self.b1, self.b2, self.b3, self.b4]

    def forward(self, x):
        x1 = mtorch.add(mtorch.matmul(x, self.W1), self.b1)
        x1 = mtorch.relu(x1)
        x2 = mtorch.add(mtorch.matmul(x1, self.W2), self.b2)
        x2 = mtorch.relu(x2)
        x3 = mtorch.add(mtorch.matmul(x2, self.W3), self.b3)
        x3 = mtorch.add(mtorch.relu(x3), x1)
        x4 = mtorch.add(mtorch.matmul(x3, self.W4), self.b4)
        return x4


def test_regression():
    np.random.seed(0)
    model = MLP(input_size=2, output_size=2)
    x = mtorch.Tensor(np.random.uniform(-1, 1, size=(200, 2)))
    y = np.zeros_like(x.np)
    y[:, 0] = np.sin(x.np[:, 0] * 10)
    y[:, 1] = np.cos(x.np[:, 1] * 10)
    y = mtorch.Tensor(y)
    optimizer = mtorch.SGDOptimizer(
        model.params, lr=5e-3, weight_decay=0, momentum=0.9)
    for i in range(100000):
        _y = model.forward(x)
        diff_y = mtorch.pow(mtorch.sub(_y, y), 2)
        loss = mtorch.mean(diff_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print("step {} loss_value {}".format(i, loss.np))
    plt.scatter(x.np[:, 0], _y.np[:, 0], label="pred")
    plt.scatter(x.np[:, 0], y.np[:, 0], label="true")
    plt.legend()
    plt.savefig("./tmp/sin.jpg")


def test_grad():
    np.random.seed(0)
    model = MLP(input_size=32, output_size=1)
    x = mtorch.Tensor(np.random.randn(100, 32))
    _y = model.forward(x)
    _y = mtorch.pow(_y, 2)
    loss = mtorch.mean(_y)
    loss.backward()
    print("loss_value ", loss.np)

    ws = [model.W1.np, model.W2.np, model.W3.np, model.W4.np]
    bs = [model.b1.np, model.b2.np, model.b3.np, model.b4.np]
    jax_loss_value = jax_loss(x.np, ws, bs)
    grad_ws_fn = grad(jax_loss, argnums=1)
    grads_ws = grad_ws_fn(x.np, ws, bs)
    grad_bs_fn = grad(jax_loss, argnums=2)
    grads_bs = grad_bs_fn(x.np, ws, bs)

    for i in range(4):
        diff = np.mean((model.params[i].grad - grads_ws[i])**2)
        print("w{}, diff {}".format(i, diff))
    for i in range(4):
        diff = np.mean((model.params[i + 4].grad - grads_bs[i])**2)
        print("b{}, diff {}".format(i, diff))
    # print(grads_ws[0])
    print("jax_loss_value ", jax_loss_value)
    exit()
    grad_jax_loss = grad(jax_loss)
    # print(loss.np)
    # print(loss.np)
    # exit()
    # loss.backward()
    # print(_y.np.shape)


def optim_test():
    np.random.seed(0)
    model = MLP(input_size=32, output_size=1)
    x = mtorch.Tensor(np.random.randn(100, 32))
    optimizer = mtorch.SGDOptimizer(model.params, lr=1e-5, weight_decay=1e-5)
    for i in range(1000):
        _y = model.forward(x)
        _y = mtorch.pow(_y, 2)
        loss = mtorch.mean(_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("loss_value ", loss.np)


if __name__ == "__main__":
    test_regression()
    # optim_test()
