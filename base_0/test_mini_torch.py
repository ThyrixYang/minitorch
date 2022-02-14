from typing import List
import numpy as np

import mini_torch as mtorch

def numerical_grad(function, inputs: List[np.ndarray], eps: float=1e-6):
    grads = []
    for i in range(len(inputs)):
        flat_x = inputs[i].reshape(-1)
        shape = inputs[i].shape
        grad = np.zeros_like(flat_x)
        for j in range(len(flat_x)):
            perturb_x = np.copy(flat_x)
            perturb_x[j] += eps
            inputs[i] = perturb_x.reshape(shape)
            f_plus = function(inputs)
            perturb_x[j] -= 2 * eps
            inputs[i] = perturb_x.reshape(shape)
            f_minus = function(inputs)
            perturb_x[j] += eps
            inputs[i] = perturb_x.reshape(shape)
            grad[j] = (f_plus - f_minus) / (2 * eps)
        grad = grad.reshape(shape)
        grads.append(grad)
    return grads


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
    
def test_loss_function(inputs):
    np.random.seed(0)
    model = MLP(input_size=32, output_size=10)
    x = mtorch.Tensor(np.random.randn(100, 32))
    for i in range(len(model.params)):
        model.params[i].np = inputs[i]
    _y = model.forward(x)
    _y = mtorch.pow(_y, 2)
    loss = mtorch.mean(_y)
    return loss.np

def test_grad():
    np.random.seed(0)
    model = MLP(input_size=32, output_size=10)
    x = mtorch.Tensor(np.random.randn(100, 32))
    _y = model.forward(x)
    _y = mtorch.pow(_y, 2)
    loss = mtorch.mean(_y)
    loss.backward()
    grad = [p.grad for p in model.params]
    
    inputs = [t.np for t in model.params]
    num_grads = numerical_grad(test_loss_function, inputs)
    
    for i in range(len(grad)):
        dist = np.mean(np.abs(num_grads[i] - grad[i]))
        print(dist)

if __name__ == "__main__":
    test_grad()
