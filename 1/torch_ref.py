import numpy as np
import torch


def torch_loss_and_grads(params, X, y):

    torch.set_default_dtype(torch.float64)
    net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3))

    with torch.no_grad():

        net[0].weight.copy_(torch.from_numpy(params["W1"].T.copy()))
        net[0].bias.copy_(torch.from_numpy(params["b1"].copy()))
        net[2].weight.copy_(torch.from_numpy(params["W2"].T.copy()))
        net[2].bias.copy_(torch.from_numpy(params["b2"].copy()))

    Xt = torch.from_numpy(X.copy())
    yt = torch.from_numpy(y.copy())
    loss = torch.nn.functional.cross_entropy(net(Xt), yt)  # mean за замовчуванням
    net.zero_grad()
    loss.backward()

    grads = {
        "W1": net[0].weight.grad.numpy().T.copy(),
        "b1": net[0].bias.grad.numpy().copy(),
        "W2": net[2].weight.grad.numpy().T.copy(),
        "b2": net[2].bias.grad.numpy().copy(),
    }

    return float(loss.item()), grads