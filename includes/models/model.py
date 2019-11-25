import torch
import logging
import numpy as np

from torch import nn


class Model(nn.Module):
    def __init__(self, name, loss_fn):
        super(Model, self).__init__()

        self.name = name
        self.loss_fn = loss_fn

    def train_step(
        self, optimizer, data_loader, epoch, device="cpu", log=False, **kwargs
    ):
        self.train()

        loss = 0
        for batch_idx, data in enumerate(data_loader):
            loss += optimizer.step(data, device=device, **kwargs)
        loss /= batch_idx

        return loss

    def test_step(self, data_loader, epoch, device="cpu", log=False):
        self.eval()

        loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)

                output = self(data)

                loss += self.loss_fn(output, target, reduction="sum").item()

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        loss /= len(data_loader.dataset)

        return loss, correct / len(data_loader.dataset)


class Layer(nn.Module):
    def __init__(self, input_dim, output_dim, weights=None):
        super(Layer, self).__init__()

        if weights is None:
            self.W = nn.Parameter(
                torch.randn(input_dim, output_dim) / np.sqrt(input_dim),
                requires_grad=True,
            )
        else:
            assert weights.shape == torch.size((input_dim, output_dim))
            self.W = nn.Parameter(weights)

    def forward(self, data, split=False):
        if split:
            a, b = data

            split = a.shape[1]
            return torch.mm(a, self.W[:split, ...]) + torch.mm(b, self.W[split:, ...])

        return torch.mm(data, self.W)


class Network(nn.Module):
    def __init__(self, activation_fn, input_dim, layer_dims, output_dim):
        super(Network, self).__init__()

        self.input_dim = input_dim
        self.activation_fn = activation_fn

        self.layers = nn.ModuleList([Layer(dims[0], dims[1]) for dims in layer_dims])

        last_dim = layer_dims[-1][1]
        self.weights = nn.Parameter(
            torch.randn((last_dim, output_dim)) / np.sqrt(last_dim), requires_grad=True
        )
        self.exposed = [0] * (len(layer_dims) - 1) + [last_dim]

    def detach(self):
        for param in self.parameters():
            param.requires_grad = False
            param.grad = None

    def forward(self, x, return_all=False):
        if not self.layers:
            if return_all:
                return 0, []
            else:
                return 0

        x = x.reshape(-1, self.input_dim)

        outs = list()
        for layer in self.layers:
            x = layer(self.activation_fn(x))

            outs.append(x)

        output = outs[-1][:, : self.exposed[-1]]
        for i in range(len(outs) - 2, -1, -1):
            output = torch.cat((outs[i][:, : self.exposed[i]], output), dim=1)

        output = torch.mm(output, self.weights)

        if return_all:
            return output, outs

        return output
