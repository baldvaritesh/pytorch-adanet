import torch
import numpy as np

from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from .model import Model
from ..optimizers import SGD


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

        self.layer_dims = layer_dims
        self.layers = nn.ModuleList([Layer(dims[0], dims[1]) for dims in layer_dims])

        last_dim = layer_dims[-1][1]
        self.weights = nn.Parameter(
            torch.randn((last_dim, output_dim)) / np.sqrt(last_dim), requires_grad=True
        )

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
            x = layer(x)
            x = self.activation_fn(x)

            outs.append(x)

        output = outs[-1]
        for out in outs[-2::-1]:
            output = torch.cat((out[:, :-output.shape[1]], output), dim=1)

        output = torch.mm(output, self.weights)
        
        if return_all:
            return output, outs

        return output


class AdaNet(Model):
    def __init__(self, name, loss_fn, activation_fn, input_dim, output_dim, width=5):
        super(AdaNet, self).__init__(name, loss_fn)

        self.width = width
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_fn = activation_fn

        self.network = Network(
            activation_fn, input_dim, [[input_dim, width]], output_dim
        )

    def generate_subnetwork(self, depth):
        dims = [(self.input_dim, self.width)]
        for ld in self.network.layers[:depth - 1]:
            dims.append((ld.W.shape[1] + self.width, self.width))

        return Network(self.activation_fn, self.input_dim, dims, self.output_dim)

    def train_subnetwork(
        self, subnetwork, optimizer, epoch, n_iters=100, device="cpu", **kwargs
    ):
        # self.network.detach()

        class Wrapper(nn.Module):
            def __init__(self, net, subnet, input_dim):
                super(Wrapper, self).__init__()

                self.net = net
                self.subnet = subnet
                self.input_dim = input_dim

            def forward(self, x):
                x = x.reshape(-1, self.input_dim)

                output, net_outs = self.net(x, return_all=True)

                layers = self.subnet.layers

                x = layers[0](x)
                if net_outs:
                    for i in range(1, len(layers)):
                        x = layers[i]((net_outs[i - 1], x), split=True)
                        x = self.subnet.activation_fn(x)
                else:
                    x = self.subnet.activation_fn(x)

                output += torch.mm(x, self.subnet.weights)

                return output


        comb = Wrapper(self.network, subnetwork, self.input_dim).to(device)
        optimizer.update_model(comb)

        for i in range(n_iters):
            optimizer.step(device=device, **kwargs)

    def add_subnetwork(self, subnetwork):
        layers = self.network.layers
        sublayers = subnetwork.layers

        if layers:
            layers[0].W.data = torch.cat((layers[0].W.data, sublayers[0].W.data), dim=1)
            for i in range(1, len(layers)):
                layer = layers[i]

                padding = torch.zeros((self.width, layer.W.shape[1]))
                weights = torch.cat((layer.W.data.cpu(), padding), dim=0)

                layer.W.data = torch.cat((weights, sublayers[i].W.data.cpu()), dim=1)

        if len(sublayers) > len(layers):
            layers.append(sublayers[-1])

        self.network.weights.data = torch.cat(
            (self.network.weights.data.cpu(), subnetwork.weights.data.cpu()), dim=0
        )

    def train_step(
        self, optimizer, epoch, n_iters=1000, device="cpu", log_interval=0, **kwargs
    ):
        self.train()

        subnet = self.generate_subnetwork(3)
        self.train_subnetwork(subnet, optimizer, epoch, n_iters, device)
        self.add_subnetwork(subnet)

        self.to(device)

    def forward(self, x):
        output = self.network.forward(x)
        output = F.log_softmax(output, dim=1)

        return output
