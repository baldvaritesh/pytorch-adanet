import torch
import logging
import numpy as np

from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from ..optimizers import SGD
from .model import Model, Network
from ..utils import RademacherComplexity


class AdaNet(Model):
    def __init__(
        self,
        name,
        loss_fn,
        activation_fn,
        input_dim,
        output_dim,
        width,
        n_iters,
        regularizer=None,
        r_inf=None,
        batch_size=None,
        gamma=1e-4,
    ):
        super(AdaNet, self).__init__(name, loss_fn)

        self.width = width
        self.n_iters = n_iters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_fn = activation_fn
        if regularizer:
            if r_inf is None:
                raise ValueError(
                    "The argument r_inf cannot be None if regularizer is not None"
                )
            if batch_size is None:
                raise ValueError(
                    "The argument batch_size cannot be None if regularizer is not None"
                )

            self.loss_fn = regularizer(
                model=self,
                dims=(batch_size, input_dim),
                r_inf=r_inf,
                loss_fn=loss_fn,
                gamma=gamma,
            )
        else:
            self.loss_fn = loss_fn

        self.network = Network(
            activation_fn, input_dim, [(input_dim, 1000)], output_dim
        )

    def generate_subnetwork(self, depth):
        dims = [(self.input_dim, self.width)]
        for ld in self.network.layers[: depth - 1]:
            dims.append((ld.W.shape[1] + self.width, self.width))

        return Network(self.activation_fn, self.input_dim, dims, self.output_dim)

    def train_subnetwork(
        self, subnetwork, optimizer, data_loader, epoch, device="cpu", **kwargs
    ):
        self.network.detach()

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
                        x = layers[i](
                            (net_outs[i - 1], self.subnet.activation_fn(x)), split=True
                        )

                output += torch.mm(x, self.subnet.weights)

                output = F.log_softmax(output, dim=1)

                return output

        model = Wrapper(self.network, subnetwork, self.input_dim).to(device)

        if isinstance(self.loss_fn, RademacherComplexity):
            self.loss_fn.update_complexities(subnetwork)

        optimizer.update_model(model)

        for i in range(self.n_iters):
            loss = 0
            for batch_idx, data in enumerate(data_loader):
                loss += optimizer.step(data, device=device, **kwargs)
            loss /= batch_idx

        return loss

    def add_subnetwork(self, subnetwork):
        layers = self.network.layers
        sublayers = subnetwork.layers

        if layers:
            for i in range(len(layers)):
                layer = layers[i]
                sublayer = sublayers[i]
                padding = torch.zeros(
                    (sublayer.W.shape[0] - layer.W.shape[0], layer.W.shape[1])
                )
                weights = torch.cat((layer.W.data.cpu(), padding), dim=0)

                layer.W.data = torch.cat((weights, sublayer.W.data.cpu()), dim=1)

        if len(sublayers) > len(layers):
            layers.append(sublayers[-1])
            self.network.exposed.append(layers[-1].W.shape[1])
        else:
            self.network.exposed[-1] = layers[-1].W.shape[1]

        self.network.weights.data = torch.cat(
            (self.network.weights.data.cpu(), subnetwork.weights.data.cpu()), dim=0
        )

    def train_step(
        self, optimizer, data_loader, epoch, device="cpu", log=False, **kwargs
    ):
        self.train()

        if epoch == 1:
            optimizer.update_model(self)
            for i in range(self.n_iters):
                loss = 0
                for batch_idx, data in enumerate(data_loader):
                    loss += optimizer.step(data, device=device, **kwargs)
                loss /= batch_idx

            self.prev_loss = loss

            return loss

        k = len(self.network.layers)
        candidate_networks = [self.generate_subnetwork(depth) for depth in [k, k + 1]]

        losses = list()
        for subnet in candidate_networks:
            losses.append(
                self.train_subnetwork(subnet, optimizer, data_loader, epoch, device)
            )

        best_subnet = np.argmin(losses)

        if self.prev_loss < losses[best_subnet]:
            return self.prev_loss

        if log:
            for i in range(len(candidate_networks)):
                logging.info(
                    "Train Epoch: {:3d} \t Candidate Network: {:3d} \t Loss: {:.6f} \t Chosen: {}".format(
                        epoch, i + 1, losses[i], i == best_subnet
                    )
                )

        self.add_subnetwork(candidate_networks[best_subnet])
        self.to(device)

        if log:
            logging.info(
                "Train Epoch: {:3d} \t Network: "
                + ",".join([str(l.W.shape[1]) for l in self.network.layers])
            )

        self.width = min(self.width * 2, 128)

        return losses[best_subnet]

    def forward(self, x):
        output = self.network(x)
        output = F.log_softmax(output, dim=1)

        return output
