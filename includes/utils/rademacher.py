import logging
import numpy as np
import torch

from torch.nn import functional as F


class RademacherComplexity:
    """
    This class implements RademacherComplexity which acts as a proxy to a regularizer
    in the learning of AdaNet. It computes an upper bound to the complexity layer by
    layer. Moreover, this can also act as a wrapper to a general loss function by
    adding a sparse regularization term to the loss computation.

    Args:
        input_dim : Tuple for batch size: m x n_0
        r_inf     : Maximum absolute value accross dimensions in the data
        dual      : conjugate of p (default: 2)
        gamma     : Fixed param
    """

    def __init__(self, model, dims, r_inf, loss_fn=None, dual=2, gamma=2e-3):
        self.dual = dual

        self.model = model
        self.subnetwork = None

        self.gamma = gamma * r_inf * np.sqrt(np.log(2 * dims[1]) / (2 * dims[0]))

        self.loss_fn = loss_fn

    def compute_layer_complexity(self, n_layer):
        return self.gamma * np.power(n_layer, 1 / self.dual)

    def update_complexities(self, subnetwork=None):
        self.subnetwork = subnetwork

        n_layers = [1]
        layers = subnetwork.layers if subnetwork else self.model.network.layers

        for layer in layers:
            n_layers.append(layer.W.shape[0] * n_layers[-1])

        self.complexities = list(map(self.compute_layer_complexity, n_layers[1:]))

    def total_complexity(self):
        if not hasattr(self, "complexities"):
            self.update_complexities()

        l1 = lambda x: torch.sum(torch.abs(x))

        net = self.model.network

        s = 0
        comp = 0
        for i, e in enumerate(net.exposed):
            comp += l1(net.weights[s:e]) * self.complexities[i]
            s = e

        if self.subnetwork:
            comp += l1(self.subnetwork.weights) * self.complexities[-1]

        return comp

    def __call__(self, output, target, reduction="mean"):
        loss = self.total_complexity()

        if self.loss_fn is not None:
            loss.add_(self.loss_fn(output, target, reduction=reduction))

        return loss
