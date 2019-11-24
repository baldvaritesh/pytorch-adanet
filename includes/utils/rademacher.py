import logging
import numpy as np
import torch

from torch.nn import functional as F


class RademacherComplexity:
    def __init__(
        self,
        model=None,
        input_dim=None,
        input_max=None,
        loss_fn=None,
        dual=2,
        gamma=1e-8,
    ):
        """
        Params to define Rademacher complexity for the nn module
        :param layers: Input Dimensions for every layer
        :param batch_dim: Tuple for batch size: m x n0
        :param batch_max: Absolute maximum value in the batch sample: r_infty
        :param dual: conjugate of p
        :param gamma: Fixed param
        """
        self._dual = dual
        self._model = model
        self._r_infty = input_max
        self._batch_dim = input_dim
        self._gamma = gamma
        self._loss_fn = loss_fn
        if input_dim is not None:
            self.n_k = [input_dim[1]]
        self.subnets = False

    def set_subnets(self):
        self.subnets = True

    def calculate_complexity(self, n_layer):
        return (
            self._r_infty
            * self._gamma
            * np.power(n_layer, 1 / self._dual)
            * np.sqrt(np.log(2 * self._batch_dim[1]) / (2 * self._batch_dim[0]))
        )

    def update_parameters(self, model):
        self._model = model

        self.n_k = [self._batch_dim[1]]
        layers = self._model.net.layers

        for i in range(len(layers)):
            self.n_k.append(
                (model.net.layers[i].W.shape[0] + model.subnet.layers[i].W.shape[0])
                * self.n_k[i]
            )

        if len(model.subnet.layers) > len(model.net.layers):
            self.n_k.append(model.subnet.layers[-1].W.shape[0] * self.n_k[-1])

    def total_complexity(self):
        comp = 0
        if not self.subnets:
            comp = torch.sum(
                torch.abs(self._model.network.weights)
            ) * self.calculate_complexity(self.n_k[0])
            return comp

        n = len(self._model.net.layers)
        start = 0
        end = self._model.net.exposed[0]
        for i in range(n):
            comp += torch.sum(
                torch.abs(self._model.net.weights[start:end])
            ) * self.calculate_complexity(self.n_k[i])

            start = self._model.net.exposed[i]
            end = start + self._model.net.exposed[(i + 1) % n]

        comp += torch.sum(
            torch.abs(self._model.subnet.weights)
        ) * self.calculate_complexity(self.n_k[-1])

        return comp

    def __call__(self, output, target, reduction="sum"):

        return (
            self._loss_fn(output, target, reduction="sum").item()
            + self.total_complexity()
        )
