import logging
import numpy as np


class RademacherComplexity:
    def __init__(self, model, input_dim, input_max, dual=2, gamma=1.0, lmbda=1.0):
        """
        Params to define Rademacher complexity for the nn module
        :param layers: Input Dimensions for every layer
        :param batch_dim: Tuple for batch size: m x n0
        :param batch_max: Absolute maximum value in the batch sample: r_infty
        :param dual: conjugate of p
        :param gamma: Fixed param
        """
        if layers is None or batch_dim is None or len(layers) == 0:
            logging.error("Initializing Radmacher complexity with invalid values")
            raise ValueError

        self._dual = dual
        self._model = model
        self._r_infty = input_max
        self._batch_dim = input_dim
        self._gamma = gamma
        self.n_k = [input_dim[1]]

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
        for i in range(len(model.net.layers)):
            layer_n_k.append(
                (model.net.layers[i][0] + model.subnet.layers[i][0]) * layer_n_k[i]
            )

        if len(model.subnet.layers) > len(model.net.layers):
            layer_n_k.append(model.subnet.layers[-1][0] * layer_n_k[-1])

    def total_complexity(self):
        comp = 0
        n = len(self.model.net.layers)
        start = 0
        end = self.model.net.exposed[0]
        for i in range(n):
            comp += torch.sum(
                torch.abs(self._model.net.weights[start:end])
            ) * self.calculate_complexity(self.n_k[i])

            start = self.model.net.exposed[i]
            end = start + self.model.net.exposed[(i + 1) % n]

        comp += torch.sum(
            torch.abs(self._model.subnet.weights)
        ) * self.calculate_complexity(self.n_k[-1])

        return comp

    def __call__(self, output, target):

        return self.loss_fn(output, target) + self.total_complexity()
