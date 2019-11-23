import logging
import numpy as np


class RademacherComplexity:
    def __init__(self, layers, batch_dim, batch_max, dual=2, gamma=None):
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
        self._layers = layers
        self._batch_dim = batch_dim
        self._r_infty = batch_max
        self._dual = dual
        if gamma is None:
            self._gamma = 1.0
        else:
            self._gamma = gamma
        self._candidate_nodes_product_per_layer = None
        self._nodes_product_per_layer = None

    def initialize_candidate(self, new_layer_dims):
        """
        Initializes new architecture based on the dimensions given
        :param new_layer_dims: the new subnetwork of dimensions being added to network
        :return: None
        """
        self._candidate_nodes_product_per_layer = [self._batch_dim[1]]
        count = self._batch_dim[1]
        new_layer_width, new_layer_depth = new_layer_dims
        for layer in range(len(self._layers)):
            if new_layer_depth > 0:
                count *= self._layers[layer] + new_layer_width
            else:
                count *= self._layers[layer]
            new_layer_depth -= 1
            self._candidate_nodes_product_per_layer.append(count)
        while new_layer_depth is not 0:
            count *= new_layer_width
            self._candidate_nodes_product_per_layer.append(count)
            new_layer_depth -= 1

    def get_candidate_nodes_per_layer(self):
        return self._candidate_nodes_product_per_layer

    def set_nodes_product_per_layer(self, chosen_candidate_nodes_per_layer):
        self._nodes_product_per_layer = chosen_candidate_nodes_per_layer

    def calculate_complexity(self, layer, candidate=True):
        n_layer = None
        if candidate:
            n_layer = self._candidate_nodes_product_per_layer[layer - 1]
        else:
            n_layer = self._nodes_product_per_layer[layer - 1]
        return (
            self._r_infty
            * self._gamma
            * np.power(n_layer, 1 / self._dual)
            * np.sqrt(np.log(2 * self._batch_dim[1]) / (2 * self._batch_dim[0]))
        )
