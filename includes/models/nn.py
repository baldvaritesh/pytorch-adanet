import torch
import numpy as np

from torch import nn
from torch.nn import functional as F

from .model import Model, Network


class NN(Model):
    def __init__(
        self,
        name,
        loss_fn,
        input_dim,
        output_dim,
        regularizer=None,
        r_inf=None,
        batch_size=None,
        gamma=1e-4,
    ):
        super(NN, self).__init__(name, loss_fn)

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

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.network = Network(
            # F.relu, input_dim, [(input_dim, 128), (128, 120)], output_dim
            F.relu,
            input_dim,
            [
                (input_dim, 768),
                (768, 760),
                (760, 752),
                (752, 704),
                (704, 640),
                (640, 256),
            ],
            output_dim,
        )
        self.network.exposed = [8, 8, 48, 64, 384, 256]
        self.network.weights = nn.Parameter(
            torch.randn((768, output_dim)) / np.sqrt(768), requires_grad=True
        )

    def forward(self, x):
        output = self.network(x)
        output = F.log_softmax(output, dim=1)

        return output
