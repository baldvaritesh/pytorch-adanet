import torch

from torch import nn
from torch.nn import functional as F

from .model import Model


class NN(Model):
    def __init__(self, name, loss_fn, input_dim, output_dim):
        super(NN, self).__init__(name, loss_fn)

        # self.dropout1 = nn.Dropout2d(0.25)
        # self.dropout2 = nn.Dropout2d(0.5)

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048, 1984)
        self.fc3 = nn.Linear(1984, 1920)
        self.fc4 = nn.Linear(1920, 1792)
        self.fc5 = nn.Linear(1792, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)

        x = self.fc5(x)

        # x = self.dropout2(x)

        output = F.log_softmax(x, dim=1)

        return output
