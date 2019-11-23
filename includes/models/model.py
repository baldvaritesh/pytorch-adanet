import torch
import logging

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
