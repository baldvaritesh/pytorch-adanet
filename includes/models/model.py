import torch
import logging

from torch import nn


class Model(nn.Module):
    def __init__(self, name, loss_fn):
        super(Model, self).__init__()

        self.name = name
        self.loss_fn = loss_fn

    def train_step(self, optimizer, epoch, device="cpu", log_interval=0, **kwargs):
        self.train()

        loss = optimizer.step(device=device, **kwargs)

        if log_interval > 0 and epoch % log_interval == 0:
            logging.info("Train Epoch: {:3d} Loss: {:.6f}".format(epoch, loss))

    def test_step(self, data_loader, epoch, device="cpu", log_interval=0):
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

        if log_interval > 0 and epoch % log_interval == 0:
            logging.info(
                "Test  Epoch: {:3d}, Loss: {:.6f}, Accuracy: {:.4f}".format(
                    epoch, loss, 100 * correct / len(data_loader.dataset)
                )
            )

        return loss, correct / len(data_loader.dataset)
