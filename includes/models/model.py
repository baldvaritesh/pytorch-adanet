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

    def test_step(self, test_loader, device="cpu", logger=None, log_interval=100):
        self.eval()

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                output = self(data)

                test_loss += self.loss_fn(output, target, reduction="sum").item()

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        if logger:
            logger.info(
                "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                    test_loss,
                    correct,
                    len(test_loader.dataset),
                    100.0 * correct / len(test_loader.dataset),
                )
            )

        return test_loss, correct / len(test_loader.dataset)
