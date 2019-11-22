import torch

from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, model, data_loader, lr=0.01, decay_rate=1.0, decay_steps=1000):
        super(SGD, self).__init__(model, data_loader, lr, decay_rate, decay_steps)

    def step(self, device="cpu", zero_grad=True):
        try:
            data, target = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.data_loader)
            data, target = next(self.data_iterator)

        data, target = data.to(device), target.to(device)

        output = self.model(data)

        loss = self.loss_fn(output, target)
        loss.backward()

        for param in self.params:
            param.data.add_(-self.lr, param.grad.data)

            if zero_grad:
                param.grad.detach_()
                param.grad.zero_()

        self.current_step += 1
        if self.decay_rate < 1.0 and self.current_step % self.decay_steps:
            self.lr *= self.decay_rate

        return loss.item()
