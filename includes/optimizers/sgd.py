import torch

from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, model, lr=0.01, decay_rate=1.0, decay_steps=0):
        super(SGD, self).__init__(model, lr, decay_rate, decay_steps)

    def step(self, data, device="cpu", zero_grad=True):
        data, target = data[0].to(device), data[1].to(device)

        output = self.model(data)

        loss = self.loss_fn(output, target)
        loss.backward()

        for param in self.params:
            if not param.requires_grad or param.grad is None:
                continue

            param.data.add_(-self.lr, param.grad.data)

            if zero_grad:
                param.grad.detach_()
                param.grad.zero_()

        self.current_step += 1
        if (
            self.decay_rate < 1.0
            and self.decay_steps != 0
            and self.current_step % self.decay_steps
        ):
            self.lr *= self.decay_rate

        return loss.item()
