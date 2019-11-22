class Optimizer:
    def __init__(self, model, data_loader, lr, decay_rate, decay_steps):
        self.model = model
        self.loss_fn = model.loss_fn
        self.params = list(model.parameters())

        self.data_loader = data_loader
        self.data_iterator = iter(self.data_loader)

        self.lr = lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

        self.current_step = 0

    def step(self):
        raise NotImplementedError


class DefaultWrapper:
    def __init__(self, model, data_loader, optimizer, **kwargs):
        self.data_loader = data_loader
        self.data_iterator = iter(self.data_loader)

        self.model = model
        self.loss_fn = model.loss_fn
        self.optimizer = optimizer(model.parameters(), **kwargs)

    def step(self, device="cpu"):
        try:
            data, target = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.data_loader)
            data, target = next(self.data_iterator)

        data, target = data.to(device), target.to(device)

        output = self.model(data)

        self.optimizer.zero_grad()

        loss = self.loss_fn(output, target)
        loss.backward()

        self.optimizer.step()

        return loss.item()
