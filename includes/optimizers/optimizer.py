class Optimizer:
    def __init__(self, model, lr, decay_rate, decay_steps):
        self.model = model
        self.loss_fn = model.loss_fn
        self.params = list(model.parameters())

        self.lr = lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

        self.current_step = 0

    def update_model(self, model):
        self.model = model
        self.params = list(model.parameters())

    def step(self):
        raise NotImplementedError


class DefaultWrapper:
    def __init__(self, model, optimizer, **kwargs):
        self.model = model
        self.loss_fn = model.loss_fn
        self.optimizer = optimizer(model.parameters(), **kwargs)

    def update_model(self, model):
        raise NotImplementedError

    def step(self, data, device="cpu"):
        data, target = data[0].to(device), data[1].to(device)

        output = self.model(data)

        self.optimizer.zero_grad()

        loss = self.loss_fn(output, target)
        loss.backward()

        self.optimizer.step()

        return loss.item()
