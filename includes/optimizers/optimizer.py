from functools import partial


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
        self.data_iterator = iter(self.data_loader)

        self.model = model
        self.params = list(model.parameters())

        self.loss_fn = model.loss_fn

        self.optimizer_fn = partial(optimizer, **kwargs)
        self.optimizer = optimizer(self.params, **kwargs)

    def update_model(self, model):
        self.model = model
        # new_params = [param for param in model.parameters() if param not in self.params]
        # self.optimizer.add_param_group({"params": new_params})
        # self.params.extend(new_params)

        self.optimizer = self.optimizer_fn(model.parameters())

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
