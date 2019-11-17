from src.utils.data_utils import MNISTBinaryData
import torch

mnist_train = MNISTBinaryData(
    root="../files", train=True, download=False, transform=None
)
mnist_test = MNISTBinaryData(
    root="../files", train=False, download=False, transform=None
)


class OneLayerNetA(torch.nn.Module):
    def __init__(self, data_in, h, data_out):
        # Define a simple one hidden layer network
        super(OneLayerNetA, self).__init__()
        self.linear1 = torch.nn.Linear(data_in, h)
        self.linear2 = torch.nn.Linear(h, data_out)

    def forward(self, x):
        h_relu1 = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu1).clamp(min=0)
        return y_pred


class OneLayerNetB(torch.nn.Module):
    def __init__(self, data_in, h, data_out):
        super(OneLayerNetB, self).__init__()
        self.linear1 = torch.nn.Linear(data_in, 2 * h)
        self.linear2 = torch.nn.Linear(2 * h, data_out)

    def forward(self, x):
        h_relu1 = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu1).clamp(min=0)
        return y_pred


class TwoLayerNetA(torch.nn.Module):
    def __init__(self, data_in, h, data_out):
        # Define a 2 layer network adanet style
        super(TwoLayerNetA, self).__init__()
        self.linear1 = torch.nn.Linear(data_in, h)
        self.linear2 = torch.nn.Linear(data_in, h)
        self.linear3 = torch.nn.Linear(2 * h, data_out)

    def forward(self, x):
        h_relu1 = self.linear1(x).clamp(min=0)
        h_relu2 = self.linear2(x).clamp(min=0)
        y_pred = self.linear3(torch.cat((h_relu1, h_relu2), 1)).clamp(min=0)
        return y_pred


def adanet_loss(output, target, model_depth):
    # Adding the Rademacher complexity
    return torch.mean((output - target) ** 2) + 0.5 * torch.sqrt(model_depth)


# Time to train the models one by one
N, D_in, H, D_out = 60000, 784, 64, 1
model = TwoLayerNetA(D_in, H, D_out)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(5000):
    print(t)
    y_p = model(mnist_train.data)
    loss = adanet_loss(y_p, mnist_train.targets, torch.tensor(2, dtype=torch.float))

    if t % 100 == 0:
        print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# print(model.parameters())
