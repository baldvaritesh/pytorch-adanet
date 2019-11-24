import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_mnist(batch_size, path="sandbox/data", shuffle=True, **kwargs):
    train_loader = DataLoader(
        datasets.MNIST(
            path,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs
    )

    input_max = float("-inf")
    for data, _ in train_loader:
        input_max = max(torch.max(torch.abs(data)).item(), input_max)

    test_loader = DataLoader(
        datasets.MNIST(
            path,
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs
    )

    return train_loader, test_loader, input_max
