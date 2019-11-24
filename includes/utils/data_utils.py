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

    r_inf = float("-inf")
    for data, _ in train_loader:
        r_inf = max(torch.max(torch.abs(data)).item(), r_inf)

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

    return train_loader, test_loader, r_inf


def load_cifar(batch_size, path="sandbox/data", shuffle=True, **kwargs):
    train_loader = DataLoader(
        datasets.CIFAR10(
            path,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs
    )

    r_inf = float("-inf")
    for data, _ in train_loader:
        r_inf = max(torch.max(torch.abs(data)).item(), r_inf)

    test_loader = DataLoader(
        datasets.CIFAR10(
            path,
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs
    )

    return train_loader, test_loader, r_inf
