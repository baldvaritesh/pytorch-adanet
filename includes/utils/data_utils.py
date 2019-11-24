import torch

from torch.utils.data import DataLoader, Subset
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


def load_cifar(batch_size, path="sandbox/data", classes=['deer', 'truck'], shuffle=True, **kwargs):
    
    class_ids = {'plane':0, 'car':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}
    class_filter = [class_ids[_class] for _class in classes]
    
    trainset = datasets.CIFAR10(
            path,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
    
    trainset_filter_indexes = []
    for i,sample in enumerate(trainset):
        label = sample[1]
        if label in class_filter:
            trainset_filter_indexes.append(i)
    trainset = Subset(trainset, trainset_filter_indexes)
    
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs
    )

    r_inf = float("-inf")
    for data, _ in train_loader:
        r_inf = max(torch.max(torch.abs(data)).item(), r_inf)

    testset = datasets.CIFAR10(
            path,
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )

    testset_filter_indexes = []
    for i,sample in enumerate(testset):
        label = sample[1]
        if label in class_filter:
            testset_filter_indexes.append(i)
    testset = Subset(testset, testset_filter_indexes)

    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs
    )

    return train_loader, test_loader, r_inf
