import torch
from torchvision.datasets import MNIST


class MNISTBinaryData(MNIST):
    def __init__(
        self,
        root="../files",
        train=False,
        download=False,
        transform=None,
        labels=[0, 1],
    ):
        assert len(labels) == 2
        # Make it better to not re-download data if it is present
        super(MNISTBinaryData, self).__init__(
            root=root, train=train, download=download, transform=transform
        )
        self._filter_according_to_labels(labels)

    def _filter_according_to_labels(self, labels):
        # reshape data to correct shape
        self.data = self.data.reshape(-1, 784)
        mask = (self.targets == labels[0]) | (self.targets == labels[1])
        self.data = self.data[mask]
        self.targets = self.targets[mask]
        # Convert to double
        self.data = self.data.type(torch.float)
        self.targets = self.targets.type(torch.float)
