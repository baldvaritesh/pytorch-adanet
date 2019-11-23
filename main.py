from __future__ import print_function, division

import torch
import logging
import argparse

from tqdm import tqdm
from torch import optim
from torch.nn import functional as F

from includes.utils import data_utils
from includes.models import CNN, AdaNet
from includes.optimizers import SGD, DefaultWrapper


parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
parser.add_argument(
    "--dataset",
    default="mnist",
    choices=["mnist"],
    help="set dataset (default: %(default)s)",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    metavar="N",
    help="input batch size for training (default: %(default)s)",
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=1000,
    metavar="N",
    help="input batch size for testing (default: %(default)s)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    metavar="N",
    help="number of epochs to train (default: %(default)s)",
)
parser.add_argument(
    "--optimizer",
    default="sgd",
    choices=["sgd", "d-sgd"],
    help="set optimizer (default: %(default)s)",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.01,
    metavar="LR",
    help="learning rate (default: %(default)s)",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.7,
    metavar="M",
    help="Learning rate step gamma (default: 0.7)",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "--seed",
    type=int,
    default=1,
    metavar="S",
    help="random seed (default: %(default)s)",
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=0,
    metavar="N",
    help="how many batches to wait before logging training status (default: %(default)s = no logging)",
)
parser.add_argument(
    "--save-model",
    action="store_true",
    default=False,
    help="for Saving the current model",
)
parser.add_argument(
    "--load-model",
    action="store_true",
    default=False,
    help="for loading a saved model",
)


def main(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    if args.dataset == "mnist":
        train_loader, test_loader = data_utils.load_mnist(args.batch_size, **kwargs)
    else:
        raise NotImplementedError

    # model = CNN("cnn", loss_fn=F.nll_loss).to(device)
    model = AdaNet(
        "adanet",
        loss_fn=F.nll_loss,
        activation_fn=F.relu,
        input_dim=784,
        output_dim=10,
    )

    model = model.to(device)

    model_path = "sandbox/models/{}_{}.pt".format(model.name, args.dataset)
    if args.load_model:
        model.load_state_dict(torch.load(model_path))

    if args.optimizer == "sgd":
        optimizer = SGD(model, train_loader, lr=args.lr)
    elif args.optimizer == "d-sgd":
        optimizer = DefaultWrapper(model, train_loader, optim.SGD, lr=args.lr)
    else:
        raise NotImplementedError

    with tqdm(range(args.epochs)) as bar:
        loss, acc = model.test_step(
            test_loader, 0, device=device, log_interval=args.log_interval
        )
        bar.set_postfix({"loss": loss, "acc": acc})

        for epoch in bar:
            model.train_step(
                optimizer, epoch + 1, device=device, log_interval=args.log_interval
            )

            loss, acc = model.test_step(
                test_loader, epoch + 1, device=device, log_interval=args.log_interval
            )
            bar.set_postfix({"loss": loss, "acc": acc})

    if args.save_model:
        torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    args = parser.parse_args()

    logging.basicConfig(
        filemode="w",
        filename="app.log",
        level=logging.INFO,
        datefmt="%d-%b-%y %H:%M:%S",
        format="%(asctime)s [%(levelname)s] : %(message)s",
    )

    main(args)
