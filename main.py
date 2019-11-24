from __future__ import print_function, division

import torch
import logging
import argparse

from tqdm import tqdm
from torch import optim
from torch.nn import functional as F

from includes.models import NN, CNN, AdaNet
from includes.optimizers import SGD, DefaultWrapper
from includes.utils import load_mnist, RademacherComplexity


parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
parser.add_argument(
    "--dataset",
    default="mnist",
    choices=["mnist"],
    help="choose dataset (default: %(default)s)",
)
parser.add_argument(
    "--model",
    default="adanet",
    choices=["nn", "cnn", "adanet"],
    help="choose model (default: %(default)s)",
)
parser.add_argument(
    "--width", default=5, type=int, help="choose model (default: %(default)s)"
)
parser.add_argument(
    "--n-iters",
    default=5,
    type=int,
    help="number of iterations for subnetwork training (default: %(default)s)",
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
    "--decay_rate",
    type=float,
    default=0.9,
    help="Learning rate decay rate (default: %(default)s)",
)
parser.add_argument(
    "--decay_steps",
    type=int,
    default=0,
    help="Learning rate decay steps (default: %(default)s = no decay)",
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
    log = args.log_interval > 0

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    if args.dataset == "mnist":
        train_loader, test_loader, input_max = load_mnist(args.batch_size, **kwargs)

        input_dim = 784
        output_dim = 10
    else:
        raise NotImplementedError

    if args.model == "cnn":
        model = CNN("cnn", loss_fn=F.nll_loss)
    elif args.model == "nn":
        model = NN("nn", loss_fn=F.nll_loss, input_dim=input_dim, output_dim=output_dim)
    elif args.model == "adanet":
        model = AdaNet(
            "adanet",
            loss_fn=F.nll_loss,
            activation_fn=F.relu,
            input_dim=input_dim,
            output_dim=output_dim,
            width=args.width,
            n_iters=args.n_iters,
            regularizer=RademacherComplexity(),
            input_max=input_max,
            batch_size=args.batch_size,
        )
    else:
        raise NotImplementedError

    model = model.to(device)

    model_path = "sandbox/models/{}_{}.pt".format(model.name, args.dataset)
    if args.load_model:
        model.load_state_dict(torch.load(model_path))

    if args.optimizer == "sgd":
        optimizer = SGD(
            model, lr=args.lr, decay_rate=args.decay_rate, decay_steps=args.decay_steps
        )
    elif args.optimizer == "d-sgd":
        optimizer = DefaultWrapper(model, optim.SGD, lr=args.lr)
    else:
        raise NotImplementedError

    with tqdm(range(args.epochs)) as bar:
        _, acc = model.test_step(test_loader, 0, device=device, log=log)
        bar.set_postfix({"acc": acc})

        prev_loss = float("inf")
        for epoch in bar:
            loss = model.train_step(
                optimizer, train_loader, epoch + 1, device=device, log=log
            )

            if log:
                if epoch % args.log_interval == 0:
                    logging.info(
                        "Train Epoch: {:3d} Loss: {:.6f}".format(epoch + 1, loss)
                    )

            _, acc = model.test_step(test_loader, epoch + 1, device=device)
            bar.set_postfix({"loss": loss, "acc": acc})

            if log:
                if epoch % args.log_interval == 0:
                    logging.info(
                        "Test  Epoch: {:3d}, Loss: {:.6f}, Accuracy: {:.4f}".format(
                            epoch + 1, loss, acc
                        )
                    )

            if prev_loss - loss < 0.01:
                if log:
                    logging.info("Train Epoch: {:3d} STOPPED".format(epoch + 1))

                break

            prev_loss = loss

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
