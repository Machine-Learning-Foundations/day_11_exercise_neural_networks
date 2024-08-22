"""Identify MNIST digits."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torchvision.datasets as tvd
import torchvision.transforms as tvt
from tqdm import tqdm


class Net(th.nn.Module):
    """MNIST Network."""

    def __init__(self) -> None:
        """Network initialization."""
        super().__init__()
        # TODO: Initialize the network.

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Network forward pass.

        Args:
            x (th.Tensor): Input Tensor of shape (BS, 1, 28, 28).

        Returns:
            th.Tensor: Network predictions of shape (BS, 10).
        """
        # TODO: Implement forward pass.
        return None


def cross_entropy(label: th.Tensor, out: th.Tensor) -> th.Tensor:
    """Cross Entropy loss.

    Args:
        label (th.Tensor): Ground truth labels.
        out (th.Tensor): Network predictions.

    Returns:
        th.Tensor: Cross-Entropy loss.
    """
    # TODO: Implement Cross-Entropy loss.
    return 0.


def sgd_step(model: Net, learning_rate: float) -> Net:
    """Perform SGD.

    Args:
        model (Net): Network objekt.
        learning_rate (float): Learning rate or step size.

    Returns:
        Net: SGD applied model.
    """
    # TODO: Implement SGD using model.parameters
    # Hint: For gradient one can use model.param.grad
    return model


def get_acc(model: Net, dataloader: th.utils.data.DataLoader) -> float:
    """Compute accuracy given specific dataloader.

    Args:
        model (Net): Network objekt.
        dataloader (th.utils.data.DataLoader): Dataloader objekt.

    Returns:
        float: Accuracy.
    """
    # TODO: Given model and dataloader compute accuracy.
    return 0.


def zero_grad(model: Net) -> Net:
    """Make gradients zero after SGD.

    Args:
        model (Net): Network object.

    Returns:
        Net: Network with zeroed gradients.
    """
    for param in model.parameters():
        param.grad.data.zero_()
    return model


def normalize_batch(imgs: th.Tensor) -> th.Tensor:
    """Normalize a specific batch of images.

    Args:
        imgs (th.Tensor): Batch of images.

    Returns:
        th.Tensor: Normalized images.
    """
    # TODO: Given images tensor, normalize the images.
    return None


# HYPERPARAMETERS
BS = 200
EPOCHS = 10
DEVICE = th.device("cuda") if th.cuda.is_available() else th.device("cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train network on MNIST.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning Rate")
    args = parser.parse_args()
    print(args)

    train_transforms = tvt.Compose(
        [
            tvt.ToTensor(),
        ]
    )
    test_transforms = tvt.Compose(
        [
            tvt.ToTensor(),
        ]
    )
    dataset = tvd.MNIST(
        "./.data", train=True, download=True, transform=train_transforms
    )
    trainset, valset = th.utils.data.random_split(dataset, [50000, 10000])

    train_loader = th.utils.data.DataLoader(trainset, batch_size=BS, shuffle=True)
    val_loader = th.utils.data.DataLoader(valset, batch_size=BS, shuffle=False)
    test_loader = th.utils.data.DataLoader(
        tvd.MNIST("./.data", train=False, download=True, transform=test_transforms),
        batch_size=10000,
        shuffle=False,
    )
    
    # TODO: Setup a dense layer network, train and test the network. 
