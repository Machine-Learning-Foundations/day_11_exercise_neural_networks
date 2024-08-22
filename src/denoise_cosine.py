"""An example focused on training a network to denoise a time series."""

from typing import Dict

import matplotlib.pyplot as plt
import torch as th
from torch.func import grad_and_value
from tqdm import tqdm


def sigmoid(x: th.Tensor) -> th.Tensor:
    """Define logistic sigmoid following 1 / (1 + e^(-x)).

    Args:
        x (th.Tensor): Input Tensor.

    Returns:
        th.Tensor: Sigmoid activated input.
    """
    # TODO: Replace 0. with the correct expression.
    return 0.


def net(params: Dict, x: th.Tensor) -> th.Tensor:
    """Set up a single layer network.

    Args:
        params (Dict): Dictionary containing W1, b, and W2.
        x (th.Tensor): Network input.

    Returns:
        th.Tensor: Network prediction.
    """
    # TODO: Implement single layer pass.
    return None


def cost(y: th.Tensor, h: th.Tensor) -> th.Tensor:
    """Compute Squared Error loss.

    Args:
        y (th.Tensor): Ground truth output.
        h (th.Tensor): Network predicted output.

    Returns:
        th.Tensor: Squared Error.
    """
    # TODO: Implement Squared Error loss.
    return 0.


def net_cost(params: Dict, x: th.Tensor, y: th.Tensor) -> th.Tensor:
    """Evaluate the network and compute the loss.

    Args:
        params (Dict): Dictionary containing W1, b, and W2.
        x (th.Tensor): Network input.
        y (th.Tensor): Squared error loss.

    Returns:
        th.Tensor: Squared Error.
    """
    # TODO: Call network, compute and return the loss.
    return None


if __name__ == "__main__":
    # TODO: Use th.manual_seed as 42 to set the seed for the network initialization
    pass
    # TODO: Choose a suitable stepsize
    step_size = 0.0
    iterations = 150
    input_neurons = output_neurons = 200
    # TODO: Choose a proper network size.
    hidden_neurons = 0

    x = th.linspace(-3 * th.pi, 3 * th.pi, 200)
    y = th.cos(x)

    # TODO: Initialize the parameters
    W1 = None
    b = None
    W2 = None

    # TODO: Instantiate grad_and_value function
    value_grad = None

    for i in (pbar := tqdm(range(iterations))):
        th.manual_seed(i)
        y_noise = y + th.randn([200])

        # TODO: Compute loss and gradients

        # TODO: Update parameters using SGD

    # TODO: Compute test y_hat using y_noise and converged parameters
    y_hat = None

    plt.title("Denoising a cosine")
    plt.plot(x, y, label="solution")
    plt.plot(x, y_hat, "x", label="fit")
    plt.plot(x, y_noise, label="input")
    plt.legend()
    plt.grid()
    plt.savefig("./figures/Denoise.png", dpi=600, bbox_inches="tight")
    plt.show()
    print("Done")
