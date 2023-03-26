"""Identify mnist digits."""
import argparse
import struct
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from tqdm import tqdm


def get_mnist_test_data() -> Tuple[np.ndarray, np.ndarray]:
    """Return the mnist test data set in numpy arrays.

    Returns:
        (array, array): A touple containing the test
        images and labels.
    """
    with open("./data/MNIST/t10k-images-idx3-ubyte", "rb") as f:
        _, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.array(np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder(">")))
        img_data_test = data.reshape((size, nrows, ncols))

    with open("./data/MNIST/t10k-labels-idx1-ubyte", "rb") as f:
        _, size = struct.unpack(">II", f.read(8))
        lbl_data_test = np.array(np.fromfile(f, dtype=np.dtype(np.uint8)))
    return img_data_test, lbl_data_test


def get_mnist_train_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load the mnist training data set.

    Returns:
        (array, array): A touple containing the training
        images and labels.
    """
    with open("./data/MNIST/train-images-idx3-ubyte", "rb") as f:
        _, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.array(np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder(">")))
        img_data_train = data.reshape((size, nrows, ncols))

    with open("./data/MNIST/train-labels-idx1-ubyte", "rb") as f:
        _, size = struct.unpack(">II", f.read(8))
        lbl_data_train = np.array(np.fromfile(f, dtype=np.dtype(np.uint8)))
    return img_data_train, lbl_data_train


def normalize(
    data: np.ndarray, mean: Optional[float] = None, std: Optional[float] = None
) -> Tuple[np.ndarray, float, float]:
    """Normalize the input array.

    After normalization the input
    distribution should be approximately standard normal.

    Args:
        data (np.array): The input array.
        mean (float): Data mean, re-computed if None. Defaults to None.
        std (float): Data standard deviation, re-computed if None. Defaults to None.

    Returns:
        (np.array, float, float): Normalized data, mean and std.
    """
    if mean is None:
        pass
        # TODO: Implement me.
    if std is None:
        pass
        # TODO: Implement me.
    return data, 0.0, 0.0


class Net(nn.Module):
    """A simple net model."""

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Run the forward pass."""
        # TODO: Implement me.
        return x


# @jax.jit
def cross_entropy(label: jnp.ndarray, out: jnp.ndarray) -> jnp.ndarray:
    """Compute the cross entropy of one-hot encoded labels and the network output.

    jnp.log(0) equals -infinity!
    Dont forget to numerically stabilize the logs using small epsilons (< 1e-5).

    Args:
        label (jnp.ndarray): The image labels of shape [batch_size, class_no].
        out (jnp.ndarray): The network output of shape [batch_size, class_no].

    Returns:
        (jnp.ndarray): The loss scalar.
    """
    # TODO: Implement me.
    return jnp.array(0.0)


# @jax.jit
def forward_step(
    variables: FrozenDict, img_batch: jnp.ndarray, label_batch: jnp.ndarray
) -> jnp.ndarray:
    """Do a forward step using your network and compute the loss.

    Args:
        variables (FrozenDict): A dictionary containing the network weights.
        img_batch (jnp.ndarray): An image batch of shape [batch_size, height, width].
        label_batch (jnp.ndarray): A label batch of shape [batch_size].

    Returns:
        jnp.ndarray: A scalar containing the loss value.
    """
    # TODO: Implement me.
    return jnp.array(0.0)


# set up autograd
loss_grad_fn = jax.value_and_grad(forward_step)


# set up SGD
@jax.jit
def sgd_step(
    variables: FrozenDict, grads: FrozenDict, learning_rate: float
) -> FrozenDict:
    """Update the variable in a SGD step.

    The idea is to compute w_{t+1} = w_t - learning_rate * g using
    jax.tree_util.tree_map.

    Args:
        variables (FrozenDict): A dictionary containing the network weights.
        grads (FrozenDict): A dictionary containing the gradients.

    Returns:
        FrozenDict: The updated network weights.
    """
    # TODO: Implement me.
    return variables


def get_acc(img_data: jnp.ndarray, label_data: jnp.ndarray) -> float:
    """Compute the network accuracy.

    Args:
        img_data (jnp.ndarray): An image batch of shape [stack_size, height, widht].
        label_data (jnp.ndarray): The corresponding labels of shape [stack_size].

    Returns:
        float: The accuracy in percent [%].
    """
    # TODO: Implement me.
    return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Networks on MNIST.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning Rate")
    args = parser.parse_args()
    print(args)

    batch_size = 200
    val_size = 1000
    epochs = 10
    img_data_train, lbl_data_train = get_mnist_train_data()
    img_data_val, lbl_data_val = img_data_train[:val_size], lbl_data_train[:val_size]
    img_data_train, lbl_data_train = (
        img_data_train[val_size:],
        lbl_data_train[val_size:],
    )
    img_data_train, mean, std = normalize(img_data_train)
    img_data_val, _, _ = normalize(img_data_val, mean, std)
    
    # TODO: Set up a dense layer network, train and test the network.
