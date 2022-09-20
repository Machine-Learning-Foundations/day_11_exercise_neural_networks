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
    # if gpu:
    #    return cp.array(img_data_test), cp.array(lbl_data_test)
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
        mean = np.mean(data)
    if std is None:
        std = np.std(data)
    return (data - mean) / std, mean, std


class Net(nn.Module):
    """A simple net model."""

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Run the forward pass."""
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=784)(x)
        x = nn.relu(x)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return nn.sigmoid(x)


@jax.jit
def cross_entropy(label: jnp.ndarray, out: jnp.ndarray) -> jnp.ndarray:
    """Compute the cross entropy of one-hot encoded labels and the network output.

    jnp.log(0) equals -infinity!
    Dont forget to numerically stabilize the logs using small epsilons.

    Args:
        label (jnp.ndarray): The image labels of shape [batch_size, class_no].
        out (jnp.ndarray): The network output of shape [batch_size, class_no].

    Returns:
        (jnp.ndarray): The loss scalar.
    """
    left = -label * jnp.log(out + 1e-8)
    right = -(1 - label) * jnp.log(1 - out + 1e-8)
    return jnp.mean(left + right)


@jax.jit
def forward_step(
    variables: FrozenDict, img_batch: jnp.ndarray, label_batch: jnp.ndarray
) -> jnp.ndarray:
    """Do a forward step using your network and compute the loss.

    Args:
        variables (FrozenDict): A dictionary containing the network weights.
        img_batch (jnp.ndarray): An image batch of shape [batch_size, height, widht].
        label_batch (jnp.ndarray): A label batch of shape [batch_size].

    Returns:
        jnp.ndarray: A scalar containing the loss value.
    """
    out = net.apply(variables, jnp.expand_dims(img_batch, -1))
    ce_loss = cross_entropy(nn.one_hot(label_batch, num_classes=10), out)
    return ce_loss


# set up autograd
loss_grad_fn = jax.value_and_grad(forward_step)


# set up SGD
# @jax.jit
def sgd_step(
    variables: FrozenDict, grads: FrozenDict, learning_rate: float
) -> FrozenDict:
    """Update the variable in a SGD step.

    The idea is to compute w_{t+1} = w_t - learning_rate * g using
    jax.tree_util.tree_ap.

    Args:
        variables (FrozenDict): A dictionary containing the network weights.
        grads (FrozenDict): A dictionary containing the gradients.

    Returns:
        FrozenDict: The updated network weights.
    """
    variables = jax.tree_util.tree_map(
        lambda p, g: p - learning_rate * g, variables, grads
    )
    return variables


def get_acc(img_data: jnp.ndarray, label_data: jnp.ndarray) -> float:
    """Compute the network accuracy.

    Args:
        img_data (jnp.ndarray): An image batch of shape [stack_size, height, widht].
        label_data (jnp.ndarray): The corresponding labels of shape [stack_size].

    Returns:
        float: The accuracy in percent [%].
    """
    out = net.apply(variables, jnp.expand_dims(img_data, -1))
    rec = jnp.argmax(out, axis=1)
    acc = jnp.sum((rec == label_data).astype(np.float32)) / len(label_data)
    return float(acc)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Networks on MNIST.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning Rate")
    args = parser.parse_args()
    print(args)

    batch_size = 50
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

    exp_list = []

    for key in (0, 1, 2, 3, 4):
        acc_list_train = []
        acc_list_val = []

        key = jax.random.PRNGKey(key)  # type: ignore
        net = Net()
        variables = net.init(
            key, jnp.ones([batch_size] + list(img_data_train.shape[1:]) + [1])
        )

        for e in range(epochs):
            shuffler = jax.random.permutation(key, len(img_data_train))
            img_data_train = img_data_train[shuffler]
            lbl_data_train = lbl_data_train[shuffler]

            img_batches = np.split(
                img_data_train, img_data_train.shape[0] // batch_size, axis=0
            )
            label_batches = np.split(
                lbl_data_train, lbl_data_train.shape[0] // batch_size, axis=0
            )

            for img_batch, label_batch in tqdm(
                zip(img_batches, label_batches), total=len(img_batches)
            ):
                img_batch, label_batch = (
                    jnp.array(np_array) for np_array in (img_batch, label_batch)
                )
                # cel = cross_entropy(nn.one_hot(label_batches[no], num_classes=10),
                #                    out)
                cel, grads = loss_grad_fn(variables, img_batch, label_batch)
                variables = sgd_step(variables, grads, args.lr)
            print("Epoch: {}, loss: {}".format(e, cel))

            train_acc = get_acc(jnp.array(img_data_train), jnp.array(lbl_data_train))
            val_acc = get_acc(jnp.array(img_data_val), jnp.array(lbl_data_val))
            print(
                "Train and Validation accuracy: {:3.3f}, {:3.3f}".format(
                    train_acc, val_acc
                )
            )
            acc_list_train.append(train_acc)
            acc_list_val.append(val_acc)

        print("Training done. Testing...")
        img_data_test, lbl_data_test = get_mnist_test_data()
        img_data_test, mean, std = normalize(img_data_test, mean, std)

        test_acc = get_acc(jnp.array(img_data_test), jnp.array(lbl_data_test))
        print("Done. Test acc: {}".format(test_acc))
        exp_list.append((test_acc, acc_list_train, acc_list_val))

    test_accs = jnp.array([exp[0] for exp in exp_list])
    train_accs = jnp.stack([jnp.array(exp[1]) for exp in exp_list])
    val_accs = jnp.stack([jnp.array(exp[2]) for exp in exp_list])

    test_mean = jnp.mean(test_accs)
    test_std = jnp.std(test_accs)

    train_mean = jnp.mean(train_accs, axis=0)
    train_std = jnp.std(train_accs, axis=0)

    val_mean = jnp.mean(val_accs, axis=0)
    val_std = jnp.std(val_accs, axis=0)

    def plot_mean_std(steps, mean, std, color, label="", marker="."):
        """Plot means and standard deviations with shaded areas."""
        plt.plot(steps, mean, label=label, color=color, marker=marker)
        plt.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    plot_mean_std(
        jnp.arange(1, 11), train_mean, train_std, colors[0], label="train acc"
    )
    plot_mean_std(jnp.arange(1, 11), val_mean, val_std, colors[1], label="val acc")
    plt.errorbar(
        jnp.array([10]),
        test_mean,
        test_std,
        color=colors[2],
        label="test acc",
        marker="x",
    )
    plt.legend()
    plt.show()
    print("done")
