"""An example focused on training a network to denoise a time series."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# def relu(x):
#     x = x.at[x < 0].set(0)
#     return x


def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """Define a logistic sigmoid following 1 / (1 + e^(-x))."""
    # TODO: Implement me.
    return 0.0


def net(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    """Set up a single layer network with output mapping.

    Args:
        params (dict): The network parameters with keys W1, W2 and b.

    Returns:
        jnp.ndarray: The network output.

    """
    # TODO: Implement me.
    return jnp.zeros_like(x)


def cost(y: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
    """Compute a squared error cost."""
    # TODO: Implement me.
    return jnp.array(0.0)


def net_cost(params: dict, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the network and the cost."""
    # TODO: Implement me.
    return jnp.array(0.0)


if __name__ == "__main__":
    step_size = 0.01
    iterations = 100
    hidden_neurons = 10

    # generate cosine signal
    x = jnp.linspace(-3 * jnp.pi, 3 * jnp.pi, 200)
    y = jnp.cos(x)

    # TODO: Create W1, W2 and b using different random keys

    for i in range(iterations):
        # add noise to cosine
        y_noise = y + jax.random.normal(jax.random.PRNGKey(i), [200])
 
        # TODO: Implement a dense neural network to denoise the cosine.
