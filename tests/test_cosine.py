"""Test the python functions from src/denoise_cosine."""

import sys

import jax
import jax.numpy as jnp
import pytest

sys.path.insert(0, "./src/")

from src.denoise_cosine import cost, net, sigmoid


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 42])
def test_sigmoid(seed) -> None:
    """See if the sigmoid is correct and really returns true."""
    test_sig = lambda x: jnp.exp(x) / (1.0 + jnp.exp(x))  # noqa: E731
    key = jax.random.PRNGKey(seed)
    x = jax.random.uniform(key, [1])
    assert jnp.allclose(sigmoid(x), test_sig(x))


testdata = [
    (jnp.array([1, 2, 3]), jnp.array([4, 5, 6]), jnp.array([13.5])),
    (jnp.array([42, 21, 11]), jnp.array([4, 5, 3]), jnp.array([882.0])),
]


@pytest.mark.parametrize("y, h, res", testdata)
def test_mse(y, h, res) -> None:
    """See it the cost is correct for two samples and returns an array."""
    assert jnp.allclose(cost(y, h), res)


test_net_data = [
    (
        {
            "W1": jnp.array([[3, 2], [1, 2]]),
            "W2": jnp.array([[3, 4], [5, 6]]),
            "b": jnp.array([42]),
        },
        jnp.array([13, 12]),
        jnp.array([7.0, 11.0]),
    ),
    (
        {
            "W1": jnp.array([[3, 42], [1, 2]]),
            "W2": jnp.array([[3, 4], [23, 6]]),
            "b": jnp.array([21]),
        },
        jnp.array([13, 12]),
        jnp.array([7.0, 29.0]),
    ),
]


@pytest.mark.parametrize("params, x, res", test_net_data)
def test_net(params, x, res) -> None:
    """See it the net is correct for two samples and returns an array."""
    print(net(params, x))
    assert jnp.allclose(net(params, x), res)
