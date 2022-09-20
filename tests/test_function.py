"""Test the python function from src."""

import sys

import jax
import jax.numpy as jnp
import pytest

sys.path.insert(0, "./src/")

from src.denoise_cosine import sigmoid


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 42])
def test_function(seed) -> None:
    """See it the sigmoid is correct really returns true."""
    test_sig = lambda x: jnp.exp(x) / (1.0 + jnp.exp(x))  # noqa: E731
    key = jax.random.PRNGKey(seed)
    x = jax.random.uniform(key, [1])
    assert jnp.allclose(sigmoid(x), test_sig(x))
