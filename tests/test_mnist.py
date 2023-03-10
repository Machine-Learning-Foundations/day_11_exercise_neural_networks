"""Test the python functions from src/mnist."""

import sys

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, "./src/")

from src.mnist import cross_entropy, normalize

testdata1 = [
    (
        np.arange(0, 10),
        4.5,
        2.8722813232690143,
        np.array(
            [
                -1.5666989,
                -1.2185436,
                -0.8703883,
                -0.522233,
                -0.1740777,
                0.1740777,
                0.522233,
                0.8703883,
                1.2185436,
                1.5666989,
            ]
        ),
    ),
    (
        np.linspace(5, 15, 6),
        10.0,
        3.415650255319866,
        np.array(
            [
                -1.46385011,
                -0.87831007,
                -0.29277002,
                0.29277002,
                0.87831007,
                1.46385011,
            ]
        ),
    ),
]


@pytest.mark.parametrize("data, mean, std, res", testdata1)
def test_normalize(data, mean, std, res) -> None:
    """Test it the data is normalized correctly."""
    result = normalize(data=data)
    norm_data = np.round(result[0], 7)
    assert np.allclose(norm_data, res)
    assert np.allclose(result[1], mean)
    assert np.allclose(result[2], std)


testdata2 = [
    (
        jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
        jnp.array([[0.2, 0.12], [0.42, 0.21], [0.22, 0.34]]),
        jnp.array(0.7511, dtype=jnp.float32),
    ),
    (
        jnp.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0]]),
        jnp.array([[0.8, 0.11], [0.22, 0.22], [0.1, 0.3], [0.08, 0.19]]),
        jnp.array(0.3803, dtype=jnp.float32),
    ),
]


@pytest.mark.parametrize("label, out, res", testdata2)
def test_cross_entropy(label, out, res) -> None:
    """Test if the cross entropy is implemented correctly."""
    result = cross_entropy(label=label, out=out)
    ce = jnp.round(result, 4)
    assert jnp.allclose(ce, res)
