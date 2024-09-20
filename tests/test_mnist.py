"""Test the python functions from src/mnist."""

import sys

import pytest
import torch as th

sys.path.insert(0, "./src/")

from src.mnist import cross_entropy, normalize_batch

testdata = [
    (
        th.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
        th.tensor([[0.2, 0.12], [0.42, 0.21], [0.22, 0.34]]),
        th.tensor(1.5022, dtype=th.float32),
    ),
    (
        th.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0]]),
        th.tensor([[0.8, 0.11], [0.22, 0.22], [0.1, 0.3], [0.08, 0.19]]),
        th.tensor(0.7607, dtype=th.float32),
    ),
]


@pytest.mark.parametrize("label, out, res", testdata)
def test_cross_entropy(label, out, res) -> None:
    """Test if the cross entropy is implemented correctly."""
    result = cross_entropy(label=label, out=out)
    ce = th.round(result, decimals=4)
    assert th.allclose(ce, res)


norm_testdata = [
    (
        th.linspace(0, 5, 10),
        th.tensor(
            [
                -1.4863,
                -1.1560,
                -0.8257,
                -0.4954,
                -0.1651,
                0.1651,
                0.4954,
                0.8257,
                1.1560,
                1.4863,
            ]
        ),
    ),
    (th.linspace(0, 1, 5), th.tensor([-1.2649, -0.6325, 0.0000, 0.6325, 1.2649])),
]


@pytest.mark.parametrize("inpt, res", norm_testdata)
def test_normalize(inpt, res) -> None:
    """Test the normalization."""
    output = normalize_batch(inpt)
    output = th.round(output, decimals=4)
    assert th.allclose(output, res)
