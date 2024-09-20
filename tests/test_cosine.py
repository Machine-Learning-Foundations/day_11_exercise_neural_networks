"""Test the python functions from src/denoise_cosine."""

import sys

import pytest
import torch as th

sys.path.insert(0, "./src/")

from src.denoise_cosine import cost, net, sigmoid


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 42])
def test_sigmoid(seed) -> None:
    """See if the sigmoid is correct and really returns true."""
    test_sig = lambda x: th.exp(x) / (1.0 + th.exp(x))  # noqa: E731
    th.manual_seed(seed)
    x = th.rand([1])
    assert th.allclose(sigmoid(x), test_sig(x))


testdata = [
    (th.tensor([1, 2, 3]), th.tensor([4, 5, 6]), th.tensor([13.5])),
    (th.tensor([42, 21, 11]), th.tensor([4, 5, 3]), th.tensor([882.0])),
]


@pytest.mark.parametrize("y, h, res", testdata)
def test_mse(y, h, res) -> None:
    """See if the cost is correct for two samples and returns an tensor."""
    assert th.allclose(cost(y, h), res)


test_net_data = [
    (
        {
            "W1": th.tensor([[3, 2], [1, 2]]).type(th.float64),
            "W2": th.tensor([[3, 4], [5, 6]]).type(th.float64),
            "b": th.tensor([42]).type(th.float64),
        },
        th.tensor([13.0, 12.0]).type(th.float64),
        th.tensor([7.0, 11.0]).type(th.float64),
    ),
    (
        {
            "W1": th.tensor([[3, 42], [1, 2]]).type(th.float64),
            "W2": th.tensor([[3, 4], [23, 6]]).type(th.float64),
            "b": th.tensor([21]).type(th.float64),
        },
        th.tensor([13.0, 12.0]).type(th.float64),
        th.tensor([7.0, 29.0]).type(th.float64),
    ),
]


@pytest.mark.parametrize("params, x, res", test_net_data)
def test_net(params, x, res) -> None:
    """See it the net is correct for two samples and returns an array."""
    output = net(params, x)
    assert th.allclose(output, res)
