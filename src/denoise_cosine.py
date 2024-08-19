"""An example focused on training a network to denoise a time series."""


import matplotlib.pyplot as plt
import torch as th


def sigmoid(x: th.Tensor) -> th.Tensor:
    """Define logistic sigmoid following 1 / (1 + e^(-x)).

    Args:
        x (th.Tensor): Input Tensor.

    Returns:
        th.Tensor: Sigmoid activated input.
    """
    # TODO: Replace 0. with the correct expression.
    return 0.


class Net(th.nn.Module):
    """Decosine Network."""

    def __init__(
        self, input_neurons: int, output_neurons: int, hidden_neurons: int
    ) -> None:
        """Initialize the network.

        Args:
            input_neurons (int): Number of input neurons.
            output_neurons (int): Number of output neurons.
            hidden_neurons (int): Number of hidden neurons.
        """
        super().__init__()
        # TODO: Create two layers using th.nn.Linear. 


    def forward(self, x: th.Tensor) -> th.Tensor:
        """Network forward pass.

        Args:
            x (th.Tensor): Input tensor of shape 1x200.

        Returns:
            th.Tensor: Network prediction of shape 1x200.
        """
        # TODO: Implment the forward pass using our sigmoid function
        # as well as the layers you created in the __init__ function.
        # return the network output instead of 0.
        return 0.


def cost(y: th.Tensor, h: th.Tensor) -> th.Tensor:
    """Compute Squared Error loss.

    Args:
        y (th.Tensor): Ground truth output.
        h (th.Tensor): Network predicted output.

    Returns:
        th.Tensor: Squared Error.
    """
    # TODO: Return squared error cost instead of 0.
    return 0.


def sgd(model: Net, step_size: float) -> Net:
    """Perform Stochastic Gradient Descent.

    Args:
        model (Net): Network object.
        step_size (float): Step size for SGD.

    Returns:
        Net: SGD applied model.
    """
    for param in model.parameters():
        # TODO: compute an update for every parameter using param.data,
        # step size as well as param.grad.data
        pass
    return model


def zero_grad(model: Net) -> Net:
    """Make gradients zero after SGD.

    Args:
        model (Net): Network object.

    Returns:
        Net: Network with zeroed gradients.
    """
    for param in model.parameters():
        # TODO: call zero_() for every parameter.
        pass
    return model


if __name__ == "__main__":
    # TODO: Use th.manual_seed to set the seed for the network initialization.
    pass
    # TODO: Choose a step size.
    step_size = 0.00
    # TODO: Chose a suitable amount of iterations.
    iterations = 100
    input_neurons = output_neurons = 200
    # TODO: Choose a network size.
    hidden_neurons = 0

    x = th.linspace(-3 * th.pi, 3 * th.pi, 200)
    y = th.cos(x)

    # TODO: Instatiate our network using the `Net`-constructor.
    model = None

    for i in range(iterations):
        th.manual_seed(i)
        y_noise = y + th.randn([200])

        # TODO: Compute the network output using your model.
        preds = 0.

        # TODO: Compute the loss value using your cost function.
        loss_val = 0.

        #TODO: Compute the gradient by calling the backward() function of your loss Tensor.
        pass

        # TODO: Use your sgd function to update your model.
        model = None
        # TODO: Use your zero grad function to reset your gradients
        model = None
        print(f"Iteration: {i}, Cost: {loss_val.item()}")

    y_hat = model(y_noise).detach().numpy()

    plt.title("Denoising a cosine")
    plt.plot(x, y, label="solution")
    plt.plot(x, y_hat, "x", label="fit")
    plt.plot(x, y_noise, label="input")
    plt.legend()
    plt.grid()
    plt.savefig("./figures/Denoise.png", dpi=600, bbox_inches="tight")
    plt.show()
    print("Done")







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
