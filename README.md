## Exercise Neural Networks

The goal of this exercise is to implement a multilayer dense neural network using `jax` and `flax`.
Type,

```bash
$ pip install -r requirements.txt
```

into the terminal to install the required software.

Jax takes care of our autograd needs. The documentation is available at https://jax.readthedocs.io/en/latest/index.html . Flax is a high-level neural network library. https://flax.readthedocs.io/en/latest/ hosts the documentation.

### Task 1: Denoising a cosine
- As a first step implement gradient descent using `jax`. 
- Train a dense layer to denoise a cosine in `src/denoise_cosine.py`:

$$ \mathbf{W}_2 \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}). $$

With W2 of shape [200, hidden_neurons], W of shape [hidden_neurons, 200] and b of shape [hidden_neurons].
Use `jax.random.uniform` to initialize your weigths.
Use i.e. `jax.value_and_grad` to compute cost and gradient at the same time.


### Task 2: Getting started on Bender
Use the `Remote - SSH` to connect to Bender using your Uni-ID.
To share GPUs the environment variable `XLA_PYTHON_CLIENT_PREALLOCATE=false` must always be set!

Modefiy your launch.json, it should look something like this:
``` json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": true,
            "env": {
               "XLA_PYTHON_CLIENT_PREALLOCATE": "False"
            }
        }
    ]
}
```

### Task 3: MNIST
Using flax set up a fully connected neural network to identify MNIST digits.
Implement your network in `src/mnist.py`.
- Use the [linen api](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html).
- Load the mnist train images using the `get_mnist_train_data` function.
- Implement `normalize` to ensure approximate normal inputs.
- Implement the `forward_step` and `sgd_step` functions. `forward_step` should return a scalar cost term you can use to compute gradients. Use the gradients to update the weights in `sgd_step`.
- Implement a function to compute the accuracy.
- Train your network for 10 passes over the entire data-set or epochs.

- Load the test data via `get_mnist_test_data`.
- Find the test accuracy.
