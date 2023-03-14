## Exercise Neural Networks

The goal of this exercise is to implement a multilayer dense neural network using `jax` and `flax`.
Type,

```bash
$ pip install -r requirements.txt
```

into the terminal to install the required software.

Jax takes care of our autograd needs. The documentation is available at https://jax.readthedocs.io/en/latest/index.html . Flax is a high-level neural network library. https://flax.readthedocs.io/en/latest/ hosts the documentation.

### Task 1: Denoising a cosine

As a first step, implement gradient descent learning of a dense neural network using `jax`. 

- Recall the definition of the sigmoid function $\sigma$

$$ \sigma(x) = \frac{1}{1 + e^{-x}} $$


- Implement the `sigmoid` function in `src/denoise_cosine.py`.


- Implement a dense dense layer in the `net` function of `src/denoise_cosine.py` the function should return

$$ \mathbf{W}_2 \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}). $$

- Use numpys `@` notation for the matrix product. [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) is an elegant way to deal with data batches.

- Initialize W2 of shape [200, hidden_neurons], W of shape [hidden_neurons, 200] and, b of shape [hidden_neurons]. Use `jax.random.uniform` to initialize your weights. Start with $\mathcal{U}[-0.1, 0.1]$ for example. `jax.random.PRNGKey` allows you to create a seed for the random number generator.

- Implement a squared error cost

$$  C_{\text{se}} = \frac{1}{2} \sum_{k=1}^{n} (\mathbf{y}_k - \mathbf{h}_k)^2 $$

- `**` denotes squares in python `jnp.sum` allows you to sum up all terms.

- Define the forward pass in `src/net_cost`. The forward pass evaluates the network and the cost function.

- Train your network to denoise a sine. `jax.value_and_grad`, allows you to compute cost and gradient at the same time. Remember the gradient descent update rule

$$ \mathbf{W}_{\tau + 1} = \mathbf{W}_\tau - \epsilon \cdot \delta\mathbf{W}_{\tau} . $$ 

- In the equation above $\mathbf{W} \in \mathbb{R}$ holds for weight matrices and biases. $\epsilon$ denotes the step size and $\delta$ the gradient operation with respect to the following weight.  Use a loop to repeat weight updates for multiple operations. Try to train for one hundret updates.



### Task 2: MNIST
Using flax, set up a fully connected neural network to identify MNIST digits.
Implement your network in `src/mnist.py`.
- Use the [linen api](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html).
- Load the mnist train images using the `get_mnist_train_data` function.
- Implement `normalize` to ensure approximately standard-normal inputs. Normalization requires subtraction of the mean and division by the standard deviation

$$ {x}_{ij} = \frac{x_{ij} - \mu}{\sigma} $$

- for $i = 1, \dots w$ and $j = 1, \dots h$ with w the image width and h the image height.

- Implement the `forward_step` and `sgd_step` functions. `forward_step` should return a scalar cost term you can use to compute gradients. Use the gradients to update the weights in `sgd_step`.

- The forward step requires the `Net` object. Implement the forward pass in it's `__call__` method. Use a combination of `flax.linen.Dense` and `flax.linen.activation.relu`.

- The forward pass ends with the evaluation of a cost function.
Write a `cross_entropy` cost function,

$$       C_{\text{ce}}(\mathbf{y}, \mathbf{o}) = - \frac{1}{n_b} \sum_{i=1}^{n_b} \sum_{k=1}^{n_o} [(\mathbf{y}_{i,k}  \ln \mathbf{o}_{i,k}) + (\mathbf{1} - \mathbf{y}_{i,k}) \ln(\mathbf{1} - \mathbf{o}_{i,k})]. $$

- With $n_o$ the number of labels and $n_b$ in the batched case.


- Implement a function to compute the accuracy. `jnp.argmax` will help.

- Implement the `sgd_step`. Consider `jax.tree_util.tree_map` for this task. Tree maps work best with a lambda expression.

- Train your network for 10 passes over the entire data-set or epochs.

- Load the test data via `get_mnist_test_data`. Evaluate the test accuracy.
