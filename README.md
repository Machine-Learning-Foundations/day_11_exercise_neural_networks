## Exercise Neural Networks

The goal of this exercise is to implement a multilayer dense neural network using `jax` and `flax`.
Type,

```bash
$ pip install -r requirements.txt
```

into the terminal to install the required software.

Jax takes care of our autograd needs. The documentation is available at https://jax.readthedocs.io/en/latest/index.html . Flax is a high-level neural network library. https://flax.readthedocs.io/en/latest/ hosts the documentation.

### Task 1: Denoising a cosine

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

$$       C_{\text{ce}}(\mathbf{y}, \mathbf{o}) = -\sum_{k=1}^{n_o} [(\mathbf{y}_k  \ln \mathbf{o}_k) + (\mathbf{1} - \mathbf{y}_k) \ln(\mathbf{1} - \mathbf{o}_k)]. $$

- With $n_o$ the number of labels or $n_o \cdot n_b$ in the batched case.


- Implement a function to compute the accuracy. `jnp.argmax` will help.

- Implement the `sgd_step`. Consider `jax.tree_util.tree_map` for this task. Tree maps work best with a lambda expression.

- Train your network for 10 passes over the entire data-set or epochs.

- Load the test data via `get_mnist_test_data`. Evaluate the test accuracy.
