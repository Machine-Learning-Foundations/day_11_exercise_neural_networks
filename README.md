## Exercise Neural Networks

The goal of this exercise is to implement a multilayer dense neural network using `jax` and `flax`.
Type,

```bash
pip install -r requirements.txt
```

into the terminal to install the required software.

Jax takes care of our autograd needs. The documentation is available at https://jax.readthedocs.io/en/latest/index.html . Flax is a high-level neural network library. https://flax.readthedocs.io/en/latest/ hosts the documentation.

### Task 1: Denoising a cosine

To get a notion of how function learning of a dense layer network works on given data, we will first have a look at the example from the lecture. In the following task you will implement gradient descent learning of a dense neural network using `jax` and use it to learn a function, e.g. a cosine.

- As a first step create a cosine function in jax and add some noise with `jax.random.normal`. Use for example a signal length of $n = 200$ samples and a period of your choosing. This will be the noisy signal that the model is supposed to learn the underlaying cosine from.

- Recall the definition of the sigmoid function $\sigma$

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$


- Implement the `sigmoid` function in `src/denoise_cosine.py`.


- Implement a dense layer in the `net` function of `src/denoise_cosine.py` the function should return
   $$\hat{y} = \mathbf{W}_2 \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b})\qquad\qquad\qquad\qquad(1)$$  
   where $\mathbf{W}_1\in \mathbb{R}^{m,n}, \mathbf{x}\in\mathbb{R}^n, \mathbf{b}\in\mathbb{R}^m$ and $m$ denotes the number of neurons and $n$ the input signal length. Suppose that the input parameters are stored in a [python dictonary](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) with the keys `W_1`, `W_2` and `b`.  
   Use numpys `@` notation for the matrix product. [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) is an elegant way to deal with data batches.

- Use `jax.random.uniform` to initialize your weigths. For a signal length of $200$ the $W_2$ matrix should have e.g. have the shape [200, `hidden_neurons`] and $W_1$ a shape of [`hidden_neurons`, 200]. Start with $\mathcal{U}[-0.1, 0.1]$ for example. `jax.random.PRNGKey` allows you to create a seed for the random number generator.

- Implement and test a squared error cost

$$C_{\text{se}} = \frac{1}{2} \sum_{k=1}^{n} (\mathbf{y}_k - \mathbf{h}_k)^2$$

- `**` denotes squares in python `jnp.sum` allows you to sum up all terms.

- Define the forward pass in `src/net_cost`. The forward pass evaluates the network and the cost function.

- Train your network to denoise a cosine. To do so, implement gradient descent on the noisy input signal and use e.g. `jax.value_and_grad` to compute cost and gradient at the same time. Remember the gradient descent update rule  

$$\mathbf{W}_{\tau + 1} = \mathbf{W}_\tau - \epsilon \cdot \delta\mathbf{W}_{\tau}.$$  


- In the equation above $\mathbf{W} \in \mathbb{R}$ holds for weight matrices and biases. $\epsilon$ denotes the step size and $\delta$ the gradient operation with respect to the following weight.  Use a loop to repeat weight updates for multiple operations. Try to train for one hundred updates.

- At last compute the network output `y_hat` on the final values to see if the network learned the underlying cosine function. Use `matplotlib.pyplot.plot` to plot the noisy signal and the network output $\hat{y}$.




### Task 2: MNIST
In this task we will go one step further. Instead of a cosine function our neural network will learn how to identify handwritten digits from the [MNSIT dataset](http://yann.lecun.com/exdb/mnist/). For that we will be using the [linen api](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html) of the module [flax](https://flax.readthedocs.io/en/latest/). Firstly, make yourself familiar with the linen api to get started with training a fully connected network in `src/mnist.py`. In this script some functions are already implemented and can easily be reused.

- Implement the `normalize` function to ensure approximate standard-normal inputs. Make use of handy numpy methods that you already know. Normalization requires subtraction of the mean and division by the standard deviation with $i = 1, \dots w$ and $j = 1, \dots h$ with $w$ the image width and $h$ the image height and $k$ running through the batch dimension:
$$ \tilde{{x}}_{ijk} = \frac{x_{ijk} - \mu}{\sigma} $$

- The forward step requires the `Net` object from its [class](https://docs.python.org/3/tutorial/classes.html). It is your fully connected neural network model. Applying weights to a `flax.linen.Module` is comparable to calculating the forward pass of the network in task 1. Implement a dense network in `Net` of your choosing using using a combination of `flax.linen.Dense` and `flax.linen.activation.relu`.

- The forward pass ends with the evaluation of a cost function.
Write a `cross_entropy` cost function,
   $$C_{\text{ce}}(\mathbf{y}, \mathbf{o}) = - \frac{1}{n_b} \sum_{i=1}^{n_b} \sum_{k=1}^{n_o} [(\mathbf{y}_{i,k}  \ln \mathbf{o}_{i,k}) + (\mathbf{1} - \mathbf{y}_{i,k}) \ln(\mathbf{1} - \mathbf{o}_{i,k})].$$
   with $n_o$ the number of labels and $n_b$ in the batched case. Test your function.

- Now implement the `forward_step` function. Calculate the network output first. Then compute the loss. It should return a scalar cost term you can use to compute gradients. Make use of the cross entropy.

- Next we want to be able to do a optimization step with stochastic gradient descent (sgd). Implement `sgd_step`. Use the gradients to update the weights. Consider `jax.tree_util.tree_map` for this task. Tree maps work best with a lambda expression.

- To evaluate the network we calculate the accuracy of the network output. Implement `get_acc` to calculate the accuracy given a batch of images and the corresponding labels for these images.

- Now is the time to move back to the main procedure. First fetch the train data via the function `get_mnist_train_data`. To be able to evaluate the network while it is being trained, we use a validation set. Split the train set into two disjoint sets: the training and the validation set. Normalize both sets. Bear in mind which mean and stds should be used for the validation set.

- Define your loss and gradient function with jax (see task 1). Next initialize the network with the `Net` object (see the `flax` documentation for help).

- Train your network for a fix number of `epochs` over the entire dataset.
    
- Last but not least, load the test data with `get_mnist_test_data` and calculate the test accuracy. Save it to a list.

- Optional: Plot the training and validation accuracies and add the test accuracy in the end.
