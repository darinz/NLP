# 03. Backpropagation and Optimization

## Introduction

Backpropagation is the core algorithm for training neural networks. It efficiently computes gradients for all parameters, enabling optimization algorithms to update weights and minimize loss. This guide covers the concepts, math, and practical code for backpropagation and optimization.

## The Learning Problem

Given a dataset $`\{(\mathbf{x}^{(i)}, y^{(i)})\}`$, we want to find network parameters $`\theta`$ that minimize a loss function $`L(\theta)`$ (e.g., mean squared error, cross-entropy).

## Forward Pass

- Compute the output of the network for a given input by applying weights, biases, and activation functions layer by layer.

## Loss Function

For example, mean squared error (MSE) for regression:

```math
L = \frac{1}{N} \sum_{i=1}^N (y^{(i)} - \hat{y}^{(i)})^2
```
where $`y^{(i)}`$ is the true label and $`\hat{y}^{(i)}`$ is the prediction.

## Backpropagation: Computing Gradients

Backpropagation uses the chain rule to compute the gradient of the loss with respect to each parameter.

### Chain Rule Example

Suppose $`z = f(y)`$, $`y = g(x)`$, then:

```math
\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}
```

### Backpropagation in a Neural Network

For each layer, compute:
- The gradient of the loss with respect to the output of the layer.
- The gradient with respect to the weights and biases.
- Propagate the gradient backward to previous layers.

## Optimization Algorithms

Once gradients are computed, optimization algorithms update the parameters.

### 1. Gradient Descent

Update rule for parameter $`\theta`$:

```math
\theta \leftarrow \theta - \eta \frac{\partial L}{\partial \theta}
```
where $`\eta`$ is the learning rate.

### 2. Stochastic Gradient Descent (SGD)

Updates parameters using a single (or small batch) data point at each step, improving efficiency and generalization.

### 3. Advanced Optimizers

- **Momentum:** Accelerates SGD by adding a fraction of the previous update.
- **RMSprop, Adam:** Adapt learning rates for each parameter.

## Python Example: Manual Backpropagation for a Simple Network

```python
import numpy as np

# Simple 1-layer network: y = w * x + b, MSE loss
x = np.array([1.0, 2.0, 3.0])
y_true = np.array([2.0, 4.0, 6.0])

w = 0.0
b = 0.0
lr = 0.01

for epoch in range(1000):
    y_pred = w * x + b
    loss = np.mean((y_true - y_pred) ** 2)
    # Gradients
    dw = -2 * np.mean((y_true - y_pred) * x)
    db = -2 * np.mean(y_true - y_pred)
    # Update
    w -= lr * dw
    b -= lr * db
    if epoch % 200 == 0:
        print(f"Epoch {epoch}: loss={loss:.4f}, w={w:.4f}, b={b:.4f}")

print(f"Trained weights: w={w:.4f}, b={b:.4f}")
```

## Key Takeaways
- Backpropagation efficiently computes gradients for all parameters.
- Optimization algorithms use gradients to update weights and minimize loss.
- Adam and other advanced optimizers are widely used in deep learning.

## References
- [Neural Networks and Deep Learning (Michael Nielsen)](http://neuralnetworksanddeeplearning.com/)
- [Deep Learning Book (Goodfellow et al.)](https://www.deeplearningbook.org/) 