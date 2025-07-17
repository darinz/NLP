# 02. Activation Functions

## Introduction

Activation functions introduce nonlinearity into neural networks, enabling them to learn complex patterns. This guide covers common activation functions, their properties, mathematical definitions, and practical usage in Python.

## Why Activation Functions?

Without activation functions, neural networks would be equivalent to a single linear transformation, regardless of the number of layers. Nonlinear activations allow networks to approximate complex, nonlinear functions.

## Common Activation Functions

### 1. Step Function

The original perceptron used a step function:

```math
y = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{otherwise} \end{cases}
```
where $`z = \mathbf{w} \cdot \mathbf{x} + b`$.

### 2. Sigmoid (Logistic) Function

Maps input to the range (0, 1):

```math
\sigma(z) = \frac{1}{1 + e^{-z}}
```
- Smooth, differentiable, but can cause vanishing gradients.

### 3. Hyperbolic Tangent (tanh)

Maps input to the range (-1, 1):

```math
\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}
```
- Zero-centered, but still susceptible to vanishing gradients.

### 4. Rectified Linear Unit (ReLU)

Most popular in deep learning:

```math
\text{ReLU}(z) = \max(0, z)
```
- Simple, efficient, helps mitigate vanishing gradients.

### 5. Leaky ReLU

Allows a small, nonzero gradient when $`z < 0`$:

```math
\text{LeakyReLU}(z) = \begin{cases} z & \text{if } z \geq 0 \\ \alpha z & \text{otherwise} \end{cases}
```
where $`\alpha`$ is a small constant (e.g., 0.01).

### 6. Softmax

Used in output layers for multi-class classification:

```math
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
```
- Converts logits to probabilities that sum to 1.

## Python Example: Plotting Activation Functions

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0, z)

def leaky_relu(z, alpha=0.01):
    return np.where(z >= 0, z, alpha * z)

z = np.linspace(-5, 5, 200)
plt.figure(figsize=(8,6))
plt.plot(z, sigmoid(z), label='Sigmoid')
plt.plot(z, tanh(z), label='Tanh')
plt.plot(z, relu(z), label='ReLU')
plt.plot(z, leaky_relu(z), label='Leaky ReLU')
plt.legend()
plt.title('Activation Functions')
plt.xlabel('z')
plt.ylabel('Activation')
plt.grid(True)
plt.show()
```

## Key Takeaways
- Activation functions enable neural networks to learn nonlinear relationships.
- ReLU and its variants are widely used in deep learning.
- Choice of activation affects training dynamics and performance.

## References
- [Neural Networks and Deep Learning (Michael Nielsen)](http://neuralnetworksanddeeplearning.com/)
- [Deep Learning Book (Goodfellow et al.)](https://www.deeplearningbook.org/) 