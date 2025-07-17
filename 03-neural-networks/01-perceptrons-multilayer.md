# 01. Perceptrons and Multilayer Networks

## Introduction

Perceptrons and multilayer networks are the building blocks of modern neural networks. This guide covers the basics of single-layer perceptrons, their limitations, and how multilayer networks (MLPs) overcome these limitations.

## The Perceptron

A **perceptron** is the simplest type of artificial neuron. It computes a weighted sum of its inputs and applies an activation function (usually a step function) to produce an output.

### Mathematical Formulation

Given input vector $`\mathbf{x} = [x_1, x_2, ..., x_n]`$ and weights $`\mathbf{w} = [w_1, w_2, ..., w_n]`$, the perceptron computes:

```math
y = f(\mathbf{w} \cdot \mathbf{x} + b)
```
where $`b`$ is the bias and $`f`$ is the activation function (e.g., step or sign function).

### Perceptron Learning Rule

The perceptron is trained using the following update rule:

```math
w_i \leftarrow w_i + \eta (y_{true} - y_{pred}) x_i
```
where $`\eta`$ is the learning rate, $`y_{true}`$ is the true label, and $`y_{pred}`$ is the predicted label.

## Limitations of Single-Layer Perceptrons
- Can only solve linearly separable problems (e.g., cannot solve XOR).
- Cannot model complex, nonlinear relationships.

## Multilayer Perceptrons (MLPs)

An **MLP** consists of multiple layers of perceptrons (neurons) with nonlinear activation functions. This allows the network to model complex, nonlinear functions.

### Architecture
- **Input layer:** Receives the input features.
- **Hidden layer(s):** One or more layers of neurons with nonlinear activations.
- **Output layer:** Produces the final prediction.

### Mathematical Formulation

For a single hidden layer MLP:

```math
\mathbf{h} = f(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)
```
```math
\mathbf{y} = g(\mathbf{W}_2 \mathbf{h} + \mathbf{b}_2)
```
where $`f`$ and $`g`$ are activation functions (e.g., ReLU, sigmoid).

## Python Example: Simple MLP for Classification

```python
import numpy as np
from sklearn.neural_network import MLPClassifier

# Example data: XOR problem
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 1, 1, 0])

# MLP with one hidden layer of 2 neurons
mlp = MLPClassifier(hidden_layer_sizes=(2,), activation='relu', solver='adam', max_iter=1000)
mlp.fit(X, y)

print("Predictions:", mlp.predict(X))
```

## Key Takeaways
- Perceptrons are the simplest neural units but are limited to linear problems.
- MLPs with nonlinear activations can solve complex, nonlinear tasks.
- MLPs are the foundation for deep learning architectures.

## References
- [Neural Networks and Deep Learning (Michael Nielsen)](http://neuralnetworksanddeeplearning.com/)
- [Deep Learning Book (Goodfellow et al.)](https://www.deeplearningbook.org/) 