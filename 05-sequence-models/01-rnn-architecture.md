# 01. RNN Architecture

## Introduction

Recurrent Neural Networks (RNNs) are a class of neural networks designed to process sequential data, such as text, speech, or time series. RNNs maintain a hidden state that captures information about previous elements in the sequence, allowing them to model temporal dependencies.

## Why RNNs?

Traditional neural networks (e.g., MLPs) cannot handle variable-length sequences or remember previous inputs. RNNs address this by using loops in their architecture to maintain context.

## Basic RNN Cell

At each time step $`t`$, the RNN receives an input $`x_t`$ and updates its hidden state $`h_t`$:

```math
h_t = \sigma(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
```
```math
y_t = W_{hy} h_t + b_y
```
where:
- $`h_{t-1}`$: previous hidden state
- $`x_t`$: current input
- $`\sigma`$: activation function (e.g., tanh or ReLU)
- $`W_{hh}, W_{xh}, W_{hy}`$: weight matrices
- $`b_h, b_y`$: bias vectors

## Unrolling the RNN

An RNN can be "unrolled" over time to visualize how the hidden state is updated at each step:

```
x_1 → [RNN] → h_1 → y_1
x_2 → [RNN] → h_2 → y_2
...
x_T → [RNN] → h_T → y_T
```

## Limitations of Vanilla RNNs
- Struggle with long-term dependencies due to vanishing/exploding gradients.
- Solutions: LSTM and GRU architectures (covered in advanced guides).

## Python Example: Simple RNN with Keras

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Example: sequence classification (toy data)
X = np.random.randn(100, 10, 8)  # 100 samples, 10 timesteps, 8 features
y = np.random.randint(0, 2, size=(100, 1))

model = Sequential([
    SimpleRNN(16, input_shape=(10, 8)),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X, y, epochs=5, batch_size=8)
```

## Key Takeaways
- RNNs process sequences by maintaining a hidden state.
- They are suitable for variable-length and sequential data.
- Vanilla RNNs have limitations for long sequences, motivating advanced variants.

## References
- [Understanding LSTM Networks (Colah's Blog)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Deep Learning Book (Goodfellow et al.)](https://www.deeplearningbook.org/) 