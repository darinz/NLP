# 01. LSTM and GRU Architectures

## Introduction

Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) are advanced RNN variants designed to address the vanishing gradient problem and capture long-range dependencies in sequences. This guide explains their architectures, math, and provides Python code examples.

## Why LSTM and GRU?

Vanilla RNNs struggle to learn long-term dependencies due to vanishing/exploding gradients. LSTM and GRU introduce gating mechanisms to control information flow and maintain memory over longer sequences.

---

## LSTM Architecture

An LSTM cell maintains a cell state $`c_t`$ and a hidden state $`h_t`$. It uses three gates:
- **Forget gate** $`f_t`$
- **Input gate** $`i_t`$
- **Output gate** $`o_t`$

### LSTM Equations

```math
f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)
i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)
o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)
\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
h_t = o_t \odot \tanh(c_t)
```
where $`\odot`$ is element-wise multiplication and $`\sigma`$ is the sigmoid function.

---

## GRU Architecture

The GRU simplifies the LSTM by combining the forget and input gates into an **update gate** $`z_t`$ and using a **reset gate** $`r_t`$.

### GRU Equations

```math
z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)
r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)
\tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h)
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
```

---

## Comparison: LSTM vs. GRU

| Feature         | LSTM         | GRU         |
|----------------|--------------|-------------|
| Gates          | 3 (input, forget, output) | 2 (update, reset) |
| Cell state     | Yes          | No (merged with hidden) |
| Complexity     | Higher       | Lower       |
| Performance    | Similar, GRU often faster |

---

## Python Example: LSTM and GRU with Keras

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense

X = np.random.randn(100, 10, 8)  # 100 samples, 10 timesteps, 8 features
y = np.random.randint(0, 2, size=(100, 1))

# LSTM model
lstm_model = Sequential([
    LSTM(16, input_shape=(10, 8)),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy')
lstm_model.summary()
lstm_model.fit(X, y, epochs=2, batch_size=8)

# GRU model
gru_model = Sequential([
    GRU(16, input_shape=(10, 8)),
    Dense(1, activation='sigmoid')
])
gru_model.compile(optimizer='adam', loss='binary_crossentropy')
gru_model.summary()
gru_model.fit(X, y, epochs=2, batch_size=8)
```

## Key Takeaways
- LSTM and GRU address the limitations of vanilla RNNs.
- Gating mechanisms help retain and control information over long sequences.
- Both are widely used in NLP and sequence modeling tasks.

## References
- [Understanding LSTM Networks (Colah's Blog)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Deep Learning Book (Goodfellow et al.)](https://www.deeplearningbook.org/) 