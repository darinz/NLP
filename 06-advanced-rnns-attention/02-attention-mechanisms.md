# 02. Attention Mechanisms

## Introduction

Attention mechanisms allow neural networks to focus on relevant parts of the input sequence when making predictions. They have revolutionized sequence modeling, enabling models to handle long-range dependencies and improving performance in tasks like translation and summarization.

## Why Attention?

Vanilla seq2seq models compress the entire input into a single context vector, which can be a bottleneck for long sequences. Attention lets the model dynamically access all encoder hidden states, providing more flexibility and context.

## Basic Attention Mechanism

Given encoder hidden states $`h_1, h_2, ..., h_T`$ and decoder state $`s_t`$ at time $`t`$:

1. **Score each encoder state:**

```math
e_{t,i} = \text{score}(s_t, h_i)
```

2. **Compute attention weights (softmax):**

```math
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^T \exp(e_{t,j})}
```

3. **Compute context vector:**

```math
c_t = \sum_{i=1}^T \alpha_{t,i} h_i
```

4. **Use $`c_t`$ and $`s_t`$ to generate the output.**

## Types of Attention

- **Additive (Bahdanau) Attention:**
  - $`e_{t,i} = v_a^T \tanh(W_a s_t + U_a h_i)`$
- **Dot-Product (Luong) Attention:**
  - $`e_{t,i} = s_t^T h_i`$

## Python Example: Dot-Product Attention (Numpy)

```python
import numpy as np

def dot_product_attention(query, keys, values):
    # query: (d,)
    # keys, values: (T, d)
    scores = np.dot(keys, query)  # (T,)
    weights = np.exp(scores) / np.sum(np.exp(scores))  # softmax
    context = np.sum(weights[:, None] * values, axis=0)
    return context, weights

# Example usage
query = np.random.rand(8)
keys = np.random.rand(5, 8)
values = np.random.rand(5, 8)
context, weights = dot_product_attention(query, keys, values)
print("Attention weights:", weights)
print("Context vector:", context)
```

## Visualizing Attention

Attention weights can be visualized as a heatmap, showing which input tokens the model focuses on for each output token.

## Key Takeaways
- Attention enables models to focus on relevant parts of the input.
- Improves performance on long sequences and complex tasks.
- Foundation for the Transformer architecture.

## References
- [Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2015)](https://arxiv.org/abs/1409.0473)
- [Effective Approaches to Attention-based Neural Machine Translation (Luong et al., 2015)](https://arxiv.org/abs/1508.04025)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) 