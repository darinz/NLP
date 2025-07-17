# 01. Self-Attention

## Introduction

Self-attention is a mechanism that allows a model to weigh the importance of different words in a sequence when encoding a particular word. It is the core innovation behind the Transformer architecture, enabling models to capture long-range dependencies and parallelize computation.

## Why Self-Attention?

- Traditional RNNs process sequences sequentially, making it hard to capture distant dependencies and parallelize.
- Self-attention enables each word to directly attend to all other words in the sequence, regardless of their distance.

## The Self-Attention Mechanism

Given an input sequence of $`n`$ tokens, each represented by an embedding $`x_i`$:

1. **Compute Queries, Keys, and Values:**
   - $`Q = XW^Q`$
   - $`K = XW^K`$
   - $`V = XW^V`$
   where $`W^Q, W^K, W^V`$ are learnable weight matrices.

2. **Compute Attention Scores:**
   - For each token, compute the dot product of its query with all keys:

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```
where $`d_k`$ is the dimension of the key vectors.

3. **Softmax Normalization:**
   - The softmax ensures the attention weights sum to 1 for each token.

4. **Weighted Sum:**
   - Each output is a weighted sum of the value vectors, weighted by the attention scores.

## Multi-Head Self-Attention

Instead of computing a single attention, the model uses multiple "heads" to attend to information from different representation subspaces:

```math
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
```
where each $`\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`$.

## Python Example: Self-Attention (Numpy)

```python
import numpy as np

def self_attention(X, Wq, Wk, Wv):
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    return weights @ V

# Example usage
X = np.random.rand(5, 8)  # 5 tokens, 8-dim embeddings
Wq = np.random.rand(8, 8)
Wk = np.random.rand(8, 8)
Wv = np.random.rand(8, 8)
output = self_attention(X, Wq, Wk, Wv)
print("Self-attention output shape:", output.shape)
```

## Key Takeaways
- Self-attention enables each token to attend to all others in a sequence.
- It is the foundation of the Transformer and modern NLP models.
- Multi-head attention allows the model to capture diverse relationships.

## References
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) 