# 02. GloVe: Global Vectors for Word Representation

## Introduction

GloVe (Global Vectors) is an unsupervised learning algorithm for obtaining vector representations for words. Developed by Pennington, Socher, and Manning at Stanford, GloVe combines the benefits of global matrix factorization (like LSA) and local context window methods (like Word2Vec).

## Motivation

GloVe is based on the idea that ratios of word-word co-occurrence probabilities encode meaning. For example, the ratio of probabilities of co-occurrence for "ice" with "solid" versus "gas" captures the relationship between these words.

## The Co-occurrence Matrix

GloVe starts by constructing a large matrix $`X`$ where $`X_{ij}`$ is the number of times word $`j`$ occurs in the context of word $`i`$.

## The GloVe Objective

The goal is to learn word vectors such that their dot product equals the logarithm of the words' probability of co-occurrence:

```math
J = \sum_{i,j=1}^V f(X_{ij}) (w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2
```

Where:
- $`w_i`$, $`\tilde{w}_j`$: word and context word vectors
- $`b_i`$, $`\tilde{b}_j`$: bias terms
- $`f(X_{ij})`$: weighting function to reduce the influence of very frequent co-occurrences
- $`V`$: vocabulary size

### Weighting Function

```math
f(x) = \begin{cases}
    (x/x_{max})^\alpha & \text{if } x < x_{max} \\
    1 & \text{otherwise}
\end{cases}
```

Typical values: $`x_{max} = 100`$, $`\alpha = 0.75`$

## Why Log Co-occurrence?

The log of co-occurrence probabilities captures the intuition that the relationship between words is often best described by ratios, not raw counts.

## Training GloVe

GloVe is trained using stochastic gradient descent to minimize the objective function above. The resulting vectors can be used just like Word2Vec embeddings.

## Python Example: Using Pre-trained GloVe Embeddings

Training GloVe from scratch is computationally intensive, but you can use pre-trained vectors easily.

```python
import numpy as np

def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Example usage (download GloVe vectors from https://nlp.stanford.edu/projects/glove/)
glove_path = 'glove.6B.50d.txt'  # Update with your path
embeddings = load_glove_embeddings(glove_path)

print("Vector for 'king':", embeddings['king'])
```

## Visualizing GloVe Embeddings

You can visualize GloVe embeddings using PCA or t-SNE, similar to Word2Vec.

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

words = ['king', 'queen', 'man', 'woman', 'apple', 'orange', 'fruit']
X = np.array([embeddings[w] for w in words])

pca = PCA(n_components=2)
result = pca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.title('GloVe Embeddings Visualized with PCA')
plt.show()
```

## Key Takeaways
- GloVe leverages global word co-occurrence statistics.
- The objective is to make the dot product of word vectors approximate the log of co-occurrence counts.
- Pre-trained GloVe vectors are widely available and useful for many NLP tasks.

## References
- [GloVe: Global Vectors for Word Representation (Pennington et al., 2014)](https://nlp.stanford.edu/pubs/glove.pdf)
- [Stanford GloVe Project](https://nlp.stanford.edu/projects/glove/) 