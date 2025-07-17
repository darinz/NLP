# 01. Word2Vec: Learning Word Embeddings

## Introduction

Word2Vec is a family of models that learn to represent words as dense vectors (embeddings) such that words with similar meanings are close together in the vector space. Developed by Mikolov et al. at Google in 2013, Word2Vec has become a foundational technique in NLP.

## Why Word Embeddings?

Traditional NLP methods represented words as one-hot vectors, which are sparse and do not capture semantic relationships. Word embeddings solve this by mapping words to dense, low-dimensional vectors that encode semantic similarity.

## Word2Vec Architectures

Word2Vec comes in two main flavors:
- **Continuous Bag of Words (CBOW):** Predicts a target word from its context.
- **Skip-Gram:** Predicts context words given a target word.

### 1. Continuous Bag of Words (CBOW)

Given a context (surrounding words), CBOW predicts the target word.

#### Mathematical Formulation

Given a sequence of words $`w_1, w_2, ..., w_T`$, the objective is to maximize the average log probability:

```math
\frac{1}{T} \sum_{t=1}^{T} \log p(w_t | context(w_t))
```

Where $`context(w_t)`$ is the set of surrounding words.

### 2. Skip-Gram

Skip-Gram does the opposite: given a target word, it predicts the surrounding context words.

#### Mathematical Formulation

```math
\frac{1}{T} \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log p(w_{t+j} | w_t)
```

Where $`c`$ is the context window size.

## The Softmax Function

The probability of a word $`w_O`$ given input word $`w_I`$ is computed using softmax:

```math
p(w_O | w_I) = \frac{\exp({v'_{w_O}}^T v_{w_I})}{\sum_{w=1}^{W} \exp({v'_w}^T v_{w_I})}
```

Where:
- $`v_{w_I}`$: input vector of word $`w_I`$
- $`v'_{w_O}`$: output vector of word $`w_O`$
- $`W`$: vocabulary size

## Training Word2Vec

Training is typically done using stochastic gradient descent and optimizations like **Negative Sampling** or **Hierarchical Softmax** to make computation feasible for large vocabularies.

### Negative Sampling

Instead of updating all weights, only a small number of negative samples (random words) are updated for each training example.

## Python Example: Training Word2Vec with Gensim

```python
from gensim.models import Word2Vec

# Example corpus
documents = [
    ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
    ["i", "love", "natural", "language", "processing"],
    ["word2vec", "creates", "word", "embeddings"]
]

# Train a Skip-Gram model
model = Word2Vec(sentences=documents, vector_size=50, window=2, sg=1, min_count=1, workers=2)

# Get the embedding for a word
vector = model.wv["word2vec"]
print("Embedding for 'word2vec':", vector)

# Find most similar words
print(model.wv.most_similar("word2vec"))
```

## Visualizing Embeddings

You can use dimensionality reduction techniques like t-SNE or PCA to visualize word embeddings in 2D.

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

words = list(model.wv.index_to_key)
X = model.wv[words]

pca = PCA(n_components=2)
result = pca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.title('Word2Vec Embeddings Visualized with PCA')
plt.show()
```

## Key Takeaways
- Word2Vec learns dense vector representations of words.
- CBOW predicts a word from its context; Skip-Gram predicts context from a word.
- Negative sampling makes training efficient.
- Embeddings capture semantic relationships and can be used in downstream NLP tasks.

## References
- [Efficient Estimation of Word Representations in Vector Space (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781)
- [Gensim Documentation](https://radimrehurek.com/gensim/) 