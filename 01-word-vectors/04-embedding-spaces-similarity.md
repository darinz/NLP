# 04. Embedding Spaces and Similarity

## Introduction

Word embeddings map words to points in a high-dimensional vector space. The structure of this space encodes semantic and syntactic relationships between words. Understanding how to measure similarity and explore embedding spaces is crucial for many NLP applications.

## Embedding Space

- Each word is represented as a dense vector (e.g., 50, 100, or 300 dimensions).
- Words with similar meanings are close together in this space.
- Embeddings can be visualized in 2D or 3D using dimensionality reduction techniques.

## Measuring Similarity

### 1. Cosine Similarity

The most common measure for word similarity is cosine similarity:

```math
\text{cosine\_sim}(\vec{a}, \vec{b}) = \frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\| \|\vec{b}\|}
```

- Ranges from -1 (opposite) to 1 (identical).
- 0 means orthogonal (unrelated).

### 2. Euclidean Distance

Another measure is Euclidean distance:

```math
\text{distance}(\vec{a}, \vec{b}) = \sqrt{\sum_{i=1}^n (a_i - b_i)^2}
```

- Smaller distance means more similar.
- Not scale-invariant (unlike cosine similarity).

## Example: Calculating Similarity in Python

```python
import numpy as np
from numpy.linalg import norm

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def euclidean_distance(a, b):
    return norm(a - b)

# Example vectors (from a trained embedding model)
vec_king = np.array([0.5, 0.8, 0.3])
vec_queen = np.array([0.45, 0.85, 0.33])
vec_apple = np.array([0.1, -0.2, 0.7])

print("Cosine similarity (king, queen):", cosine_similarity(vec_king, vec_queen))
print("Cosine similarity (king, apple):", cosine_similarity(vec_king, vec_apple))
print("Euclidean distance (king, queen):", euclidean_distance(vec_king, vec_queen))
```

## Analogy Reasoning in Embedding Space

Word embeddings can solve analogies using vector arithmetic:

```math
\text{king} - \text{man} + \text{woman} \approx \text{queen}
```

This works because relationships are encoded as consistent vector offsets.

### Example in Python (with Gensim)

```python
from gensim.models import Word2Vec
# Assume 'model' is a trained Word2Vec or FastText model
result = model.wv.most_similar(positive=['king', 'woman'], negative=['man'])
print(result)  # [('queen', ...), ...]
```

## Visualizing Embedding Spaces

You can use PCA or t-SNE to reduce dimensions for visualization.

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Assume 'model' is a trained embedding model
words = ['king', 'queen', 'man', 'woman', 'apple', 'orange']
X = np.array([model.wv[w] for w in words])

pca = PCA(n_components=2)
result = pca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.title('Word Embeddings Visualized with PCA')
plt.show()
```

## Key Takeaways
- Embedding spaces encode semantic relationships as geometric structure.
- Cosine similarity is the most common measure for word similarity.
- Vector arithmetic enables analogy reasoning.
- Visualization helps interpret embedding spaces.

## References
- [Efficient Estimation of Word Representations in Vector Space (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781)
- [Gensim Documentation](https://radimrehurek.com/gensim/) 