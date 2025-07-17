# 03. FastText: Subword-Aware Word Embeddings

## Introduction

FastText is an extension of Word2Vec developed by Facebook AI Research. It improves upon Word2Vec by representing words as bags of character n-grams, allowing it to generate embeddings for rare and out-of-vocabulary words.

## Motivation

Word2Vec and GloVe treat each word as an atomic unit, which means they cannot handle rare or unseen words well. FastText addresses this by breaking words into subword units (character n-grams), capturing morphological information and improving performance on morphologically rich languages.

## How FastText Works

- Each word is represented as a collection of character n-grams (e.g., for 'where' and n=3: <wh, whe, her, ere, re>).
- The word vector is the sum of the vectors of its n-grams.
- The model is trained similarly to Skip-Gram or CBOW, but on n-grams instead of whole words.

### Mathematical Formulation

Given a word $`w`$, let $`G_w`$ be the set of its n-grams. The word vector is:

```math
v_w = \sum_{g \in G_w} z_g
```

Where $`z_g`$ is the vector for n-gram $`g`$.

The Skip-Gram objective is then applied as in Word2Vec, but using these subword vectors.

## Advantages of FastText

- Handles out-of-vocabulary words by composing them from n-grams.
- Captures subword information (prefixes, suffixes, roots).
- Works well for morphologically rich languages.

## Python Example: Training FastText with Gensim

```python
from gensim.models import FastText

# Example corpus
documents = [
    ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
    ["i", "love", "natural", "language", "processing"],
    ["fasttext", "creates", "subword", "embeddings"]
]

# Train a FastText model
model = FastText(sentences=documents, vector_size=50, window=2, min_count=1, workers=2, sg=1)

# Get the embedding for a word
vector = model.wv["fasttext"]
print("Embedding for 'fasttext':", vector)

# Get embedding for an out-of-vocabulary word
print("Embedding for 'unseenword':", model.wv["unseenword"])

# Find most similar words
print(model.wv.most_similar("fasttext"))
```

## Visualizing FastText Embeddings

You can visualize FastText embeddings using PCA or t-SNE, as with Word2Vec and GloVe.

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
plt.title('FastText Embeddings Visualized with PCA')
plt.show()
```

## Key Takeaways
- FastText represents words as bags of character n-grams.
- It can generate embeddings for out-of-vocabulary words.
- Useful for languages with rich morphology.

## References
- [Enriching Word Vectors with Subword Information (Bojanowski et al., 2017)](https://arxiv.org/abs/1607.04606)
- [Gensim FastText Documentation](https://radimrehurek.com/gensim/models/fasttext.html) 