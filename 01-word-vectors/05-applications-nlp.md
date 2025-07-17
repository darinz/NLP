# 05. Applications of Word Embeddings in NLP Models

## Introduction

Word embeddings are foundational for modern NLP. They provide dense, information-rich representations of words that can be used in a variety of downstream tasks. This guide covers key applications and provides practical code examples.

## 1. Text Classification

Word embeddings are used as input features for text classification models (e.g., sentiment analysis, topic classification).

### Example: Sentiment Analysis with Embeddings

```python
import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression

# Example data
documents = [
    ["i", "love", "this", "movie"],
    ["this", "film", "was", "terrible"]
]
labels = [1, 0]  # 1: positive, 0: negative

# Train Word2Vec
model = Word2Vec(sentences=documents, vector_size=20, window=2, min_count=1, workers=1)

# Average word vectors for each document
def document_vector(doc):
    return np.mean([model.wv[w] for w in doc if w in model.wv], axis=0)

X = np.array([document_vector(doc) for doc in documents])

# Train classifier
clf = LogisticRegression()
clf.fit(X, labels)

# Predict
print(clf.predict(X))
```

## 2. Named Entity Recognition (NER)

NER models use word embeddings as input features to identify entities (names, locations, etc.) in text.

- Embeddings can be combined with sequence models (e.g., LSTM, CRF) for improved performance.

## 3. Machine Translation

Embeddings are used in encoder-decoder architectures for neural machine translation (NMT).

- Source and target words are mapped to embeddings.
- Embeddings help models generalize to unseen word combinations.

## 4. Question Answering

Embeddings help models understand context and match questions to relevant answers.

- Used in retrieval-based and generative QA systems.

## 5. Semantic Similarity & Search

Embeddings enable semantic search by comparing vector representations of queries and documents.

### Example: Semantic Search with Cosine Similarity

```python
from numpy.linalg import norm

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

query = ["love", "movie"]
documents = [
    ["i", "love", "this", "movie"],
    ["this", "film", "was", "terrible"]
]

query_vec = np.mean([model.wv[w] for w in query if w in model.wv], axis=0)
doc_vecs = [np.mean([model.wv[w] for w in doc if w in model.wv], axis=0) for doc in documents]

similarities = [cosine_similarity(query_vec, doc_vec) for doc_vec in doc_vecs]
print("Semantic similarities:", similarities)
```

## 6. Sequence Labeling (POS Tagging, Chunking)

Embeddings are used as input to sequence models (e.g., LSTM, GRU) for tasks like part-of-speech tagging and chunking.

## 7. Transfer Learning & Pretrained Embeddings

Pretrained embeddings (Word2Vec, GloVe, FastText) can be used to initialize models, improving performance and reducing training time.

## Key Takeaways
- Word embeddings are used in almost every modern NLP model.
- They improve performance by capturing semantic and syntactic information.
- Pretrained embeddings are widely used for transfer learning.

## References
- [Neural Network Methods for Natural Language Processing (Goldberg, 2016)](https://arxiv.org/abs/1511.00743)
- [Gensim Documentation](https://radimrehurek.com/gensim/) 