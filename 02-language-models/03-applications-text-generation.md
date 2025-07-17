# 03. Applications in Text Generation and Understanding

## Introduction

Language models are central to many NLP applications, including text generation, completion, summarization, and understanding. This guide covers key applications and provides practical code examples.

## 1. Text Generation

Language models can generate new text by sampling words according to their predicted probabilities.

### Example: Generating Text with a Trained Model

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Assume 'model', 'word2idx', 'idx2word', 'maxlen' are defined as in previous examples

def generate_text(seed_text, num_words=5):
    result = seed_text.split()
    for _ in range(num_words):
        tokens = [word2idx.get(w, 0) for w in result[-(maxlen-1):]]
        tokens = pad_sequences([tokens], maxlen=maxlen-1)
        pred = model.predict(tokens, verbose=0)
        idx = np.argmax(pred)
        next_word = idx2word.get(idx, '<unk>')
        result.append(next_word)
    return ' '.join(result)

print(generate_text("i love", num_words=5))
```

## 2. Text Completion

Given a partial sentence, language models can suggest likely continuations.

- Used in search engines, chatbots, and code editors.

## 3. Text Summarization

Language models can be used in extractive and abstractive summarization:
- **Extractive:** Selects important sentences/phrases.
- **Abstractive:** Generates new sentences to summarize content.

## 4. Machine Translation

Sequence-to-sequence models with attention (based on language models) are used for translating text between languages.

## 5. Question Answering

Language models help understand context and generate or retrieve answers to questions.

## 6. Sentiment Analysis and Classification

Language models provide features for classifying text sentiment, topic, or intent.

### Example: Sentiment Classification with Embeddings

```python
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
import numpy as np

documents = [
    ["i", "love", "nlp"],
    ["nlp", "is", "hard"]
]
labels = [1, 0]  # 1: positive, 0: negative

model = Word2Vec(sentences=documents, vector_size=10, window=2, min_count=1, workers=1)

def doc_vector(doc):
    return np.mean([model.wv[w] for w in doc if w in model.wv], axis=0)

X = np.array([doc_vector(doc) for doc in documents])
clf = LogisticRegression().fit(X, labels)
print(clf.predict(X))
```

## 7. Perplexity: Evaluating Language Models

**Perplexity** measures how well a language model predicts a sample. Lower perplexity indicates better performance.

```math
\text{Perplexity} = P(w_1, w_2, ..., w_T)^{-1/T}
```

Or, for log probabilities:

```math
\text{Perplexity} = \exp\left(-\frac{1}{T} \sum_{t=1}^T \log P(w_t | w_1, ..., w_{t-1})\right)
```

## Key Takeaways
- Language models are used in generation, completion, translation, summarization, and understanding.
- Perplexity is a standard metric for evaluating language models.
- Neural models enable more advanced applications than traditional N-gram models.

## References
- [Speech and Language Processing (Jurafsky & Martin)](https://web.stanford.edu/~jurafsky/slp3/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) 