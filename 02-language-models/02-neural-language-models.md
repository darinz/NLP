# 02. Neural Language Models

## Introduction

Neural language models use neural networks to predict the probability of the next word in a sequence, overcoming many limitations of traditional N-gram models. They can capture long-range dependencies and learn distributed representations of words.

## Motivation

N-gram models are limited by their fixed context window and data sparsity. Neural models, especially those using embeddings and deep architectures, can generalize better and handle larger contexts.

## Basic Neural Language Model Architecture

A simple feedforward neural language model (Bengio et al., 2003) predicts the next word given the previous $`n-1`$ words.

### Model Structure
- Input: Previous $`n-1`$ words, each mapped to a word embedding vector.
- Hidden layer(s): Nonlinear transformation (e.g., ReLU, tanh).
- Output: Softmax layer producing a probability distribution over the vocabulary.

### Mathematical Formulation

Given a context $`(w_{t-(n-1)}, ..., w_{t-1})`$, the model computes:

```math
P(w_t | w_{t-(n-1)}, ..., w_{t-1}) = \text{softmax}(f(E(w_{t-(n-1)}), ..., E(w_{t-1)}))
```
where $`E(w)`$ is the embedding of word $`w`$ and $`f`$ is a neural network function.

## Recurrent Neural Network (RNN) Language Models

RNNs process sequences of arbitrary length by maintaining a hidden state $`h_t`$ that summarizes the sequence up to time $`t`$.

```math
h_t = \sigma(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
```
```math
P(w_t | w_1, ..., w_{t-1}) = \text{softmax}(W_{hy} h_t + b_y)
```
where $`x_t`$ is the embedding of $`w_t`$.

## Long Short-Term Memory (LSTM) and GRU

LSTM and GRU are advanced RNN variants that address the vanishing gradient problem, allowing the model to capture longer dependencies.

## Python Example: Simple RNN Language Model (with Keras)

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Example data (toy corpus)
sentences = [
    "i love nlp",
    "nlp is fun",
    "i love deep learning"
]

# Build vocabulary
words = set(" ".join(sentences).split())
word2idx = {w: i+1 for i, w in enumerate(sorted(words))}  # 0 is reserved for padding
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(word2idx) + 1

# Prepare sequences
sequences = []
for s in sentences:
    tokens = [word2idx[w] for w in s.split()]
    for i in range(1, len(tokens)):
        seq = tokens[:i+1]
        sequences.append(seq)

maxlen = max(len(seq) for seq in sequences)
sequences = pad_sequences(sequences, maxlen=maxlen)
X = sequences[:, :-1]
y = to_categorical(sequences[:, -1], num_classes=vocab_size)

# Build model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=8, input_length=maxlen-1),
    SimpleRNN(16),
    Dense(vocab_size, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
model.fit(X, y, epochs=200, verbose=0)

# Predict next word
def predict_next(text):
    tokens = [word2idx[w] for w in text.split() if w in word2idx]
    tokens = pad_sequences([tokens], maxlen=maxlen-1)
    pred = model.predict(tokens, verbose=0)
    idx = np.argmax(pred)
    return idx2word.get(idx, "<unk>")

print("Next word after 'i love':", predict_next("i love"))
```

## Limitations
- Neural models require more data and computation than N-gram models.
- Training can be slow for large vocabularies.

## Key Takeaways
- Neural language models can capture long-range dependencies and learn word embeddings.
- RNNs, LSTMs, and GRUs are popular architectures for sequence modeling.
- Neural models outperform N-gram models on most NLP tasks.

## References
- [A Neural Probabilistic Language Model (Bengio et al., 2003)](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- [Speech and Language Processing (Jurafsky & Martin)](https://web.stanford.edu/~jurafsky/slp3/) 