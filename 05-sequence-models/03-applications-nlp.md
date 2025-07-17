# 03. Applications of Sequence Models in NLP

## Introduction

Sequence models, especially RNNs and their variants, are widely used in NLP for tasks that require understanding and generating sequences. This guide covers key applications, concepts, and practical code examples.

## 1. Language Modeling

Language models predict the next word in a sequence, enabling text generation, autocomplete, and more.

### Example: Next-Word Prediction

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Toy data: 100 samples, 10 timesteps, 8 features
X = np.random.randn(100, 10, 8)
y = np.random.randint(0, 2, size=(100, 1))

model = Sequential([
    SimpleRNN(16, input_shape=(10, 8)),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X, y, epochs=2, batch_size=8)
```

## 2. Machine Translation

Seq2seq models translate text from one language to another.

- Input: Source language sentence
- Output: Target language sentence

## 3. Text Summarization

Sequence models generate concise summaries of longer texts.
- **Extractive:** Selects important sentences/phrases.
- **Abstractive:** Generates new sentences to summarize content.

## 4. Named Entity Recognition (NER)

NER identifies entities (names, locations, etc.) in text. Sequence models label each token in a sequence.

### Example: Sequence Labeling with RNN (Keras)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, TimeDistributed
import numpy as np

X = np.random.randn(100, 10, 8)  # 100 samples, 10 timesteps, 8 features
y = np.random.randint(0, 2, size=(100, 10, 1))  # 0/1 label per timestep

model = Sequential([
    SimpleRNN(16, return_sequences=True, input_shape=(10, 8)),
    TimeDistributed(Dense(1, activation='sigmoid'))
])
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X, y, epochs=2, batch_size=8)
```

## 5. Sentiment Analysis

Sequence models can classify the sentiment of a sentence or document, especially when context and order matter.

## 6. Speech Recognition

RNNs and seq2seq models convert audio sequences into text.

## 7. Question Answering

Sequence models help match questions to answers or generate answers from context.

## Key Takeaways
- Sequence models are essential for tasks involving ordered data in NLP.
- RNNs, LSTMs, GRUs, and seq2seq models power many state-of-the-art applications.
- Python libraries like TensorFlow and PyTorch make it easy to build and train sequence models.

## References
- [Speech and Language Processing (Jurafsky & Martin)](https://web.stanford.edu/~jurafsky/slp3/)
- [TensorFlow Sequence Models Guide](https://www.tensorflow.org/guide/keras/rnn) 