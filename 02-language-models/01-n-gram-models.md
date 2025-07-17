# 01. N-gram Language Models

## Introduction

N-gram language models are statistical models that predict the probability of a word based on the previous $`n-1`$ words. They are foundational in NLP for tasks like text generation, spelling correction, and speech recognition.

## What is an N-gram?

An **N-gram** is a contiguous sequence of $`n`$ items (typically words) from a given text. For example:
- Unigram: $`n=1`$ ("the")
- Bigram: $`n=2`$ ("the cat")
- Trigram: $`n=3`$ ("the cat sat")

## N-gram Model Definition

The probability of a word sequence $`w_1, w_2, ..., w_T`$ is approximated as:

```math
P(w_1, w_2, ..., w_T) \approx \prod_{t=1}^T P(w_t | w_{t-(n-1)}, ..., w_{t-1})
```

For a bigram model ($`n=2`$):

```math
P(w_1, w_2, ..., w_T) \approx P(w_1) \prod_{t=2}^T P(w_t | w_{t-1})
```

## Estimating Probabilities

Probabilities are estimated from counts in a corpus:

```math
P(w_t | w_{t-1}) = \frac{\text{Count}(w_{t-1}, w_t)}{\text{Count}(w_{t-1})}
```

## Smoothing

To handle unseen N-grams, smoothing techniques are used. The simplest is **add-one (Laplace) smoothing**:

```math
P_{Laplace}(w_t | w_{t-1}) = \frac{\text{Count}(w_{t-1}, w_t) + 1}{\text{Count}(w_{t-1}) + V}
```
where $`V`$ is the vocabulary size.

## Python Example: Bigram Model

```python
from collections import defaultdict, Counter
import numpy as np

corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat lay on the rug"
]

# Tokenize
sentences = [s.lower().split() for s in corpus]

# Count bigrams and unigrams
bigram_counts = defaultdict(Counter)
unigram_counts = Counter()
for sentence in sentences:
    for i in range(len(sentence)):
        unigram_counts[sentence[i]] += 1
        if i > 0:
            bigram_counts[sentence[i-1]][sentence[i]] += 1

V = len(unigram_counts)

def bigram_prob(w_prev, w, smoothing=True):
    if smoothing:
        return (bigram_counts[w_prev][w] + 1) / (unigram_counts[w_prev] + V)
    else:
        return bigram_counts[w_prev][w] / unigram_counts[w_prev]

print(f"P('sat' | 'cat') = {bigram_prob('cat', 'sat'):.3f}")
print(f"P('on' | 'sat') = {bigram_prob('sat', 'on'):.3f}")
```

## Limitations
- N-gram models suffer from data sparsity for large $`n`$.
- They cannot capture long-range dependencies.
- Smoothing only partially addresses the problem of unseen N-grams.

## Key Takeaways
- N-gram models are simple and effective for many tasks.
- They are limited by context window size and data sparsity.
- Smoothing is essential for practical use.

## References
- [Speech and Language Processing (Jurafsky & Martin)](https://web.stanford.edu/~jurafsky/slp3/)
- [NLTK Book: Language Modeling](https://www.nltk.org/book/ch02.html) 