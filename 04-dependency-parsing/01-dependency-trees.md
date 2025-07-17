# 01. Dependency Trees

## Introduction

Dependency trees are a way to represent the grammatical structure of a sentence by showing how words are related to each other. Each word (except the root) depends on another word, forming a tree structure that captures syntactic relationships.

## What is a Dependency Tree?

- A **dependency tree** is a directed tree where:
  - Each node is a word in the sentence.
  - Each edge represents a dependency (syntactic relation) from a head (governing word) to a dependent (modifier).
  - There is a single root (often the main verb).

### Example

Consider the sentence: "She enjoys reading books."

- "enjoys" is the root (main verb).
- "She" depends on "enjoys" (subject).
- "reading" depends on "enjoys" (object).
- "books" depends on "reading" (object of gerund).

This can be visualized as:

```
  enjoys
   /   \
She  reading
         |
       books
```

## Formal Definition

A dependency tree for a sentence with $`n`$ words is a directed tree $`T = (V, E)`$ where:
- $`V = \{0, 1, ..., n\}`$ (0 is the root, 1 to $`n`$ are words)
- $`E`$ is a set of directed edges $`(h, d)`$ from head $`h`$ to dependent $`d`$
- Each word (except root) has exactly one head
- The tree is connected and acyclic

## Dependency Relations

Common dependency relations include:
- **nsubj**: nominal subject
- **obj**: object
- **amod**: adjectival modifier
- **det**: determiner
- **root**: root of the sentence

## Python Example: Visualizing Dependency Trees with spaCy

```python
import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
sentence = "She enjoys reading books."
doc = nlp(sentence)
displacy.serve(doc, style="dep")  # Opens a browser visualization
```

## Key Takeaways
- Dependency trees represent syntactic structure as head-dependent relations.
- Each word (except root) has one head, forming a tree.
- Widely used in syntactic parsing and downstream NLP tasks.

## References
- [Speech and Language Processing (Jurafsky & Martin)](https://web.stanford.edu/~jurafsky/slp3/)
- [spaCy Dependency Parsing](https://spacy.io/usage/linguistic-features#dependency-parse) 