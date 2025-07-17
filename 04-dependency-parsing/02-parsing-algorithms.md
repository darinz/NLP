# 02. Parsing Algorithms

## Introduction

Parsing algorithms are used to construct dependency trees from sentences, revealing the syntactic structure and relationships between words. This guide covers key parsing algorithms, their mathematical foundations, and practical Python examples.

## Types of Dependency Parsing Algorithms

1. **Transition-based parsing**
2. **Graph-based parsing**

---

## 1. Transition-Based Parsing

Transition-based parsers incrementally build a dependency tree by applying a sequence of actions (transitions) to a data structure (stack, buffer).

### Arc-Standard Algorithm

- Uses a stack and a buffer.
- At each step, chooses one of three actions:
  - **SHIFT**: Move the first word from the buffer to the stack.
  - **LEFT-ARC**: Add a dependency from the top of the stack to the second-top, then remove the second-top.
  - **RIGHT-ARC**: Add a dependency from the second-top of the stack to the top, then remove the top.

#### Example Step
Suppose the stack is $`[ROOT, I, saw]`$ and the buffer is $`[her, duck]`$.
- **SHIFT** moves "her" to the stack.
- **LEFT-ARC** or **RIGHT-ARC** creates dependencies.

### Python Example: Simulating Arc-Standard Steps

```python
stack = ["ROOT"]
buffer = ["I", "saw", "her", "duck"]
deps = []

# Example: SHIFT, SHIFT, SHIFT, RIGHT-ARC, SHIFT, RIGHT-ARC, RIGHT-ARC
stack.append(buffer.pop(0))  # SHIFT: I
stack.append(buffer.pop(0))  # SHIFT: saw
stack.append(buffer.pop(0))  # SHIFT: her
# RIGHT-ARC: saw -> her
deps.append((stack[-2], stack[-1]))
stack.pop()
stack.append(buffer.pop(0))  # SHIFT: duck
# RIGHT-ARC: her -> duck
deps.append((stack[-2], stack[-1]))
stack.pop()
# RIGHT-ARC: I -> saw
deps.append((stack[-2], stack[-1]))
stack.pop()
print("Dependencies:", deps)
```

---

## 2. Graph-Based Parsing

Graph-based parsers score all possible dependency trees and select the highest-scoring one, often using dynamic programming.

### Maximum Spanning Tree (MST) Parsing

- Construct a directed graph where nodes are words and edges are possible dependencies, each with a score.
- Find the maximum spanning tree (MST) rooted at ROOT.

#### Chu-Liu/Edmonds' Algorithm
- Finds the MST in a directed graph.

### Mathematical Formulation

Given a sentence with $`n`$ words, let $`G = (V, E)`$ be a directed graph:
- $`V = \{0, 1, ..., n\}`$ (0 is ROOT)
- $`E`$ is the set of possible dependencies
- Each edge $`(h, d)`$ has a score $`s(h, d)`$

The goal is to find the tree $`T^*`$ maximizing the total score:

```math
T^* = \arg\max_{T \in \mathcal{T}} \sum_{(h, d) \in T} s(h, d)
```
where $`\mathcal{T}`$ is the set of all valid trees.

## Python Example: Using spaCy for Dependency Parsing

```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("I saw her duck.")
for token in doc:
    print(f"{token.text} <--{token.dep_}-- {token.head.text}")
```

## Key Takeaways
- Transition-based parsers build trees incrementally with actions.
- Graph-based parsers score all possible trees and select the best.
- Modern NLP libraries (e.g., spaCy, Stanza) use efficient parsing algorithms.

## References
- [Speech and Language Processing (Jurafsky & Martin)](https://web.stanford.edu/~jurafsky/slp3/)
- [spaCy Dependency Parsing](https://spacy.io/usage/linguistic-features#dependency-parse) 