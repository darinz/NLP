# 02. Knowledge Extraction

## Introduction

Knowledge extraction is the process of automatically identifying and structuring information (entities, relationships, facts) from unstructured text. It is a key component of knowledge-based QA systems and knowledge graphs.

## Why Knowledge Extraction?

- Enables machines to understand and reason over text.
- Supports downstream tasks like QA, search, and recommendation.
- Forms the basis for building knowledge graphs and databases.

## Key Steps in Knowledge Extraction

### 1. Named Entity Recognition (NER)
- Identify entities (people, organizations, locations, etc.) in text.

### 2. Relation Extraction
- Identify relationships between entities (e.g., "Barack Obama was born in Hawaii").

### 3. Event Extraction
- Identify events and their participants (who did what, when, where).

## Mathematical Formulation

Given a sentence $`s`$, extract a set of triples $`(e_1, r, e_2)`$ where $`e_1`$ and $`e_2`$ are entities and $`r`$ is the relation.

```math
\text{Extract}(s) = \{(e_1, r, e_2)\}
```

## Python Example: NER and Relation Extraction with spaCy

```python
import spacy
nlp = spacy.load("en_core_web_sm")
text = "Barack Obama was born in Hawaii."
doc = nlp(text)

# Named Entity Recognition
for ent in doc.ents:
    print(ent.text, ent.label_)

# Simple Relation Extraction (subject, verb, object)
for token in doc:
    if token.dep_ == "ROOT":
        subj = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
        obj = [w for w in token.rights if w.dep_ in ("dobj", "obj")]
        if subj and obj:
            print(f"Relation: {subj[0].text} --{token.text}--> {obj[0].text}")
```

## Advanced: Knowledge Graph Construction
- Aggregate extracted triples into a graph structure.
- Nodes: entities; Edges: relations.

## Key Takeaways
- Knowledge extraction structures information for reasoning and retrieval.
- NER and relation extraction are core techniques.
- Tools like spaCy and Hugging Face Transformers make extraction accessible.

## References
- [spaCy Usage: Information Extraction](https://spacy.io/usage/examples#information-extraction)
- [A Survey on Open Information Extraction (Niklaus et al., 2018)](https://arxiv.org/abs/1806.05599) 