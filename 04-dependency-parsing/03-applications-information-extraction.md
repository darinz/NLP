# 03. Applications in Information Extraction

## Introduction

Dependency parsing is a powerful tool for information extraction (IE) in NLP. By analyzing grammatical relationships, we can extract structured information such as entities, relations, and events from unstructured text.

## What is Information Extraction?

Information extraction is the process of automatically extracting structured information (e.g., facts, relationships) from unstructured text. Common IE tasks include:
- Named Entity Recognition (NER)
- Relation Extraction
- Event Extraction

## Why Use Dependency Parsing for IE?

- Dependency trees reveal syntactic relationships between words.
- They help identify subject, object, and predicate in sentences.
- Enable extraction of relations that are not adjacent in the text.

## Example: Relation Extraction

Consider the sentence: "Marie Curie discovered polonium."
- Subject: "Marie Curie" (nsubj of "discovered")
- Object: "polonium" (obj of "discovered")
- Relation: "discovered"

## Python Example: Extracting Subject-Verb-Object Triples with spaCy

```python
import spacy
nlp = spacy.load("en_core_web_sm")
sentence = "Marie Curie discovered polonium in 1898."
doc = nlp(sentence)

for token in doc:
    if token.dep_ == "ROOT":
        subj = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
        obj = [w for w in token.rights if w.dep_ in ("dobj", "obj")]
        if subj and obj:
            print(f"Subject: {subj[0].text}, Verb: {token.text}, Object: {obj[0].text}")
```

## Advanced Applications

- **Event Extraction:** Identify events and their participants using dependency paths.
- **Slot Filling:** Extract specific information (e.g., who did what to whom, when, where).
- **Question Answering:** Use dependency structure to match questions to relevant answers.

## Mathematical Formulation: Dependency Path Extraction

Given a dependency tree $`T = (V, E)`$ for a sentence, a relation between two entities $`e_1`$ and $`e_2`$ can be defined by the shortest path $`P(e_1, e_2)`$ in $`T`$.

```math
P(e_1, e_2) = \text{ShortestPath}_T(e_1, e_2)
```

Features extracted from $`P(e_1, e_2)`$ can be used for relation classification.

## Key Takeaways
- Dependency parsing enables extraction of structured information from text.
- Subject-verb-object triples are a common pattern for relation extraction.
- Dependency paths are useful features for advanced IE tasks.

## References
- [Speech and Language Processing (Jurafsky & Martin)](https://web.stanford.edu/~jurafsky/slp3/)
- [spaCy Usage: Information Extraction](https://spacy.io/usage/examples#information-extraction) 