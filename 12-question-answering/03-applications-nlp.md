# 03. Applications in NLP

## Introduction

Question answering (QA) and knowledge extraction techniques are widely used in real-world NLP applications. They enable systems to understand, retrieve, and reason over information, powering a range of intelligent services.

## Key Applications

### 1. Virtual Assistants and Chatbots
- Use QA systems to answer user queries, provide recommendations, and automate tasks.
- Combine retrieval, extraction, and generative models for robust performance.

### 2. Search Engines
- Enhance search by extracting answers directly from documents (e.g., featured snippets).
- Use knowledge extraction to build and update knowledge graphs for better search relevance.

### 3. Customer Support Automation
- Automatically answer customer questions using extractive and generative QA.
- Route complex queries to human agents with supporting evidence.

### 4. Healthcare and Legal QA
- Extract and reason over domain-specific knowledge (e.g., medical literature, legal documents).
- Support professionals with evidence-based answers.

### 5. Business Intelligence
- Extract structured knowledge from reports, emails, and documents.
- Enable querying of enterprise knowledge bases.

## Mathematical Perspective: End-to-End QA Pipeline

Given a user query $`q`$ and a corpus $`C`$:

```math
\text{Answer}(q, C) = \text{Generate}(q, \text{Retrieve}(q, C), \text{Extract}(q, c))
```
where $`\text{Retrieve}`$ finds relevant contexts $`c`$, $`\text{Extract}`$ finds answer spans, and $`\text{Generate}`$ produces the final answer.

## Python Example: QA in a Chatbot with Hugging Face Transformers

```python
from transformers import pipeline

# Build a QA pipeline
qa = pipeline("question-answering")

# Example: customer support chatbot
context = "Our return policy allows returns within 30 days of purchase. Please keep your receipt."
question = "How long do I have to return an item?"
result = qa(question=question, context=context)
print("Chatbot answer:", result["answer"])
```

## Python Example: Knowledge Graph Querying with spaCy

```python
import spacy
nlp = spacy.load("en_core_web_sm")
text = "Elon Musk founded SpaceX in 2002."
doc = nlp(text)
triples = []
for token in doc:
    if token.dep_ == "ROOT":
        subj = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
        obj = [w for w in token.rights if w.dep_ in ("dobj", "obj")]
        if subj and obj:
            triples.append((subj[0].text, token.text, obj[0].text))
print("Extracted triples:", triples)
```

## Key Takeaways
- QA and knowledge extraction power a wide range of NLP applications.
- They enable intelligent assistants, advanced search, and automated support.
- Python libraries like Hugging Face Transformers and spaCy make these applications accessible.

## References
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [spaCy Usage: Information Extraction](https://spacy.io/usage/examples#information-extraction) 