# 01. Retrieval-Augmented Generation (RAG)

## Introduction

Retrieval-Augmented Generation (RAG) is a technique that combines large language models (LLMs) with external knowledge sources. By retrieving relevant documents and conditioning generation on them, RAG improves factual accuracy, reduces hallucination, and enables up-to-date responses.

## Why Retrieval-Augmented Generation?

- LLMs have limited context and may not know recent or domain-specific facts.
- Retrieval provides access to external knowledge, enhancing model capabilities.
- RAG enables dynamic, context-aware, and factually grounded generation.

## RAG Architecture

1. **Retriever:**
   - Given a query, retrieves relevant documents/passages from a large corpus.
   - Common retrievers: BM25, dense vector search (e.g., DPR, FAISS).
2. **Generator:**
   - Conditions on the query and retrieved documents to generate an answer.
   - Typically a sequence-to-sequence model (e.g., BART, T5).

## Mathematical Formulation

Given a query $`q`$ and a corpus $`C`$:

```math
P(y | q, C) = \sum_{d \in \text{Retrieve}(q, C)} P(y | q, d) P(d | q, C)
```
where $`d`$ is a retrieved document, $`P(d | q, C)`$ is the retriever score, and $`P(y | q, d)`$ is the generator probability.

## Python Example: RAG with Hugging Face Transformers

```python
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

# Load RAG model and tokenizer
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True)
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)

# Example query
question = "Who wrote the novel 1984?"
inputs = tokenizer(question, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
```

## Key Takeaways
- RAG combines retrieval and generation for factually accurate, up-to-date answers.
- Retrieval enables LLMs to access external knowledge beyond their training data.
- Hugging Face Transformers provides RAG models and tools for easy experimentation.

## References
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [Hugging Face RAG Documentation](https://huggingface.co/docs/transformers/model_doc/rag) 