# 01. Question Answering (QA) Systems

## Introduction

Question Answering (QA) systems are designed to automatically answer questions posed in natural language. They combine techniques from information retrieval, natural language understanding, and deep learning to extract, reason, and respond to queries.

## Types of QA Systems

### 1. Closed-Domain QA
- Answers questions within a specific domain (e.g., medical, legal).
- Relies on a curated knowledge base or documents.

### 2. Open-Domain QA
- Answers questions on any topic using large, unstructured corpora (e.g., Wikipedia).
- Combines retrieval and reading comprehension models.

## QA System Architectures

### 1. Retrieval-Based QA
- Retrieves relevant documents or passages from a corpus.
- Ranks and selects the best answer.

### 2. Extractive QA
- Finds and extracts the answer span from a given context.
- Example: SQuAD-style QA.

### 3. Generative QA
- Generates answers in natural language, possibly synthesizing information from multiple sources.
- Example: GPT-based QA.

## Mathematical Formulation

Given a question $`q`$ and context $`c`$, the goal is to find the answer $`a^*`$:

```math
a^* = \arg\max_a P(a | q, c)
```

For extractive QA, $`a`$ is a span in $`c`$; for generative QA, $`a`$ is a generated sequence.

## Python Example: Extractive QA with Hugging Face Transformers

```python
from transformers import pipeline
qa = pipeline("question-answering")
context = "The Transformer architecture was introduced in 2017 by Vaswani et al. It uses self-attention to process sequences in parallel."
question = "Who introduced the Transformer architecture?"
result = qa(question=question, context=context)
print("Answer:", result["answer"])
```

## Python Example: Generative QA with GPT-2

```python
from transformers import pipeline
generator = pipeline("text-generation", model="gpt2")
prompt = "Q: What is the capital of France?\nA:"
result = generator(prompt, max_length=20, num_return_sequences=1)
print(result[0]["generated_text"])
```

## Key Takeaways
- QA systems can be retrieval-based, extractive, or generative.
- Modern QA leverages deep learning and large language models.
- Hugging Face Transformers makes QA system development accessible.

## References
- [SQuAD: 100,000+ Questions for Machine Comprehension of Text (Rajpurkar et al., 2016)](https://arxiv.org/abs/1606.05250)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index) 