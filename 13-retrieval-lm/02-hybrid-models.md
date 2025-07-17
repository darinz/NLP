# 02. Hybrid Models

## Introduction

Hybrid models in NLP combine different architectures or techniques—such as retrieval, generation, and symbolic reasoning—to leverage their complementary strengths. These models are especially effective for complex tasks requiring both factual accuracy and reasoning.

## Why Hybrid Models?

- Pure generation models may hallucinate or lack up-to-date knowledge.
- Retrieval models are factual but may lack fluency or reasoning.
- Hybrid models combine the best of both worlds: factual grounding and fluent, context-aware generation.

## Types of Hybrid Models

### 1. Retrieval-Augmented Generation (RAG)
- Combines a retriever and a generator (see previous guide).

### 2. Fusion-in-Decoder (FiD)
- Retrieves multiple passages and fuses them in the decoder for answer generation.

**FiD math:**

```math
P(y | q, D) = \text{Decoder}(q, d_1, d_2, ..., d_k)
```
where $`D = \{d_1, ..., d_k\}`$ are retrieved documents.

### 3. Symbolic + Neural Models
- Use symbolic reasoning (e.g., rules, knowledge graphs) alongside neural models for tasks like QA and reasoning.

### 4. Multi-Stage Pipelines
- Use a retriever to select documents, a reader to extract answers, and a generator to synthesize responses.

## Python Example: Fusion-in-Decoder (FiD) with T5 (Pseudo-code)

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Example: concatenate question with multiple retrieved passages
question = "Who discovered penicillin?"
passages = [
    "Penicillin was discovered by Alexander Fleming in 1928.",
    "Fleming's discovery changed the course of medicine.",
]
inputs = [f"question: {question} context: {p}" for p in passages]
input_ids = tokenizer(inputs, return_tensors="pt", padding=True).input_ids
# Stack passage representations for FiD (simplified)
outputs = model.generate(input_ids)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
```

## Key Takeaways
- Hybrid models combine retrieval, generation, and reasoning for robust NLP systems.
- Fusion-in-Decoder and symbolic-neural hybrids are powerful approaches.
- Python libraries like Hugging Face Transformers enable hybrid modeling.

## References
- [Fusion-in-Decoder: Efficient Retrieval-Augmented Text Generation (Izacard & Grave, 2021)](https://arxiv.org/abs/2007.01282)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401) 