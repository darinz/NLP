# 01. Language Model Pretraining

## Introduction

Language model pretraining is a foundational technique in modern NLP. Models are first trained on large unlabeled corpora to learn general language patterns, then fine-tuned for specific tasks. This approach has led to breakthroughs in performance across many NLP benchmarks.

## What is Language Model Pretraining?

- **Pretraining:** Train a model on a generic language modeling objective using massive text data (e.g., Wikipedia, books).
- **Fine-tuning:** Adapt the pretrained model to a specific downstream task (e.g., sentiment analysis, QA) with labeled data.

## Pretraining Objectives

### 1. Causal Language Modeling (CLM)
Predict the next word given previous words:

```math
P(w_t | w_1, w_2, ..., w_{t-1})
```
Used in models like GPT.

### 2. Masked Language Modeling (MLM)
Predict masked words in a sentence:

```math
P(w_k | w_1, ..., w_{k-1}, [MASK], w_{k+1}, ..., w_T)
```
Used in models like BERT.

### 3. Next Sentence Prediction (NSP)
Predict if one sentence follows another (used in BERT):

- Input: Sentence A, Sentence B
- Output: Is B the next sentence after A?

## Why Pretrain?

- Leverages vast amounts of unlabeled data.
- Learns general language representations transferable to many tasks.
- Reduces labeled data requirements for downstream tasks.

## Python Example: Pretraining a Simple Language Model (with Hugging Face Transformers)

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

# Load pretrained BERT for masked language modeling
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
result = fill_mask("NLP is [MASK] for modern AI.")
for r in result:
    print(f"Prediction: {r['sequence']} (score: {r['score']:.4f})")
```

## Key Takeaways
- Pretraining on large corpora enables models to learn general language features.
- Causal and masked language modeling are common pretraining objectives.
- Pretrained models can be fine-tuned for a wide range of NLP tasks.

## References
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)
- [GPT: Improving Language Understanding by Generative Pre-Training (Radford et al., 2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index) 