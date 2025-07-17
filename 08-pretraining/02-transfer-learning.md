# 02. Transfer Learning

## Introduction

Transfer learning is a paradigm where a model trained on one task (usually with large, generic data) is adapted to a different, often more specific, task. In NLP, transfer learning with pretrained language models has become the standard approach for achieving state-of-the-art results.

## Why Transfer Learning?

- Labeled data for specific NLP tasks is often scarce and expensive to obtain.
- Pretrained models capture general language knowledge from large corpora.
- Fine-tuning adapts this knowledge to new tasks with minimal data.

## Transfer Learning Workflow

1. **Pretrain** a model on a large, generic task (e.g., language modeling).
2. **Fine-tune** the pretrained model on a specific downstream task (e.g., sentiment analysis, NER).

## Mathematical Perspective

Let $`\mathcal{D}_S`$ be the source domain (pretraining data) and $`\mathcal{D}_T`$ the target domain (task data). The model parameters $`\theta`$ are first optimized for the source task:

```math
\theta^* = \arg\min_{\theta} L_S(\theta; \mathcal{D}_S)
```

Then, $`\theta^*`$ is used as initialization for the target task:

```math
\theta_T^* = \arg\min_{\theta} L_T(\theta; \mathcal{D}_T)
```

## Types of Transfer Learning in NLP

- **Feature-based:** Use pretrained embeddings (e.g., Word2Vec, GloVe) as features for downstream models.
- **Fine-tuning:** Update all or part of the pretrained model's parameters on the new task (e.g., BERT, GPT).
- **Frozen representations:** Use the pretrained model as a fixed feature extractor.

## Python Example: Fine-Tuning BERT for Text Classification

```python
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# Load pretrained BERT and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name)

# Example data
texts = ["I love NLP!", "Transformers are amazing."]
labels = [1, 1]  # Positive sentiment
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="tf")

# Fine-tune (dummy example)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), loss=model.compute_loss)
model.fit(inputs.data, tf.convert_to_tensor(labels), epochs=1)
```

## Key Takeaways
- Transfer learning leverages knowledge from large-scale pretraining for new tasks.
- Fine-tuning pretrained models is the dominant approach in NLP.
- Hugging Face Transformers makes transfer learning accessible and efficient.

## References
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)
- [A Primer in BERTology: What We Know About How BERT Works (Rogers et al., 2020)](https://arxiv.org/abs/2002.12327)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index) 