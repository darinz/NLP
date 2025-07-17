# 02. Supervised Fine-Tuning (SFT)

## Introduction

Supervised Fine-Tuning (SFT) is a post-training technique where a pretrained language model is further trained on a labeled dataset of human demonstrations. SFT is often the first step in aligning large language models with specific tasks or user preferences.

## Why SFT?

- Pretrained models are generic and may not perform optimally on specific tasks.
- SFT adapts the model to follow instructions, generate helpful responses, or perform domain-specific tasks.

## SFT Pipeline Overview

1. **Collect Labeled Data:**
   - Gather a dataset of input-output pairs (e.g., prompts and ideal completions).
2. **Fine-Tune the Model:**
   - Train the model to minimize the loss between its outputs and the human-provided targets.

## Mathematical Formulation

Given a dataset $`\mathcal{D} = \{(x^{(i)}, y^{(i)})\}`$ of input-output pairs, the SFT objective is:

```math
\theta^* = \arg\min_{\theta} \frac{1}{N} \sum_{i=1}^N L(f_\theta(x^{(i)}), y^{(i)})
```
where $`L`$ is a loss function (e.g., cross-entropy), $`f_\theta`$ is the model, and $`N`$ is the number of examples.

## Python Example: SFT with Hugging Face Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch

# Example data (toy)
texts = ["Translate English to French: Hello, how are you?", "Summarize: The cat sat on the mat."]
labels = ["Bonjour, comment ça va?", "Cat sat on mat."]

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
with tokenizer.as_target_tokenizer():
    targets = tokenizer(labels, padding=True, truncation=True, return_tensors="pt")

# Dummy dataset
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs["input_ids"]
        self.targets = targets["input_ids"]
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return {"input_ids": self.inputs[idx], "labels": self.targets[idx]}

dataset = DummyDataset(inputs, targets)

# Training
training_args = TrainingArguments(output_dir="./results", num_train_epochs=1, per_device_train_batch_size=2)
trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
```

## Key Takeaways
- SFT adapts pretrained models to specific tasks using labeled data.
- It is the first step in RLHF pipelines and instruction-following models.
- Hugging Face Transformers makes SFT accessible for many tasks.

## References
- [Training language models to follow instructions with human feedback (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index) 