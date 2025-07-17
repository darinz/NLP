# 02. Parameter-Efficient Fine-Tuning (PEFT) Methods

## Introduction

Parameter-Efficient Fine-Tuning (PEFT) methods adapt large language models to new tasks by updating only a small subset of parameters. This enables rapid, memory-efficient adaptation, making it feasible to fine-tune very large models on modest hardware.

## Why PEFT?

- Full fine-tuning of large models is resource-intensive.
- PEFT reduces memory, compute, and storage requirements.
- Enables fast adaptation to many tasks/domains with minimal overhead.

## Common PEFT Methods

### 1. Adapters
- Small neural modules inserted between layers of the pretrained model.
- Only adapter parameters are updated during fine-tuning; the base model is frozen.

**Adapter math:**

```math
h' = h + \text{Adapter}(h)
```
where $`h`$ is the hidden state and $`\text{Adapter}(h)`$ is a small bottleneck network.

### 2. LoRA (Low-Rank Adaptation)
- Injects trainable low-rank matrices into the attention and/or feed-forward layers.
- Only the low-rank matrices are updated.

**LoRA math:**

```math
W' = W + \Delta W, \quad \Delta W = AB
```
where $`A \in \mathbb{R}^{d \times r}`$, $`B \in \mathbb{R}^{r \times k}`$, and $`r \ll d, k`$.

### 3. Prefix Tuning
- Prepends trainable "prefix" vectors to the input sequence at each layer.
- Only prefix parameters are updated.

**Prefix tuning math:**

```math
\text{Input} = [\text{Prefix}; \text{Original Input}]
```

## Python Example: LoRA with Hugging Face PEFT

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Configure LoRA
lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.1)
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

# Now, only LoRA parameters will be updated during training
```

## Key Takeaways
- PEFT methods enable efficient adaptation of large models with minimal resource usage.
- Adapters, LoRA, and prefix tuning are widely used in research and industry.
- Hugging Face PEFT library makes these methods accessible.

## References
- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [Parameter-Efficient Transfer Learning (PEFT) Survey](https://arxiv.org/abs/2303.04337)
- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft/index) 