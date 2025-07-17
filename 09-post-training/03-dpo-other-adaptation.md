# 03. DPO and Other Adaptation Methods

## Introduction

Direct Preference Optimization (DPO) and other adaptation methods are post-training techniques that further align language models with human preferences or specific objectives. These methods are alternatives or complements to RLHF and SFT, aiming for efficient and effective adaptation.

## Direct Preference Optimization (DPO)

DPO is a recent method that directly optimizes model parameters to increase the likelihood of preferred outputs over less-preferred ones, without requiring reinforcement learning.

### DPO Objective
Given a pair of outputs $`(y^+, y^-)`$ for the same prompt, where $`y^+`$ is preferred over $`y^-`$, the DPO loss is:

```math
L_{\text{DPO}}(\theta) = -\log \frac{\exp(\beta \log \pi_\theta(y^+))}{\exp(\beta \log \pi_\theta(y^+)) + \exp(\beta \log \pi_\theta(y^-))}
```
where $`\pi_\theta`$ is the model, and $`\beta`$ is a temperature parameter.

- DPO can be seen as a form of pairwise preference learning.
- It avoids the complexity of reward modeling and RL.

## Other Adaptation Methods

### 1. Instruction Tuning
- Fine-tune models on datasets of instructions and responses (e.g., FLAN, Alpaca).
- Improves model's ability to follow user instructions.

### 2. Parameter-Efficient Fine-Tuning (PEFT)
- Adapt only a small subset of model parameters (e.g., adapters, LoRA, prefix tuning).
- Reduces compute and memory requirements.

### 3. Mixture-of-Experts (MoE)
- Use multiple expert subnetworks, routing each input to the most relevant experts.
- Increases model capacity without proportional increase in compute.

## Python Example: DPO Loss (PyTorch)

```python
import torch
import torch.nn.functional as F

def dpo_loss(logp_pos, logp_neg, beta=1.0):
    # logp_pos, logp_neg: log-probabilities of positive/negative samples
    logits = beta * torch.stack([logp_pos, logp_neg], dim=-1)
    labels = torch.zeros(logits.size(0), dtype=torch.long)  # 0 = positive preferred
    return F.cross_entropy(logits, labels)

# Example usage
logp_pos = torch.tensor([2.0, 1.5])  # log-probabilities for preferred outputs
logp_neg = torch.tensor([0.5, 1.0])  # log-probabilities for less-preferred outputs
loss = dpo_loss(logp_pos, logp_neg, beta=2.0)
print("DPO loss:", loss.item())
```

## Key Takeaways
- DPO directly optimizes for human preferences using pairwise comparisons.
- Instruction tuning and PEFT are efficient alternatives for model adaptation.
- These methods enable rapid, scalable, and targeted adaptation of large language models.

## References
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)
- [Parameter-Efficient Transfer Learning (PEFT) Survey](https://arxiv.org/abs/2303.04337)
- [Instruction Tuning with FLAN](https://arxiv.org/abs/2210.11416) 