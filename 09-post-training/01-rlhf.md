# 01. Reinforcement Learning from Human Feedback (RLHF)

## Introduction

Reinforcement Learning from Human Feedback (RLHF) is a post-training technique that aligns language models with human preferences. It combines supervised learning and reinforcement learning to adapt models for specific tasks, improving safety, helpfulness, and user satisfaction.

## Why RLHF?

- Pretrained models may generate outputs that are unhelpful, unsafe, or misaligned with user intent.
- RLHF uses human feedback to guide the model toward preferred behaviors.

## RLHF Pipeline Overview

1. **Supervised Fine-Tuning (SFT):**
   - Fine-tune the model on a dataset of human demonstrations.
2. **Reward Model Training:**
   - Train a reward model to predict human preferences between pairs of model outputs.
3. **Reinforcement Learning (RL):**
   - Use the reward model to optimize the language model via RL (e.g., Proximal Policy Optimization, PPO).

## Mathematical Formulation

### Reward Model
Given a pair of outputs $`(y^A, y^B)`$ for the same prompt, the reward model $`r_\phi`$ is trained so that:

```math
P(y^A \succ y^B) = \frac{\exp(r_\phi(y^A))}{\exp(r_\phi(y^A)) + \exp(r_\phi(y^B))}
```

### RL Objective
The language model (policy $`\pi_\theta`$) is updated to maximize expected reward:

```math
J(\theta) = \mathbb{E}_{y \sim \pi_\theta}[r_\phi(y)]
```

Often, a KL penalty is added to keep the new policy close to the original (preference for conservative updates):

```math
J_{KL}(\theta) = \mathbb{E}_{y \sim \pi_\theta}[r_\phi(y) - \beta \log \frac{\pi_\theta(y)}{\pi_{\text{ref}}(y)}]
```
where $`\beta`$ controls the strength of the penalty.

## Python Example: Reward Model Training (Pseudo-code)

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Dummy reward model
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model  # e.g., a transformer
        self.scorer = nn.Linear(base_model.hidden_size, 1)
    def forward(self, x):
        h = self.base(x)[0][:, -1, :]  # last hidden state
        return self.scorer(h).squeeze(-1)

# Training loop (pairwise preference)
model = RewardModel(base_model)
optimizer = optim.Adam(model.parameters())
for batch in dataloader:
    yA, yB, label = batch  # label: 1 if yA preferred, 0 if yB preferred
    rA = model(yA)
    rB = model(yB)
    loss = -torch.log(torch.sigmoid(rA - rB)) * label - torch.log(torch.sigmoid(rB - rA)) * (1 - label)
    loss = loss.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Key Takeaways
- RLHF aligns models with human values and preferences.
- Combines supervised learning, reward modeling, and RL.
- Widely used in modern conversational AI (e.g., ChatGPT).

## References
- [Training language models to follow instructions with human feedback (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- [Deep Reinforcement Learning from Human Preferences (Christiano et al., 2017)](https://arxiv.org/abs/1706.03741)
- [OpenAI Blog: RLHF](https://openai.com/research/learning-from-human-feedback) 