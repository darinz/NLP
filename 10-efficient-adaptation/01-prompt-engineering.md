# 01. Prompt Engineering

## Introduction

Prompt engineering is the practice of designing and optimizing input prompts to elicit desired behaviors from large language models (LLMs). It is a key technique for efficiently adapting models to new tasks without retraining.

## Why Prompt Engineering?

- LLMs can perform a wide range of tasks based on how they are prompted.
- Carefully crafted prompts can improve accuracy, reliability, and task alignment.
- No need for model fine-tuning or additional training data.

## Types of Prompting

### 1. Zero-Shot Prompting
Ask the model to perform a task without any examples:

```
Prompt: Translate to French: "How are you?"
Output: "Comment ça va ?"
```

### 2. Few-Shot Prompting
Provide a few input-output examples in the prompt:

```
Prompt:
English: "Good morning" → French: "Bonjour"
English: "Thank you" → French: "Merci"
English: "How are you?" → French:
Output: "Comment ça va ?"
```

### 3. Chain-of-Thought Prompting
Encourage the model to reason step by step:

```
Prompt: If there are 3 cars and each car has 4 wheels, how many wheels are there in total?
Let's think step by step.
Output: There are 3 cars. Each car has 4 wheels. 3 × 4 = 12. So, there are 12 wheels in total.
```

## Prompt Engineering Strategies

- **Instruction clarity:** Be explicit about the task.
- **Contextual cues:** Provide relevant context or constraints.
- **Formatting:** Use lists, bullet points, or structured templates.
- **Demonstrations:** Show examples of desired behavior.
- **Step-by-step reasoning:** Encourage intermediate steps for complex tasks.

## Python Example: Prompting with Hugging Face Transformers

```python
from transformers import pipeline

# Zero-shot prompt
generator = pipeline("text-generation", model="gpt2")
prompt = "Translate to French: How are you?"
result = generator(prompt, max_length=30, num_return_sequences=1)
print(result[0]["generated_text"])

# Few-shot prompt
few_shot_prompt = (
    "English: Good morning → French: Bonjour\n"
    "English: Thank you → French: Merci\n"
    "English: How are you? → French:"
)
result = generator(few_shot_prompt, max_length=50, num_return_sequences=1)
print(result[0]["generated_text"])
```

## Key Takeaways
- Prompt engineering enables rapid adaptation of LLMs to new tasks.
- Zero-shot, few-shot, and chain-of-thought prompting are powerful techniques.
- No retraining is required—just change the prompt!

## References
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [Language Models are Few-Shot Learners (Brown et al., 2020)](https://arxiv.org/abs/2005.14165) 