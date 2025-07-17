# 01. Evaluation Metrics

## Introduction

Evaluation metrics are essential for assessing the performance of NLP models. They provide quantitative measures to compare models, diagnose weaknesses, and guide improvements.

## Why Evaluation Metrics?

- Enable objective comparison of models.
- Help identify strengths and weaknesses.
- Guide model selection and tuning.

## Common Evaluation Metrics in NLP

### 1. Accuracy
Proportion of correct predictions:

```math
\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total predictions}}
```

### 2. Precision, Recall, F1 Score
Used for classification, especially with imbalanced classes.

- **Precision:** Fraction of predicted positives that are correct.

```math
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
```

- **Recall:** Fraction of actual positives that are correctly predicted.

```math
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
```

- **F1 Score:** Harmonic mean of precision and recall.

```math
\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
```

### 3. BLEU (Bilingual Evaluation Understudy)
Measures similarity between machine-generated and reference translations (higher is better).

```math
\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^N w_n \log p_n\right)
```
where $`p_n`$ is the precision for n-grams, $`w_n`$ are weights, and BP is the brevity penalty.

### 4. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
Measures overlap between generated and reference summaries (higher is better).
- **ROUGE-N:** n-gram overlap
- **ROUGE-L:** Longest common subsequence

### 5. Perplexity
Used for language models; lower is better.

```math
\text{Perplexity} = \exp\left(-\frac{1}{N} \sum_{i=1}^N \log P(w_i)\right)
```

## Python Example: Classification Metrics with scikit-learn

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_true = [1, 0, 1, 1, 0]
y_pred = [1, 0, 0, 1, 1]

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))
```

## Python Example: BLEU and ROUGE with NLTK

```python
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

reference = [["the", "cat", "is", "on", "the", "mat"]]
candidate = ["the", "cat", "sat", "on", "the", "mat"]
print("BLEU:", sentence_bleu(reference, candidate))

scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
score = scorer.score("The cat is on the mat.", "The cat sat on the mat.")
print("ROUGE:", score)
```

## Key Takeaways
- Choose metrics appropriate for your task (classification, generation, etc.).
- Use multiple metrics for a comprehensive evaluation.
- Python libraries like scikit-learn and NLTK make metric computation easy.

## References
- [scikit-learn Metrics Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [NLTK BLEU Documentation](https://www.nltk.org/_modules/nltk/translate/bleu_score.html)
- [ROUGE Score Documentation](https://pypi.org/project/rouge-score/) 