# NLP: Transformer Seq2Seq Machine Translation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/darinz/NLP-Tutorials/blob/main/Transformer-Translation/transformer_translation.ipynb)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)

In this tutorial, you will learn to build and train a state-of-the-art machine translation model using the Transformer architecture. We'll implement the complete pipeline from data preprocessing to model training, utilizing the Multi30k dataset for German-to-English translation.

## What You'll Learn

- **Transformer Architecture**: Understanding the "Attention is All You Need" model
- **Multi-Head Attention**: Implementing self-attention and cross-attention mechanisms
- **Positional Encoding**: Adding positional information to sequence data
- **Modern NLP Pipeline**: Using TorchText and SpaCy for data processing
- **Training Techniques**: Optimizing transformer models for translation tasks

## Translation Task

This tutorial focuses on **German to English translation** using the [Multi30k](http://www.statmt.org/wmt16/multimodal-task.html#task1) dataset. You'll learn to:
- Preprocess multilingual text data using modern NLP tools
- Build transformer models from scratch
- Implement attention mechanisms and positional encoding
- Train and evaluate translation models with state-of-the-art techniques

## Prerequisites

Before starting this tutorial, it is recommended that you have:

- [PyTorch](https://pytorch.org/) installed or use [Google Colab](https://colab.research.google.com/?utm_source=scs-index)
- Basic understanding of [Python programming language](https://www.python.org/doc/)
- Familiarity with [PyTorch Tensors](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- Completion of the [RNN Machine Translation](../RNN-Translation/README.md) tutorial (recommended)

## Quick Start

### Option 1: Google Colab (Recommended)
Click the "Open in Colab" button above to run this tutorial instantly without any setup.

### Option 2: Local Environment Setup

#### Conda Environment Setup

Use the following commands to manage your conda environment:

```bash
# Create a new conda environment
conda env create -f environment.yml

# Activate the environment
conda activate nlp

# Remove the environment (if needed)
conda remove --name nlp --all

# Update the environment when new libraries are added
conda env update -f environment.yml --prune
```

#### Manual Installation

If you prefer manual installation:

```bash
# Install PyTorch (visit pytorch.org for your specific setup)
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt

# Install additional required packages
pip install -U portalocker
pip install -U torchtext
pip install -U spacy

# Download SpaCy language models
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

## Learning Resources

### Transformer Architecture

It's essential to understand the Transformer architecture and attention mechanisms:

- **[Attention is All You Need](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)** - Original Transformer paper
- **[The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding)** - Detailed implementation guide

### Additional Resources

- [PyTorch Tutorials](https://pytorch.org/tutorials/) - Official PyTorch documentation
- [Colab PyTorch Tutorial](https://pytorch.org/tutorials/beginner/colab) - Tips for running PyTorch in Google Colab
- [TorchText Documentation](https://pytorch.org/text/stable/index.html) - Text processing utilities
- [SpaCy Documentation](https://spacy.io/usage) - Industrial-strength NLP library

## Project Structure

```
Transformer-Translation/
├── transformer_translation.ipynb  # Main tutorial notebook
├── environment.yml                # Conda environment file
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```

## Expected Outcomes

After completing this tutorial, you will be able to:

1. **Implement Transformer architecture** from scratch using PyTorch
2. **Build multi-head attention mechanisms** for sequence processing
3. **Preprocess multilingual data** using modern NLP tools
4. **Train transformer models** for machine translation
5. **Apply transformer architecture** to other NLP tasks

## Key Concepts Covered

### Transformer Components
- **Multi-Head Attention**: Parallel attention mechanisms for different representation subspaces
- **Positional Encoding**: Adding positional information to input embeddings
- **Feed-Forward Networks**: Position-wise fully connected layers
- **Layer Normalization**: Stabilizing training and improving convergence

### Modern NLP Tools
- **TorchText**: PyTorch's text processing utilities
- **SpaCy**: Industrial-strength NLP library for tokenization
- **Multi30k Dataset**: Multilingual parallel corpus for training

### Training Process
- **Learning Rate Scheduling**: Warmup and decay strategies
- **Label Smoothing**: Regularization technique for better generalization
- **Beam Search**: Inference technique for high-quality translations

## Contributing

Found an issue or have a suggestion? Please feel free to:
- Open an issue
- Submit a pull request
- Share your results and improvements

## License

This tutorial is part of the [NLP-Tutorials](../README.md) collection and is licensed under the MIT License.

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- The NLP research community for foundational papers and datasets
- Contributors who provide feedback and improvements

---

**Previous Tutorial**: [RNN Machine Translation](../RNN-Translation/README.md) ← Learn sequence-to-sequence models with attention

**Back to Overview**: [NLP-Tutorials](../README.md) ← Complete tutorial collection
