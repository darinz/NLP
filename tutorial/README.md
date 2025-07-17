# NLP-Tutorials

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](CONTRIBUTING.md)

A curated collection of comprehensive NLP tutorials designed to help you quickly gain hands-on experience with real-world NLP use cases. Each tutorial is self-contained with detailed explanations, code examples, and practical implementations.

## Tutorials

### 1. [RNN Name Classification](RNN-Classification/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/darinz/NLP-Tutorials/blob/main/RNN-Classification/nlp_rnn_name_classification.ipynb)

Learn to build and train Recurrent Neural Networks (RNNs) from scratch for name classification across 18 different languages. This tutorial covers:
- Text preprocessing and character-level encoding
- RNN architecture implementation
- Training and evaluation techniques
- Multi-language name classification

**Technologies:** PyTorch, NumPy, Jupyter

### 2. [RNN Machine Translation](RNN-Translation/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/darinz/NLP-Tutorials/blob/main/RNN-Translation/rnn_translation.ipynb)

Build a sequence-to-sequence RNN model with attention mechanism for English-French translation. This tutorial covers:
- Encoder-decoder architecture
- Attention mechanism implementation
- Sequence-to-sequence learning
- Machine translation pipeline

**Technologies:** PyTorch, NumPy, Jupyter

### 3. [Transformer Machine Translation](Transformer-Translation/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/darinz/NLP-Tutorials/blob/main/Transformer-Translation/transformer_translation.ipynb)

Implement the Transformer architecture for German-English translation using the Multi30k dataset. This tutorial covers:
- Transformer architecture components
- Multi-head attention mechanisms
- Positional encoding
- Modern NLP best practices

**Technologies:** PyTorch, TorchText, SpaCy, NumPy, Jupyter

## Prerequisites

- Python 3.8 or higher
- Basic understanding of Python programming
- Familiarity with PyTorch tensors and neural networks
- Jupyter Notebook or Google Colab

## Quick Start

### Option 1: Google Colab (Recommended)
Each tutorial includes a "Open in Colab" button for instant access without local setup.

### Option 2: Local Setup
1. Clone the repository:
```bash
git clone https://github.com/darinz/NLP-Tutorials.git
cd NLP-Tutorials
```

2. Choose a tutorial and set up its environment:
```bash
cd RNN-Classification  # or RNN-Translation or Transformer-Translation
conda env create -f environment.yml
conda activate nlp
```

3. Launch Jupyter:
```bash
jupyter lab
```

## Learning Path

We recommend following the tutorials in this order for optimal learning:

1. **RNN Name Classification** - Start with basic RNN concepts and text processing
2. **RNN Machine Translation** - Learn sequence-to-sequence models with attention
3. **Transformer Machine Translation** - Master modern transformer architecture

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- The NLP research community for foundational papers and datasets
- Contributors and users who provide feedback and improvements

## Repository Stats

![GitHub stars](https://img.shields.io/github/stars/darinz/NLP-Tutorials?style=social)
![GitHub forks](https://img.shields.io/github/forks/darinz/NLP-Tutorials?style=social)
![GitHub issues](https://img.shields.io/github/issues/darinz/NLP-Tutorials)
![GitHub pull requests](https://img.shields.io/github/issues-pr/darinz/NLP-Tutorials)

---

**Star this repository if you find it helpful!**
