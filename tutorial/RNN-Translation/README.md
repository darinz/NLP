# NLP: Recurrent Neural Network (RNN) Seq2Seq Machine Translation with Attention

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/darinz/NLP-Tutorials/blob/main/RNN-Translation/rnn_translation.ipynb)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)

In this tutorial, we guide you through building and training your own Recurrent Neural Network (RNN) for Machine Translation with Attention, starting from scratch. We break down the process into bite-sized steps, focusing on the essential building blocks of Natural Language Processing and data preprocessing. By crafting each component yourself, you'll gain a deep understanding of how RNNs handle text data and perform machine translation.

## What You'll Learn

- **Sequence-to-Sequence Architecture**: Building encoder-decoder models
- **Attention Mechanisms**: Implementing attention for better translation quality
- **Text Preprocessing**: Tokenization, vocabulary building, and data preparation
- **Training Techniques**: Loss functions, optimization, and evaluation metrics
- **Machine Translation Pipeline**: End-to-end translation system development

## Translation Task

This tutorial focuses on **English to French translation** using a custom dataset. You'll learn to:
- Preprocess parallel text data
- Build vocabulary from source and target languages
- Implement attention mechanisms for improved translation quality
- Train and evaluate translation models

## Prerequisites

Before starting this tutorial, it is recommended that you have:

- [PyTorch](https://pytorch.org/) installed or use [Google Colab](https://colab.research.google.com/?utm_source=scs-index)
- Basic understanding of [Python programming language](https://www.python.org/doc/)
- Familiarity with [PyTorch Tensors](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- Completion of the [RNN Name Classification](../RNN-Classification/README.md) tutorial (recommended)

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
```

## Learning Resources

### Sequence-to-Sequence and Attention

It's helpful to understand Seq2Seq networks and attention mechanisms:

- **[Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)** - Original Seq2Seq paper
- **[Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)** - Google's Seq2Seq implementation
- **[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)** - Attention mechanism introduction
- **[A Neural Conversational Model](https://arxiv.org/abs/1506.05869)** - Advanced Seq2Seq applications

### Additional Resources

- [PyTorch Tutorials](https://pytorch.org/tutorials/) - Official PyTorch documentation
- [Colab PyTorch Tutorial](https://pytorch.org/tutorials/beginner/colab) - Tips for running PyTorch in Google Colab

## Project Structure

```
RNN-Translation/
├── data/
│   └── eng-fra.txt          # English-French parallel corpus
├── rnn_translation.ipynb    # Main tutorial notebook
├── environment.yml          # Conda environment file
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Expected Outcomes

After completing this tutorial, you will be able to:

1. **Build Seq2Seq models** with encoder-decoder architecture
2. **Implement attention mechanisms** for improved translation quality
3. **Preprocess parallel text data** for machine translation
4. **Train and evaluate** translation models
5. **Apply Seq2Seq models** to other sequence-to-sequence tasks

## Key Concepts Covered

### Architecture Components
- **Encoder**: Processes input sequence and creates context vector
- **Decoder**: Generates output sequence using context and attention
- **Attention Mechanism**: Helps decoder focus on relevant parts of input

### Training Process
- **Teacher Forcing**: Training technique for faster convergence
- **Beam Search**: Inference technique for better translation quality
- **BLEU Score**: Evaluation metric for translation quality

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

**Previous Tutorial**: [RNN Name Classification](../RNN-Classification/README.md) ← Learn basic RNN concepts

**Next Tutorial**: [Transformer Machine Translation](../Transformer-Translation/README.md) → Master modern transformer architecture
