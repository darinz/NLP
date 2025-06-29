# NLP: RNN Name Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/darinz/NLP-Tutorials/blob/main/RNN-Classification/nlp_rnn_name_classification.ipynb)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)

In this tutorial, we guide you through building and training your own Recurrent Neural Network (RNN) from scratch for name classification across **18 different languages**. We break down the process into bite-sized steps, focusing on the essential building blocks of Natural Language Processing data preprocessing. By crafting each component yourself, you'll gain a deep understanding of how RNNs handle text data and make confident predictions on names.

## What You'll Learn

- **Text Preprocessing**: Character-level encoding and data preparation
- **RNN Architecture**: Building recurrent neural networks from scratch
- **Training Techniques**: Loss functions, optimization, and evaluation
- **Multi-language Classification**: Handling names from 18 different languages
- **Practical Implementation**: Real-world NLP pipeline development

## Supported Languages

The tutorial includes name datasets for the following languages:
- Arabic, Chinese, Czech, Dutch, English, French, German, Greek
- Irish, Italian, Japanese, Korean, Polish, Portuguese
- Russian, Scottish, Spanish, Vietnamese

## Prerequisites

Before starting this tutorial, it is recommended that you have:

- [PyTorch](https://pytorch.org/) installed or use [Google Colab](https://colab.research.google.com/?utm_source=scs-index)
- Basic understanding of [Python programming language](https://www.python.org/doc/)
- Familiarity with [PyTorch Tensors](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)

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

### RNN Fundamentals

It's helpful to understand RNNs and how they work before diving in:

- **[The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)** - Real-world examples and applications
- **[Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)** - Comprehensive guide to LSTMs and RNNs in general

### Additional Resources

- [PyTorch Tutorials](https://pytorch.org/tutorials/) - Official PyTorch documentation
- [Colab PyTorch Tutorial](https://pytorch.org/tutorials/beginner/colab) - Tips for running PyTorch in Google Colab

## Project Structure

```
RNN-Classification/
├── data/
│   └── names/
│       ├── Arabic.txt
│       ├── Chinese.txt
│       ├── Czech.txt
│       ├── Dutch.txt
│       ├── English.txt
│       ├── French.txt
│       ├── German.txt
│       ├── Greek.txt
│       ├── Irish.txt
│       ├── Italian.txt
│       ├── Japanese.txt
│       ├── Korean.txt
│       ├── Polish.txt
│       ├── Portuguese.txt
│       ├── Russian.txt
│       ├── Scottish.txt
│       ├── Spanish.txt
│       └── Vietnamese.txt
├── nlp_rnn_name_classification.ipynb
├── environment.yml
├── requirements.txt
└── README.md
```

## Expected Outcomes

After completing this tutorial, you will be able to:

1. **Preprocess text data** for RNN input
2. **Build RNN models** from scratch using PyTorch
3. **Train and evaluate** neural networks for classification tasks
4. **Handle multi-language** text classification
5. **Apply RNNs** to real-world NLP problems

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

**Next Tutorial**: [RNN Machine Translation](../RNN-Translation/README.md) → Learn sequence-to-sequence models with attention mechanisms

```bibtex
@article{karpathy2015unreasonable,
  title={The Unreasonable Effectiveness of Recurrent Neural Networks},
  author={Karpathy, Andrej},
  journal={arXiv preprint arXiv:1506.02078},
  year={2015}
}

@article{colah2015understanding,
  title={Understanding LSTM Networks},
  author={Colah, Christopher},
  journal={arXiv preprint arXiv:1503.04069},
  year={2015}
}
```
