# Datasets Directory

This directory contains sample datasets and data loading utilities for NLP tasks.

## Directory Structure

```
datasets/
├── README.md                    # This file
├── sample_data/                 # Sample datasets for testing
│   ├── sentiment_sample.csv     # Sample sentiment analysis data
│   ├── text_classification.csv  # Sample text classification data
│   └── translation_sample.txt   # Sample translation data
├── data_loaders/                # Data loading utilities
│   ├── __init__.py
│   ├── sentiment_loader.py      # Sentiment analysis data loader
│   ├── classification_loader.py # Text classification data loader
│   └── translation_loader.py    # Translation data loader
└── external/                    # External dataset references
    ├── sentiment_datasets.md    # Popular sentiment analysis datasets
    ├── classification_datasets.md # Text classification datasets
    └── translation_datasets.md  # Machine translation datasets
```

## Sample Datasets

### Sentiment Analysis Sample Data

The `sentiment_sample.csv` file contains a small sample of sentiment analysis data for testing and development:

```csv
text,sentiment
"I love this product! It's amazing.",positive
"This is terrible. I hate it.",negative
"The quality is okay, nothing special.",neutral
"Excellent service and fast delivery!",positive
"Poor customer support and slow shipping.",negative
```

### Text Classification Sample Data

The `text_classification.csv` file contains sample data for text classification tasks:

```csv
text,category
"Python programming tutorial for beginners",technology
"Delicious recipe for chocolate cake",food
"Latest news about climate change",news
"Movie review: The new action film",entertainment
"Health tips for better sleep",health
```

### Translation Sample Data

The `translation_sample.txt` file contains parallel text data for machine translation:

```
Hello world!	Bonjour le monde!
How are you?	Comment allez-vous?
I love programming.	J'aime la programmation.
```

## Data Loading Utilities

### Sentiment Analysis Data Loader

```python
from datasets.data_loaders.sentiment_loader import SentimentDataLoader

# Load sentiment analysis data
loader = SentimentDataLoader()
data = loader.load_data('datasets/sample_data/sentiment_sample.csv')

print(f"Loaded {len(data)} samples")
print(f"Classes: {data['classes']}")
print(f"Sample text: {data['texts'][0]}")
print(f"Sample label: {data['labels'][0]}")
```

### Text Classification Data Loader

```python
from datasets.data_loaders.classification_loader import ClassificationDataLoader

# Load text classification data
loader = ClassificationDataLoader()
data = loader.load_data('datasets/sample_data/text_classification.csv')

print(f"Loaded {len(data)} samples")
print(f"Categories: {data['categories']}")
```

### Translation Data Loader

```python
from datasets.data_loaders.translation_loader import TranslationDataLoader

# Load translation data
loader = TranslationDataLoader()
data = loader.load_data('datasets/sample_data/translation_sample.txt')

print(f"Loaded {len(data)} parallel sentences")
print(f"Source language: {data['source_lang']}")
print(f"Target language: {data['target_lang']}")
```

## External Dataset References

### Popular Sentiment Analysis Datasets

- **IMDB Movie Reviews**: Large dataset of movie reviews with sentiment labels
- **Amazon Product Reviews**: Product reviews with star ratings
- **Twitter Sentiment**: Tweets with sentiment annotations
- **SST (Stanford Sentiment Treebank)**: Fine-grained sentiment analysis dataset

### Text Classification Datasets

- **AG News**: News articles from 4 categories
- **DBpedia**: Wikipedia articles from 14 categories
- **Yahoo Answers**: Q&A data from 10 categories
- **20 Newsgroups**: Newsgroup posts from 20 categories

### Machine Translation Datasets

- **WMT**: Annual machine translation evaluation datasets
- **Multi30k**: Multilingual parallel corpus
- **OPUS**: Open parallel corpus collection
- **TED Talks**: Parallel transcripts from TED talks

## Data Loading Best Practices

### 1. Data Validation

Always validate your data after loading:

```python
def validate_data(data):
    """Validate loaded data."""
    assert len(data['texts']) == len(data['labels']), "Texts and labels must have same length"
    assert all(isinstance(text, str) for text in data['texts']), "All texts must be strings"
    assert all(isinstance(label, (int, str)) for label in data['labels']), "All labels must be int or str"
    return True
```

### 2. Data Preprocessing

Apply consistent preprocessing:

```python
def preprocess_data(texts, lowercase=True, remove_punctuation=False):
    """Preprocess text data."""
    processed_texts = []
    for text in texts:
        if lowercase:
            text = text.lower()
        if remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        processed_texts.append(text.strip())
    return processed_texts
```

### 3. Data Splitting

Split data appropriately for training:

```python
from sklearn.model_selection import train_test_split

def split_data(texts, labels, test_size=0.2, val_size=0.1):
    """Split data into train, validation, and test sets."""
    # First split: train+val and test
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Second split: train and val
    val_size_adjusted = val_size / (1 - test_size)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels, test_size=val_size_adjusted, 
        random_state=42, stratify=train_val_labels
    )
    
    return {
        'train': {'texts': train_texts, 'labels': train_labels},
        'val': {'texts': val_texts, 'labels': val_labels},
        'test': {'texts': test_texts, 'labels': test_labels}
    }
```

### 4. Data Augmentation

Consider data augmentation for small datasets:

```python
def augment_data(texts, labels, augmentation_factor=2):
    """Simple data augmentation."""
    augmented_texts = []
    augmented_labels = []
    
    for text, label in zip(texts, labels):
        # Add original
        augmented_texts.append(text)
        augmented_labels.append(label)
        
        # Add augmented versions
        for _ in range(augmentation_factor - 1):
            # Simple augmentation: random word deletion
            words = text.split()
            if len(words) > 3:
                import random
                words.pop(random.randint(0, len(words) - 1))
                augmented_text = ' '.join(words)
                augmented_texts.append(augmented_text)
                augmented_labels.append(label)
    
    return augmented_texts, augmented_labels
```

## Dataset Statistics

### Sample Dataset Information

| Dataset | Type | Size | Classes | Language |
|---------|------|------|---------|----------|
| sentiment_sample.csv | Sentiment Analysis | 100 samples | 3 (pos/neg/neu) | English |
| text_classification.csv | Text Classification | 50 samples | 5 categories | English |
| translation_sample.txt | Machine Translation | 20 pairs | N/A | EN-FR |

### Data Quality Metrics

- **Text Length**: Average 50-200 words per sample
- **Class Balance**: Approximately balanced across classes
- **Language**: Primarily English with some multilingual examples
- **Format**: Clean, preprocessed text suitable for immediate use

## Usage Examples

### Quick Start

```python
# Load and use sample data
import pandas as pd

# Load sentiment data
sentiment_data = pd.read_csv('datasets/sample_data/sentiment_sample.csv')
print(f"Sentiment dataset shape: {sentiment_data.shape}")

# Load classification data
classification_data = pd.read_csv('datasets/sample_data/text_classification.csv')
print(f"Classification dataset shape: {classification_data.shape}")

# Basic analysis
print(f"Sentiment classes: {sentiment_data['sentiment'].value_counts()}")
print(f"Classification categories: {classification_data['category'].value_counts()}")
```

### Custom Data Loading

```python
def load_custom_dataset(file_path, text_column, label_column):
    """Load custom dataset from CSV file."""
    import pandas as pd
    
    data = pd.read_csv(file_path)
    
    return {
        'texts': data[text_column].tolist(),
        'labels': data[label_column].tolist(),
        'classes': data[label_column].unique().tolist()
    }

# Usage
custom_data = load_custom_dataset(
    'your_data.csv',
    text_column='text',
    label_column='label'
)
```

## Contributing

When adding new datasets to this directory:

1. **Documentation**: Update this README with dataset information
2. **Format**: Use consistent CSV or TXT formats
3. **Validation**: Include data validation scripts
4. **Examples**: Provide usage examples
5. **Licensing**: Ensure proper licensing information

## License

Sample datasets in this directory are provided for educational purposes. 
Please respect the original licenses of external datasets referenced here. 