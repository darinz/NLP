# Text Preprocessing Guide

A comprehensive guide to text preprocessing techniques for Natural Language Processing (NLP) tasks.

## Table of Contents
1. [Introduction](#introduction)
2. [Text Cleaning](#text-cleaning)
3. [Text Normalization](#text-normalization)
4. [Tokenization](#tokenization)
5. [Stop Word Removal](#stop-word-removal)
6. [Stemming and Lemmatization](#stemming-and-lemmatization)
7. [Handling Special Cases](#handling-special-cases)
8. [Complete Pipeline](#complete-pipeline)
9. [Best Practices](#best-practices)

## Introduction

Text preprocessing is the foundation of any NLP pipeline. Raw text data often contains noise, inconsistencies, and formatting issues that need to be addressed before analysis.

### Why Preprocessing Matters
- **Consistency**: Standardizes text format across documents
- **Noise Reduction**: Removes irrelevant information
- **Performance**: Improves model training efficiency
- **Accuracy**: Enhances model performance

## Text Cleaning

### Basic Cleaning Operations

```python
import re
import string
import unicodedata

def basic_cleaning(text):
    """
    Perform basic text cleaning operations.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def remove_punctuation(text, keep_apostrophe=True):
    """
    Remove punctuation from text.
    """
    if keep_apostrophe:
        # Keep apostrophes for contractions
        punctuation = string.punctuation.replace("'", "")
    else:
        punctuation = string.punctuation
    
    return text.translate(str.maketrans('', '', punctuation))

def remove_numbers(text, replace_with=''):
    """
    Remove or replace numbers in text.
    """
    return re.sub(r'\d+', replace_with, text)

def remove_urls(text):
    """
    Remove URLs from text.
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)

def remove_emails(text):
    """
    Remove email addresses from text.
    """
    email_pattern = re.compile(r'\S+@\S+')
    return email_pattern.sub('', text)

def remove_html_tags(text):
    """
    Remove HTML tags from text.
    """
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub('', text)
```

### Advanced Cleaning

```python
def normalize_unicode(text):
    """
    Normalize unicode characters.
    """
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Remove diacritics
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    
    return text

def handle_contractions(text):
    """
    Expand common contractions.
    """
    contractions = {
        "n't": " not",
        "'ll": " will",
        "'re": " are",
        "'ve": " have",
        "'m": " am",
        "'d": " would",
        "'s": " is"  # Note: This is simplified
    }
    
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    return text

def clean_special_characters(text):
    """
    Clean special characters while preserving important ones.
    """
    # Keep alphanumeric, spaces, and some punctuation
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\;\:\-\(\)]', '', text)
    
    # Clean up multiple punctuation
    text = re.sub(r'[\.\,\!\?\;\:]+', '.', text)
    
    return text
```

## Text Normalization

### Case Normalization

```python
def normalize_case(text, method='lower'):
    """
    Normalize text case.
    
    Args:
        text (str): Input text
        method (str): 'lower', 'upper', 'title', 'sentence'
    """
    if method == 'lower':
        return text.lower()
    elif method == 'upper':
        return text.upper()
    elif method == 'title':
        return text.title()
    elif method == 'sentence':
        return text.capitalize()
    else:
        return text
```

### Whitespace Normalization

```python
def normalize_whitespace(text):
    """
    Normalize whitespace in text.
    """
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove spaces around punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text
```

## Tokenization

### Word Tokenization

```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def tokenize_words(text):
    """
    Tokenize text into words using NLTK.
    """
    return word_tokenize(text)

def tokenize_sentences(text):
    """
    Tokenize text into sentences using NLTK.
    """
    return sent_tokenize(text)

def simple_word_tokenize(text):
    """
    Simple word tokenization using whitespace.
    """
    return text.split()

def regex_tokenize(text, pattern=r'\w+'):
    """
    Tokenize using regex pattern.
    """
    return re.findall(pattern, text)
```

### Subword Tokenization

```python
from transformers import AutoTokenizer

def get_subword_tokenizer(model_name='bert-base-uncased'):
    """
    Get a subword tokenizer from Hugging Face.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def tokenize_with_subwords(text, tokenizer):
    """
    Tokenize text using subword tokenization.
    """
    tokens = tokenizer.tokenize(text)
    return tokens
```

## Stop Word Removal

```python
from nltk.corpus import stopwords

# Download stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def remove_stopwords(tokens, language='english'):
    """
    Remove stopwords from tokenized text.
    """
    stop_words = set(stopwords.words(language))
    return [token for token in tokens if token.lower() not in stop_words]

def custom_stopwords_removal(tokens, custom_stopwords):
    """
    Remove custom stopwords.
    """
    return [token for token in tokens if token.lower() not in custom_stopwords]
```

## Stemming and Lemmatization

```python
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

# Download WordNet
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def get_wordnet_pos(tag):
    """
    Map POS tag to WordNet POS tag.
    """
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }
    return tag_dict.get(tag[0], wordnet.NOUN)

def apply_stemming(tokens):
    """
    Apply Porter stemming to tokens.
    """
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

def apply_lemmatization(tokens):
    """
    Apply lemmatization to tokens.
    """
    lemmatizer = WordNetLemmatizer()
    # Get POS tags for better lemmatization
    pos_tags = nltk.pos_tag(tokens)
    
    return [lemmatizer.lemmatize(token, get_wordnet_pos(pos)) 
            for token, pos in pos_tags]
```

## Handling Special Cases

### Handling Abbreviations

```python
def expand_abbreviations(text):
    """
    Expand common abbreviations.
    """
    abbreviations = {
        "Mr.": "Mister",
        "Mrs.": "Missus",
        "Dr.": "Doctor",
        "Prof.": "Professor",
        "vs.": "versus",
        "etc.": "et cetera",
        "i.e.": "that is",
        "e.g.": "for example"
    }
    
    for abbr, expansion in abbreviations.items():
        text = text.replace(abbr, expansion)
    
    return text
```

### Handling Emojis and Special Characters

```python
import emoji

def handle_emojis(text, method='remove'):
    """
    Handle emojis in text.
    
    Args:
        text (str): Input text
        method (str): 'remove', 'replace', 'keep'
    """
    if method == 'remove':
        return emoji.replace_emojis(text, replace='')
    elif method == 'replace':
        return emoji.demojize(text)
    else:
        return text

def clean_special_symbols(text):
    """
    Clean special symbols and characters.
    """
    # Remove or replace special symbols
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    return text
```

## Complete Pipeline

```python
class TextPreprocessor:
    """
    Complete text preprocessing pipeline.
    """
    
    def __init__(self, 
                 lowercase=True,
                 remove_punct=True,
                 remove_numbers=True,
                 remove_urls=True,
                 remove_emails=True,
                 remove_html=True,
                 normalize_unicode=True,
                 expand_contractions=True,
                 remove_stopwords=True,
                 apply_stemming=False,
                 apply_lemmatization=True):
        
        self.lowercase = lowercase
        self.remove_punct = remove_punct
        self.remove_numbers = remove_numbers
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_html = remove_html
        self.normalize_unicode = normalize_unicode
        self.expand_contractions = expand_contractions
        self.remove_stopwords = remove_stopwords
        self.apply_stemming = apply_stemming
        self.apply_lemmatization = apply_lemmatization
    
    def preprocess(self, text):
        """
        Apply complete preprocessing pipeline.
        """
        # Basic cleaning
        if self.remove_html:
            text = remove_html_tags(text)
        
        if self.remove_urls:
            text = remove_urls(text)
        
        if self.remove_emails:
            text = remove_emails(text)
        
        if self.normalize_unicode:
            text = normalize_unicode(text)
        
        if self.expand_contractions:
            text = handle_contractions(text)
        
        if self.lowercase:
            text = text.lower()
        
        if self.remove_numbers:
            text = remove_numbers(text)
        
        if self.remove_punct:
            text = remove_punctuation(text)
        
        # Normalize whitespace
        text = normalize_whitespace(text)
        
        # Tokenization
        tokens = tokenize_words(text)
        
        # Stop word removal
        if self.remove_stopwords:
            tokens = remove_stopwords(tokens)
        
        # Stemming or lemmatization
        if self.apply_stemming:
            tokens = apply_stemming(tokens)
        elif self.apply_lemmatization:
            tokens = apply_lemmatization(tokens)
        
        return tokens
    
    def preprocess_text(self, text):
        """
        Return preprocessed text as string.
        """
        tokens = self.preprocess(text)
        return ' '.join(tokens)

# Usage example
def example_usage():
    """
    Example usage of the text preprocessor.
    """
    # Sample text
    text = """
    Hello! This is a sample text with URLs (https://example.com), 
    emails (test@email.com), and HTML tags <b>like this</b>.
    It also has numbers 123, contractions like don't, and emojis 😊.
    """
    
    # Create preprocessor
    preprocessor = TextPreprocessor()
    
    # Preprocess
    processed_tokens = preprocessor.preprocess(text)
    processed_text = preprocessor.preprocess_text(text)
    
    print("Original text:", text)
    print("Processed tokens:", processed_tokens)
    print("Processed text:", processed_text)
    
    return processed_tokens, processed_text
```

## Best Practices

### 1. **Task-Specific Preprocessing**
- **Classification**: Focus on cleaning and normalization
- **Translation**: Preserve sentence structure
- **Generation**: Maintain context and flow
- **Summarization**: Keep important information

### 2. **Language Considerations**
- Use language-specific stopwords
- Consider morphological complexity
- Handle language-specific characters

### 3. **Performance Optimization**
- Process text in batches
- Use efficient libraries (SpaCy, NLTK)
- Cache preprocessing results

### 4. **Quality Control**
- Validate preprocessing results
- Maintain data consistency
- Document preprocessing steps

### 5. **Common Pitfalls to Avoid**
- Over-cleaning (removing important information)
- Inconsistent preprocessing across datasets
- Not considering downstream tasks
- Ignoring domain-specific requirements

## Example Applications

### Sentiment Analysis Preprocessing

```python
def preprocess_for_sentiment(text):
    """
    Preprocessing specifically for sentiment analysis.
    """
    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_punct=False,  # Keep punctuation for sentiment
        remove_numbers=True,
        remove_urls=True,
        remove_emails=True,
        remove_html=True,
        normalize_unicode=True,
        expand_contractions=True,
        remove_stopwords=False,  # Keep stopwords for sentiment
        apply_stemming=False,
        apply_lemmatization=True
    )
    
    return preprocessor.preprocess_text(text)
```

### Machine Translation Preprocessing

```python
def preprocess_for_translation(text):
    """
    Preprocessing specifically for machine translation.
    """
    preprocessor = TextPreprocessor(
        lowercase=False,  # Preserve case for translation
        remove_punct=False,  # Keep punctuation
        remove_numbers=False,  # Keep numbers
        remove_urls=True,
        remove_emails=True,
        remove_html=True,
        normalize_unicode=True,
        expand_contractions=False,  # Keep contractions
        remove_stopwords=False,  # Keep all words
        apply_stemming=False,
        apply_lemmatization=False  # Preserve original forms
    )
    
    return preprocessor.preprocess_text(text)
```

## Conclusion

Text preprocessing is a crucial step in any NLP pipeline. The key is to choose the right preprocessing steps based on your specific task and data characteristics. Always validate your preprocessing choices and ensure they improve your model's performance.

Remember: **Good preprocessing leads to better model performance!** 