"""
Text Preprocessing Example

This module demonstrates practical text preprocessing techniques
commonly used in NLP projects.
"""

import re
import string
import unicodedata
from typing import List, Optional, Callable
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TextCleaner:
    """
    A comprehensive text cleaning utility for NLP preprocessing.
    
    This class provides various text cleaning methods that can be
    combined to create custom preprocessing pipelines.
    """
    
    def __init__(self, 
                 remove_urls: bool = True,
                 remove_emails: bool = True,
                 remove_html: bool = True,
                 remove_numbers: bool = False,
                 remove_punctuation: bool = False,
                 lowercase: bool = True,
                 normalize_unicode: bool = True):
        """
        Initialize the TextCleaner with configuration options.
        
        Args:
            remove_urls: Whether to remove URLs from text
            remove_emails: Whether to remove email addresses
            remove_html: Whether to remove HTML tags
            remove_numbers: Whether to remove numbers
            remove_punctuation: Whether to remove punctuation
            lowercase: Whether to convert text to lowercase
            normalize_unicode: Whether to normalize unicode characters
        """
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_html = remove_html
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        self.normalize_unicode = normalize_unicode
        
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.html_pattern = re.compile('<.*?>')
        self.number_pattern = re.compile(r'\d+')
    
    def clean_text(self, text: str) -> str:
        """
        Apply all configured cleaning operations to the text.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        # Normalize unicode
        if self.normalize_unicode:
            text = self._normalize_unicode(text)
        
        # Remove HTML tags
        if self.remove_html:
            text = self._remove_html_tags(text)
        
        # Remove URLs
        if self.remove_urls:
            text = self._remove_urls(text)
        
        # Remove emails
        if self.remove_emails:
            text = self._remove_emails(text)
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove numbers
        if self.remove_numbers:
            text = self._remove_numbers(text)
        
        # Remove punctuation
        if self.remove_punctuation:
            text = self._remove_punctuation(text)
        
        # Normalize whitespace
        text = self._normalize_whitespace(text)
        
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        text = unicodedata.normalize('NFKD', text)
        text = ''.join([c for c in text if not unicodedata.combining(c)])
        return text
    
    def _remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text."""
        return self.html_pattern.sub('', text)
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        return self.url_pattern.sub('', text)
    
    def _remove_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        return self.email_pattern.sub('', text)
    
    def _remove_numbers(self, text: str) -> str:
        """Remove numbers from text."""
        return self.number_pattern.sub('', text)
    
    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text."""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Clean a batch of texts.
        
        Args:
            texts: List of texts to clean
            
        Returns:
            List of cleaned texts
        """
        return [self.clean_text(text) for text in texts]


class Tokenizer:
    """
    Text tokenization utilities.
    """
    
    def __init__(self, 
                 remove_stopwords: bool = True,
                 language: str = 'english',
                 stem: bool = False,
                 lemmatize: bool = True):
        """
        Initialize the Tokenizer.
        
        Args:
            remove_stopwords: Whether to remove stopwords
            language: Language for stopwords
            stem: Whether to apply stemming
            lemmatize: Whether to apply lemmatization
        """
        self.remove_stopwords = remove_stopwords
        self.language = language
        self.stem = stem
        self.lemmatize = lemmatize
        
        # Initialize NLTK components
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words(language))
        
        if self.stem:
            self.stemmer = PorterStemmer()
        
        if self.lemmatize:
            self.lemmatizer = WordNetLemmatizer()
    
    def tokenize_words(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        tokens = word_tokenize(text)
        
        if self.remove_stopwords:
            tokens = [token for token in tokens if token.lower() not in self.stop_words]
        
        if self.stem:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Tokenize text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        return sent_tokenize(text)


class TextPreprocessor:
    """
    Complete text preprocessing pipeline.
    
    This class combines text cleaning and tokenization into a
    single preprocessing pipeline.
    """
    
    def __init__(self, 
                 cleaner_config: Optional[dict] = None,
                 tokenizer_config: Optional[dict] = None):
        """
        Initialize the TextPreprocessor.
        
        Args:
            cleaner_config: Configuration for TextCleaner
            tokenizer_config: Configuration for Tokenizer
        """
        cleaner_config = cleaner_config or {}
        tokenizer_config = tokenizer_config or {}
        
        self.cleaner = TextCleaner(**cleaner_config)
        self.tokenizer = Tokenizer(**tokenizer_config)
    
    def preprocess(self, text: str, return_tokens: bool = False) -> str:
        """
        Preprocess text using the complete pipeline.
        
        Args:
            text: Input text
            return_tokens: Whether to return tokens or cleaned text
            
        Returns:
            Preprocessed text or tokens
        """
        # Clean text
        cleaned_text = self.cleaner.clean_text(text)
        
        if return_tokens:
            # Tokenize
            tokens = self.tokenizer.tokenize_words(cleaned_text)
            return tokens
        else:
            return cleaned_text
    
    def preprocess_batch(self, texts: List[str], return_tokens: bool = False) -> List:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of texts to preprocess
            return_tokens: Whether to return tokens or cleaned text
            
        Returns:
            List of preprocessed texts or tokens
        """
        results = []
        for text in texts:
            result = self.preprocess(text, return_tokens)
            results.append(result)
        return results


def example_usage():
    """
    Example usage of the text preprocessing utilities.
    """
    # Sample text
    sample_text = """
    Hello! This is a sample text with URLs (https://example.com), 
    emails (test@email.com), and HTML tags <b>like this</b>.
    It also has numbers 123, contractions like don't, and emojis 😊.
    """
    
    print("Original text:")
    print(sample_text)
    print("-" * 50)
    
    # Example 1: Basic cleaning
    cleaner = TextCleaner(
        remove_urls=True,
        remove_emails=True,
        remove_html=True,
        remove_numbers=False,
        remove_punctuation=False,
        lowercase=True
    )
    
    cleaned_text = cleaner.clean_text(sample_text)
    print("Cleaned text:")
    print(cleaned_text)
    print("-" * 50)
    
    # Example 2: Tokenization
    tokenizer = Tokenizer(
        remove_stopwords=True,
        stem=False,
        lemmatize=True
    )
    
    tokens = tokenizer.tokenize_words(cleaned_text)
    print("Tokens:")
    print(tokens)
    print("-" * 50)
    
    # Example 3: Complete preprocessing pipeline
    preprocessor = TextPreprocessor(
        cleaner_config={
            'remove_urls': True,
            'remove_emails': True,
            'remove_html': True,
            'lowercase': True
        },
        tokenizer_config={
            'remove_stopwords': True,
            'lemmatize': True
        }
    )
    
    processed_tokens = preprocessor.preprocess(sample_text, return_tokens=True)
    print("Processed tokens:")
    print(processed_tokens)
    print("-" * 50)
    
    # Example 4: Batch processing
    texts = [
        "I love this product! It's amazing.",
        "This is terrible. I'm disappointed.",
        "It's okay, nothing special."
    ]
    
    processed_batch = preprocessor.preprocess_batch(texts, return_tokens=True)
    print("Batch processed texts:")
    for i, tokens in enumerate(processed_batch):
        print(f"Text {i+1}: {tokens}")


if __name__ == "__main__":
    example_usage() 