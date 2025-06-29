"""
Text Utilities for NLP Tasks

This module provides utility functions for text processing,
analysis, and NLP task support.
"""

import re
import string
import unicodedata
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.util import ngrams


class TextAnalyzer:
    """
    Comprehensive text analysis utility.
    """
    
    def __init__(self, language='english'):
        """
        Initialize text analyzer.
        
        Args:
            language: Language for stopwords and processing
        """
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
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
    
    def clean_text(self, text: str, 
                   remove_punctuation: bool = True,
                   remove_numbers: bool = False,
                   remove_extra_spaces: bool = True,
                   lowercase: bool = True) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text
            remove_punctuation: Whether to remove punctuation
            remove_numbers: Whether to remove numbers
            remove_extra_spaces: Whether to normalize whitespace
            lowercase: Whether to convert to lowercase
            
        Returns:
            Cleaned text
        """
        if lowercase:
            text = text.lower()
        
        if remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        if remove_extra_spaces:
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with stopwords removed
        """
        words = word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)
    
    def lemmatize_text(self, text: str) -> str:
        """
        Lemmatize text.
        
        Args:
            text: Input text
            
        Returns:
            Lemmatized text
        """
        words = word_tokenize(text)
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)
    
    def stem_text(self, text: str) -> str:
        """
        Stem text.
        
        Args:
            text: Input text
            
        Returns:
            Stemmed text
        """
        words = word_tokenize(text)
        stemmed_words = [self.stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)
    
    def get_word_frequencies(self, text: str) -> Dict[str, int]:
        """
        Get word frequency distribution.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of word frequencies
        """
        words = word_tokenize(text.lower())
        return Counter(words)
    
    def get_ngrams(self, text: str, n: int = 2) -> List[Tuple[str, ...]]:
        """
        Get n-grams from text.
        
        Args:
            text: Input text
            n: N-gram size
            
        Returns:
            List of n-grams
        """
        words = word_tokenize(text)
        return list(ngrams(words, n))
    
    def get_sentence_lengths(self, text: str) -> List[int]:
        """
        Get sentence lengths.
        
        Args:
            text: Input text
            
        Returns:
            List of sentence lengths
        """
        sentences = sent_tokenize(text)
        return [len(word_tokenize(sent)) for sent in sentences]
    
    def get_text_statistics(self, text: str) -> Dict[str, any]:
        """
        Get comprehensive text statistics.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with text statistics
        """
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        word_freq = self.get_word_frequencies(text)
        
        # Basic statistics
        stats = {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'unique_words': len(set(words)),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'avg_sentence_length': np.mean([len(word_tokenize(sent)) for sent in sentences]) if sentences else 0,
            'vocabulary_diversity': len(set(words)) / len(words) if words else 0,
            'most_common_words': word_freq.most_common(10)
        }
        
        # Readability metrics
        if sentences and words:
            # Flesch Reading Ease (simplified)
            avg_sentence_length = stats['avg_sentence_length']
            avg_syllables_per_word = self._estimate_syllables_per_word(words)
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            stats['flesch_reading_ease'] = max(0, min(100, flesch_score))
        
        return stats
    
    def _estimate_syllables_per_word(self, words: List[str]) -> float:
        """
        Estimate syllables per word (simplified).
        
        Args:
            words: List of words
            
        Returns:
            Average syllables per word
        """
        syllable_count = 0
        for word in words:
            word = word.lower()
            count = 0
            vowels = "aeiouy"
            on_vowel = False
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not on_vowel:
                    count += 1
                on_vowel = is_vowel
            if word.endswith('e'):
                count -= 1
            if count == 0:
                count = 1
            syllable_count += count
        
        return syllable_count / len(words) if words else 0


class TextSimilarity:
    """
    Text similarity computation utilities.
    """
    
    def __init__(self):
        """Initialize text similarity calculator."""
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )
        self.count_vectorizer = CountVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )
    
    def cosine_similarity_tfidf(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity using TF-IDF vectors.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score
        """
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0.0
    
    def jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Compute Jaccard similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Jaccard similarity score
        """
        words1 = set(word_tokenize(text1.lower()))
        words2 = set(word_tokenize(text2.lower()))
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def edit_distance_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity based on edit distance.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Edit distance similarity score
        """
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        distance = levenshtein_distance(text1, text2)
        max_length = max(len(text1), len(text2))
        return 1 - (distance / max_length) if max_length > 0 else 1.0
    
    def semantic_similarity(self, texts: List[str], method: str = 'tfidf') -> np.ndarray:
        """
        Compute semantic similarity matrix for multiple texts.
        
        Args:
            texts: List of texts
            method: Similarity method ('tfidf', 'count', 'jaccard')
            
        Returns:
            Similarity matrix
        """
        n = len(texts)
        similarity_matrix = np.zeros((n, n))
        
        if method == 'tfidf':
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
                similarity_matrix = cosine_similarity(tfidf_matrix)
            except:
                pass
        elif method == 'count':
            try:
                count_matrix = self.count_vectorizer.fit_transform(texts)
                similarity_matrix = cosine_similarity(count_matrix)
            except:
                pass
        elif method == 'jaccard':
            for i in range(n):
                for j in range(n):
                    similarity_matrix[i][j] = self.jaccard_similarity(texts[i], texts[j])
        
        return similarity_matrix


class TextPreprocessor:
    """
    Advanced text preprocessing utilities.
    """
    
    def __init__(self, 
                 remove_urls: bool = True,
                 remove_emails: bool = True,
                 remove_phone_numbers: bool = True,
                 normalize_whitespace: bool = True,
                 remove_unicode: bool = False):
        """
        Initialize text preprocessor.
        
        Args:
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses
            remove_phone_numbers: Whether to remove phone numbers
            normalize_whitespace: Whether to normalize whitespace
            remove_unicode: Whether to remove unicode characters
        """
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_phone_numbers = remove_phone_numbers
        self.normalize_whitespace = normalize_whitespace
        self.remove_unicode = remove_unicode
    
    def preprocess(self, text: str) -> str:
        """
        Apply comprehensive text preprocessing.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if self.remove_urls:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        if self.remove_emails:
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        if self.remove_phone_numbers:
            text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        if self.remove_unicode:
            text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def batch_preprocess(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]


class TextFeatureExtractor:
    """
    Extract various features from text.
    """
    
    def __init__(self):
        """Initialize feature extractor."""
        self.analyzer = TextAnalyzer()
    
    def extract_basic_features(self, text: str) -> Dict[str, any]:
        """
        Extract basic text features.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of basic features
        """
        stats = self.analyzer.get_text_statistics(text)
        
        features = {
            'char_count': stats['char_count'],
            'word_count': stats['word_count'],
            'sentence_count': stats['sentence_count'],
            'unique_words': stats['unique_words'],
            'avg_word_length': stats['avg_word_length'],
            'avg_sentence_length': stats['avg_sentence_length'],
            'vocabulary_diversity': stats['vocabulary_diversity']
        }
        
        return features
    
    def extract_linguistic_features(self, text: str) -> Dict[str, any]:
        """
        Extract linguistic features.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of linguistic features
        """
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        
        # POS tag distribution
        pos_tags = nltk.pos_tag(words)
        pos_counts = Counter(tag for word, tag in pos_tags)
        
        # Named entity recognition
        try:
            ne_tree = nltk.ne_chunk(pos_tags)
            ne_count = len([chunk for chunk in ne_tree if hasattr(chunk, 'label')])
        except:
            ne_count = 0
        
        features = {
            'noun_count': pos_counts.get('NN', 0) + pos_counts.get('NNS', 0) + pos_counts.get('NNP', 0) + pos_counts.get('NNPS', 0),
            'verb_count': pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) + pos_counts.get('VBG', 0) + pos_counts.get('VBN', 0) + pos_counts.get('VBP', 0) + pos_counts.get('VBZ', 0),
            'adjective_count': pos_counts.get('JJ', 0) + pos_counts.get('JJR', 0) + pos_counts.get('JJS', 0),
            'adverb_count': pos_counts.get('RB', 0) + pos_counts.get('RBR', 0) + pos_counts.get('RBS', 0),
            'named_entity_count': ne_count,
            'stopword_ratio': len([w for w in words if w in self.analyzer.stop_words]) / len(words) if words else 0
        }
        
        return features
    
    def extract_sentiment_features(self, text: str) -> Dict[str, any]:
        """
        Extract sentiment-related features.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of sentiment features
        """
        # Simple sentiment indicators (in practice, use proper sentiment lexicons)
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'happy', 'joy'}
        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'sad', 'angry', 'frustrated', 'disappointed'}
        
        words = set(word_tokenize(text.lower()))
        
        features = {
            'positive_word_count': len(words.intersection(positive_words)),
            'negative_word_count': len(words.intersection(negative_words)),
            'sentiment_ratio': (len(words.intersection(positive_words)) - len(words.intersection(negative_words))) / len(words) if words else 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
        }
        
        return features


# Utility functions
def normalize_text(text: str) -> str:
    """
    Basic text normalization.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters (keep alphanumeric and spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    return text


def get_common_words(texts: List[str], top_n: int = 10) -> List[Tuple[str, int]]:
    """
    Get most common words across multiple texts.
    
    Args:
        texts: List of texts
        top_n: Number of top words to return
        
    Returns:
        List of (word, count) tuples
    """
    all_words = []
    for text in texts:
        words = word_tokenize(text.lower())
        all_words.extend(words)
    
    word_counts = Counter(all_words)
    return word_counts.most_common(top_n)


def create_vocabulary(texts: List[str], min_freq: int = 1, max_vocab_size: int = 10000) -> Dict[str, int]:
    """
    Create vocabulary from texts.
    
    Args:
        texts: List of texts
        min_freq: Minimum word frequency
        max_vocab_size: Maximum vocabulary size
        
    Returns:
        Dictionary mapping words to indices
    """
    word_counts = Counter()
    for text in texts:
        words = word_tokenize(text.lower())
        word_counts.update(words)
    
    # Filter by minimum frequency
    filtered_words = {word: count for word, count in word_counts.items() 
                     if count >= min_freq}
    
    # Sort by frequency and limit size
    sorted_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)
    vocab_words = ['<PAD>', '<UNK>'] + [word for word, _ in sorted_words[:max_vocab_size-2]]
    
    return {word: idx for idx, word in enumerate(vocab_words)}


# Example usage
if __name__ == "__main__":
    # Example text
    sample_text = """
    Natural language processing (NLP) is a subfield of linguistics, computer science, 
    and artificial intelligence concerned with the interactions between computers and 
    human language, in particular how to program computers to process and analyze 
    large amounts of natural language data.
    """
    
    # Initialize analyzer
    analyzer = TextAnalyzer()
    
    # Get text statistics
    stats = analyzer.get_text_statistics(sample_text)
    print("Text Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Initialize similarity calculator
    similarity = TextSimilarity()
    
    # Example similarity computation
    text1 = "Natural language processing is fascinating."
    text2 = "NLP is an interesting field of study."
    
    cos_sim = similarity.cosine_similarity_tfidf(text1, text2)
    jaccard_sim = similarity.jaccard_similarity(text1, text2)
    
    print(f"\nSimilarity Scores:")
    print(f"  Cosine (TF-IDF): {cos_sim:.3f}")
    print(f"  Jaccard: {jaccard_sim:.3f}")
    
    # Initialize feature extractor
    extractor = TextFeatureExtractor()
    
    # Extract features
    basic_features = extractor.extract_basic_features(sample_text)
    linguistic_features = extractor.extract_linguistic_features(sample_text)
    sentiment_features = extractor.extract_sentiment_features(sample_text)
    
    print(f"\nBasic Features:")
    for key, value in basic_features.items():
        print(f"  {key}: {value}")
    
    print(f"\nLinguistic Features:")
    for key, value in linguistic_features.items():
        print(f"  {key}: {value}")
    
    print(f"\nSentiment Features:")
    for key, value in sentiment_features.items():
        print(f"  {key}: {value}") 