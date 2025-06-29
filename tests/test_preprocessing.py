"""
Unit tests for text preprocessing utilities.

This module contains comprehensive tests for text preprocessing
functions and classes used in the repository.
"""

import unittest
import sys
import os
import numpy as np
from unittest.mock import Mock, patch

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.preprocessing.text_cleaner import TextCleaner, Tokenizer, TextPreprocessor


class TestTextCleaner(unittest.TestCase):
    """Test cases for TextCleaner class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cleaner = TextCleaner()
        self.sample_text = "Hello, World! This is a test. It has numbers 123 and symbols @#$%."
    
    def test_initialization(self):
        """Test TextCleaner initialization."""
        self.assertIsNotNone(self.cleaner)
        self.assertTrue(hasattr(self.cleaner, 'remove_punctuation'))
        self.assertTrue(hasattr(self.cleaner, 'remove_numbers'))
        self.assertTrue(hasattr(self.cleaner, 'lowercase'))
    
    def test_basic_cleaning(self):
        """Test basic text cleaning."""
        cleaned = self.cleaner.clean(self.sample_text)
        
        self.assertIsInstance(cleaned, str)
        self.assertNotEqual(cleaned, self.sample_text)
        self.assertIn('hello', cleaned.lower())
    
    def test_remove_punctuation(self):
        """Test punctuation removal."""
        cleaner = TextCleaner(remove_punctuation=True)
        cleaned = cleaner.clean(self.sample_text)
        
        # Check that punctuation is removed
        punctuation = '.,!?;:'
        for char in punctuation:
            self.assertNotIn(char, cleaned)
    
    def test_remove_numbers(self):
        """Test number removal."""
        cleaner = TextCleaner(remove_numbers=True)
        cleaned = cleaner.clean(self.sample_text)
        
        # Check that numbers are removed
        self.assertNotIn('123', cleaned)
    
    def test_lowercase_conversion(self):
        """Test lowercase conversion."""
        cleaner = TextCleaner(lowercase=True)
        cleaned = cleaner.clean(self.sample_text)
        
        # Check that text is lowercase
        self.assertEqual(cleaned, cleaned.lower())
    
    def test_remove_extra_spaces(self):
        """Test extra space removal."""
        text_with_spaces = "  Hello   World  !  "
        cleaned = self.cleaner.clean(text_with_spaces)
        
        # Check that extra spaces are removed
        self.assertNotIn('  ', cleaned)
        self.assertEqual(cleaned.strip(), cleaned)
    
    def test_remove_special_characters(self):
        """Test special character removal."""
        text_with_special = "Hello @#$% World!"
        cleaned = self.cleaner.clean(text_with_special)
        
        # Check that special characters are removed
        special_chars = '@#$%'
        for char in special_chars:
            self.assertNotIn(char, cleaned)
    
    def test_batch_cleaning(self):
        """Test batch text cleaning."""
        texts = [
            "Hello, World!",
            "This is a test.",
            "Numbers 123 and symbols @#$%."
        ]
        
        cleaned_texts = self.cleaner.clean_batch(texts)
        
        self.assertEqual(len(cleaned_texts), len(texts))
        for cleaned in cleaned_texts:
            self.assertIsInstance(cleaned, str)
    
    def test_empty_text(self):
        """Test cleaning empty text."""
        empty_text = ""
        cleaned = self.cleaner.clean(empty_text)
        
        self.assertEqual(cleaned, "")
    
    def test_whitespace_only_text(self):
        """Test cleaning whitespace-only text."""
        whitespace_text = "   \n\t   "
        cleaned = self.cleaner.clean(whitespace_text)
        
        self.assertEqual(cleaned, "")


class TestTokenizer(unittest.TestCase):
    """Test cases for Tokenizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer = Tokenizer()
        self.sample_text = "Hello world! This is a test sentence."
    
    def test_initialization(self):
        """Test Tokenizer initialization."""
        self.assertIsNotNone(self.tokenizer)
        self.assertTrue(hasattr(self.tokenizer, 'remove_stopwords'))
        self.assertTrue(hasattr(self.tokenizer, 'lemmatize'))
    
    def test_basic_tokenization(self):
        """Test basic tokenization."""
        tokens = self.tokenizer.tokenize(self.sample_text)
        
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        self.assertIn('hello', [token.lower() for token in tokens])
    
    def test_remove_stopwords(self):
        """Test stopword removal."""
        tokenizer = Tokenizer(remove_stopwords=True)
        tokens = tokenizer.tokenize(self.sample_text)
        
        # Check that common stopwords are removed
        stopwords = ['is', 'a', 'this']
        for stopword in stopwords:
            self.assertNotIn(stopword.lower(), [token.lower() for token in tokens])
    
    def test_lemmatization(self):
        """Test lemmatization."""
        tokenizer = Tokenizer(lemmatize=True)
        text = "running cats dogs"
        tokens = tokenizer.tokenize(text)
        
        # Check that words are lemmatized
        lemmatized_words = [token.lower() for token in tokens]
        self.assertIn('run', lemmatized_words)
        self.assertIn('cat', lemmatized_words)
        self.assertIn('dog', lemmatized_words)
    
    def test_batch_tokenization(self):
        """Test batch tokenization."""
        texts = [
            "Hello world!",
            "This is a test.",
            "Another sentence here."
        ]
        
        tokenized_texts = self.tokenizer.tokenize_batch(texts)
        
        self.assertEqual(len(tokenized_texts), len(texts))
        for tokens in tokenized_texts:
            self.assertIsInstance(tokens, list)
    
    def test_empty_text_tokenization(self):
        """Test tokenization of empty text."""
        empty_text = ""
        tokens = self.tokenizer.tokenize(empty_text)
        
        self.assertEqual(tokens, [])
    
    def test_single_word_tokenization(self):
        """Test tokenization of single word."""
        single_word = "Hello"
        tokens = self.tokenizer.tokenize(single_word)
        
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].lower(), 'hello')


class TestTextPreprocessor(unittest.TestCase):
    """Test cases for TextPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = TextPreprocessor()
        self.sample_text = "Hello, World! This is a test. It has numbers 123 and symbols @#$%."
    
    def test_initialization(self):
        """Test TextPreprocessor initialization."""
        self.assertIsNotNone(self.preprocessor)
        self.assertIsNotNone(self.preprocessor.cleaner)
        self.assertIsNotNone(self.preprocessor.tokenizer)
    
    def test_full_preprocessing(self):
        """Test full preprocessing pipeline."""
        processed = self.preprocessor.preprocess(self.sample_text)
        
        self.assertIsInstance(processed, str)
        self.assertNotEqual(processed, self.sample_text)
    
    def test_batch_preprocessing(self):
        """Test batch preprocessing."""
        texts = [
            "Hello, World!",
            "This is a test.",
            "Numbers 123 and symbols @#$%."
        ]
        
        processed_texts = self.preprocessor.preprocess_batch(texts)
        
        self.assertEqual(len(processed_texts), len(texts))
        for processed in processed_texts:
            self.assertIsInstance(processed, str)
    
    def test_preprocessing_with_custom_settings(self):
        """Test preprocessing with custom settings."""
        preprocessor = TextPreprocessor(
            remove_punctuation=True,
            remove_numbers=True,
            remove_stopwords=True,
            lemmatize=True
        )
        
        processed = preprocessor.preprocess(self.sample_text)
        
        # Check that preprocessing was applied
        self.assertNotIn('123', processed)
        self.assertNotIn(',', processed)
        self.assertNotIn('!', processed)
    
    def test_empty_text_preprocessing(self):
        """Test preprocessing of empty text."""
        empty_text = ""
        processed = self.preprocessor.preprocess(empty_text)
        
        self.assertEqual(processed, "")
    
    def test_preprocessing_statistics(self):
        """Test preprocessing statistics."""
        texts = [
            "Hello world!",
            "This is a test.",
            "Another sentence here."
        ]
        
        stats = self.preprocessor.get_statistics(texts)
        
        self.assertIn('total_texts', stats)
        self.assertIn('avg_text_length', stats)
        self.assertIn('total_tokens', stats)
        self.assertIn('unique_tokens', stats)
        
        self.assertEqual(stats['total_texts'], len(texts))
        self.assertGreater(stats['total_tokens'], 0)


class TestTextUtils(unittest.TestCase):
    """Test cases for text utility functions."""
    
    def test_normalize_text(self):
        """Test text normalization function."""
        from examples.utils.text_utils import normalize_text
        
        text = "  Hello, World!  This is a TEST.  "
        normalized = normalize_text(text)
        
        self.assertIsInstance(normalized, str)
        self.assertNotIn(',', normalized)
        self.assertNotIn('!', normalized)
        self.assertNotIn('  ', normalized)
        self.assertEqual(normalized, normalized.lower())
    
    def test_get_common_words(self):
        """Test common words extraction."""
        from examples.utils.text_utils import get_common_words
        
        texts = [
            "Hello world hello",
            "World is great",
            "Hello there world"
        ]
        
        common_words = get_common_words(texts, top_n=3)
        
        self.assertIsInstance(common_words, list)
        self.assertLessEqual(len(common_words), 3)
        
        for word, count in common_words:
            self.assertIsInstance(word, str)
            self.assertIsInstance(count, int)
            self.assertGreater(count, 0)
    
    def test_create_vocabulary(self):
        """Test vocabulary creation."""
        from examples.utils.text_utils import create_vocabulary
        
        texts = [
            "Hello world",
            "World is great",
            "Hello there"
        ]
        
        vocab = create_vocabulary(texts, min_freq=1, max_vocab_size=10)
        
        self.assertIsInstance(vocab, dict)
        self.assertIn('<PAD>', vocab)
        self.assertIn('<UNK>', vocab)
        self.assertIn('hello', vocab)
        self.assertIn('world', vocab)
        
        # Check that indices are sequential
        indices = list(vocab.values())
        self.assertEqual(indices, list(range(len(indices))))


class TestTextAnalyzer(unittest.TestCase):
    """Test cases for TextAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from examples.utils.text_utils import TextAnalyzer
        self.analyzer = TextAnalyzer()
        self.sample_text = "Hello world! This is a test sentence. It contains multiple words."
    
    def test_initialization(self):
        """Test TextAnalyzer initialization."""
        self.assertIsNotNone(self.analyzer)
        self.assertIsNotNone(self.analyzer.stop_words)
        self.assertIsNotNone(self.analyzer.lemmatizer)
        self.assertIsNotNone(self.analyzer.stemmer)
    
    def test_clean_text(self):
        """Test text cleaning."""
        cleaned = self.analyzer.clean_text(self.sample_text)
        
        self.assertIsInstance(cleaned, str)
        self.assertNotEqual(cleaned, self.sample_text)
    
    def test_remove_stopwords(self):
        """Test stopword removal."""
        text = "This is a test sentence with stopwords"
        filtered = self.analyzer.remove_stopwords(text)
        
        # Check that stopwords are removed
        stopwords = ['this', 'is', 'a', 'with']
        for stopword in stopwords:
            self.assertNotIn(stopword, filtered.lower())
    
    def test_lemmatize_text(self):
        """Test text lemmatization."""
        text = "running cats dogs"
        lemmatized = self.analyzer.lemmatize_text(text)
        
        # Check that words are lemmatized
        lemmatized_words = lemmatized.lower().split()
        self.assertIn('run', lemmatized_words)
        self.assertIn('cat', lemmatized_words)
        self.assertIn('dog', lemmatized_words)
    
    def test_get_word_frequencies(self):
        """Test word frequency extraction."""
        text = "hello world hello test world"
        frequencies = self.analyzer.get_word_frequencies(text)
        
        self.assertIsInstance(frequencies, dict)
        self.assertEqual(frequencies['hello'], 2)
        self.assertEqual(frequencies['world'], 2)
        self.assertEqual(frequencies['test'], 1)
    
    def test_get_ngrams(self):
        """Test n-gram extraction."""
        text = "hello world test"
        bigrams = self.analyzer.get_ngrams(text, n=2)
        
        self.assertIsInstance(bigrams, list)
        self.assertEqual(len(bigrams), 2)
        self.assertIn(('hello', 'world'), bigrams)
        self.assertIn(('world', 'test'), bigrams)
    
    def test_get_text_statistics(self):
        """Test text statistics extraction."""
        stats = self.analyzer.get_text_statistics(self.sample_text)
        
        self.assertIsInstance(stats, dict)
        self.assertIn('char_count', stats)
        self.assertIn('word_count', stats)
        self.assertIn('sentence_count', stats)
        self.assertIn('unique_words', stats)
        self.assertIn('avg_word_length', stats)
        self.assertIn('avg_sentence_length', stats)
        self.assertIn('vocabulary_diversity', stats)
        self.assertIn('most_common_words', stats)
        
        # Check that statistics are reasonable
        self.assertGreater(stats['char_count'], 0)
        self.assertGreater(stats['word_count'], 0)
        self.assertGreater(stats['sentence_count'], 0)
        self.assertGreaterEqual(stats['vocabulary_diversity'], 0)
        self.assertLessEqual(stats['vocabulary_diversity'], 1)


class TestTextSimilarity(unittest.TestCase):
    """Test cases for TextSimilarity class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from examples.utils.text_utils import TextSimilarity
        self.similarity = TextSimilarity()
        self.text1 = "Hello world"
        self.text2 = "Hello there"
        self.text3 = "Completely different text"
    
    def test_initialization(self):
        """Test TextSimilarity initialization."""
        self.assertIsNotNone(self.similarity)
        self.assertIsNotNone(self.similarity.tfidf_vectorizer)
        self.assertIsNotNone(self.similarity.count_vectorizer)
    
    def test_cosine_similarity_tfidf(self):
        """Test TF-IDF cosine similarity."""
        similarity_score = self.similarity.cosine_similarity_tfidf(self.text1, self.text2)
        
        self.assertIsInstance(similarity_score, float)
        self.assertGreaterEqual(similarity_score, 0.0)
        self.assertLessEqual(similarity_score, 1.0)
        
        # Similar texts should have higher similarity
        similarity_similar = self.similarity.cosine_similarity_tfidf(self.text1, self.text2)
        similarity_different = self.similarity.cosine_similarity_tfidf(self.text1, self.text3)
        
        self.assertGreaterEqual(similarity_similar, similarity_different)
    
    def test_jaccard_similarity(self):
        """Test Jaccard similarity."""
        similarity_score = self.similarity.jaccard_similarity(self.text1, self.text2)
        
        self.assertIsInstance(similarity_score, float)
        self.assertGreaterEqual(similarity_score, 0.0)
        self.assertLessEqual(similarity_score, 1.0)
    
    def test_edit_distance_similarity(self):
        """Test edit distance similarity."""
        similarity_score = self.similarity.edit_distance_similarity(self.text1, self.text2)
        
        self.assertIsInstance(similarity_score, float)
        self.assertGreaterEqual(similarity_score, 0.0)
        self.assertLessEqual(similarity_score, 1.0)
    
    def test_semantic_similarity_matrix(self):
        """Test semantic similarity matrix."""
        texts = [self.text1, self.text2, self.text3]
        similarity_matrix = self.similarity.semantic_similarity(texts, method='tfidf')
        
        self.assertIsInstance(similarity_matrix, np.ndarray)
        self.assertEqual(similarity_matrix.shape, (len(texts), len(texts)))
        
        # Check that diagonal elements are 1.0 (self-similarity)
        for i in range(len(texts)):
            self.assertAlmostEqual(similarity_matrix[i][i], 1.0, places=5)
        
        # Check that matrix is symmetric
        self.assertTrue(np.allclose(similarity_matrix, similarity_matrix.T))


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2) 