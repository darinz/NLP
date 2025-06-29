"""
Utility functions and classes for NLP tasks.

This module provides various utility functions for text processing,
analysis, and NLP task support.
"""

from .text_utils import (
    TextAnalyzer,
    TextSimilarity,
    TextPreprocessor,
    TextFeatureExtractor,
    normalize_text,
    get_common_words,
    create_vocabulary
)

__all__ = [
    'TextAnalyzer',
    'TextSimilarity', 
    'TextPreprocessor',
    'TextFeatureExtractor',
    'normalize_text',
    'get_common_words',
    'create_vocabulary'
] 