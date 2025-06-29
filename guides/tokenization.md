# Tokenization in Natural Language Processing

A comprehensive guide to text tokenization techniques, from basic word-level to advanced subword methods.

## Table of Contents
1. [Introduction](#introduction)
2. [Types of Tokenization](#types-of-tokenization)
3. [Word-Level Tokenization](#word-level-tokenization)
4. [Subword Tokenization](#subword-tokenization)
5. [Character-Level Tokenization](#character-level-tokenization)
6. [Sentence Tokenization](#sentence-tokenization)
7. [Implementation Examples](#implementation-examples)
8. [Best Practices](#best-practices)
9. [Advanced Techniques](#advanced-techniques)

## Introduction

Tokenization is the process of breaking down text into smaller units called tokens. These tokens can be words, subwords, characters, or sentences, depending on the application and model requirements.

### Why Tokenization Matters
- **Model Input**: Neural networks require numerical input
- **Vocabulary Management**: Controls vocabulary size and handles unknown words
- **Language Understanding**: Different tokenization strategies capture different linguistic patterns
- **Performance**: Affects model training speed and memory usage

## Types of Tokenization

### 1. Word-Level Tokenization
- **Definition**: Splits text into individual words
- **Pros**: Simple, interpretable, preserves word meaning
- **Cons**: Large vocabulary, doesn't handle unknown words well
- **Use Cases**: Traditional NLP, simple text classification

### 2. Subword Tokenization
- **Definition**: Splits words into smaller meaningful units
- **Pros**: Handles unknown words, smaller vocabulary, multilingual
- **Cons**: More complex, less interpretable
- **Use Cases**: Modern transformers, multilingual models

### 3. Character-Level Tokenization
- **Definition**: Splits text into individual characters
- **Pros**: Small vocabulary, handles any text
- **Cons**: Loses word-level semantics, longer sequences
- **Use Cases**: Language modeling, text generation

### 4. Sentence-Level Tokenization
- **Definition**: Splits text into sentences
- **Pros**: Preserves sentence structure, useful for document-level tasks
- **Cons**: Requires additional processing for word-level tasks
- **Use Cases**: Document classification, summarization

## Word-Level Tokenization

### Basic Word Tokenization

```python
import re
import nltk
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt')

def basic_word_tokenize(text: str) -> list:
    """
    Basic word tokenization using regex.
    """
    # Simple regex-based tokenization
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

def nltk_word_tokenize(text: str) -> list:
    """
    Word tokenization using NLTK.
    """
    return word_tokenize(text)

# Example usage
text = "Hello, world! This is a sample text with punctuation."
print("Basic:", basic_word_tokenize(text))
print("NLTK:", nltk_word_tokenize(text))
```

### Advanced Word Tokenization

```python
import spacy
from nltk.tokenize import word_tokenize, WordPunctTokenizer
from nltk.tokenize import TreebankWordTokenizer

class AdvancedWordTokenizer:
    """
    Advanced word tokenization with multiple strategies.
    """
    
    def __init__(self, method='nltk'):
        self.method = method
        self.nlp = spacy.load('en_core_web_sm') if method == 'spacy' else None
    
    def tokenize(self, text: str) -> list:
        """
        Tokenize text using the specified method.
        """
        if self.method == 'nltk':
            return word_tokenize(text)
        elif self.method == 'wordpunct':
            return WordPunctTokenizer().tokenize(text)
        elif self.method == 'treebank':
            return TreebankWordTokenizer().tokenize(text)
        elif self.method == 'spacy':
            doc = self.nlp(text)
            return [token.text for token in doc if not token.is_space]
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def tokenize_with_info(self, text: str) -> list:
        """
        Tokenize with additional linguistic information.
        """
        if self.method != 'spacy':
            raise ValueError("Linguistic info only available with spaCy")
        
        doc = self.nlp(text)
        tokens = []
        for token in doc:
            if not token.is_space:
                tokens.append({
                    'text': token.text,
                    'lemma': token.lemma_,
                    'pos': token.pos_,
                    'tag': token.tag_,
                    'dep': token.dep_,
                    'is_stop': token.is_stop
                })
        return tokens

# Example usage
tokenizer = AdvancedWordTokenizer('spacy')
text = "The quick brown fox jumps over the lazy dog."
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)

# With linguistic information
tokens_info = tokenizer.tokenize_with_info(text)
for token in tokens_info[:3]:
    print(f"Text: {token['text']}, POS: {token['pos']}, Lemma: {token['lemma']}")
```

## Subword Tokenization

### Byte Pair Encoding (BPE)

```python
import re
from collections import defaultdict, Counter

class BytePairEncoder:
    """
    Implementation of Byte Pair Encoding (BPE).
    """
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}
    
    def get_stats(self, words):
        """
        Count frequency of adjacent pairs of symbols.
        """
        pairs = defaultdict(int)
        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs
    
    def merge_pair(self, pair, words):
        """
        Merge all occurrences of the pair in the vocabulary.
        """
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        new_words = {}
        
        for word, freq in words.items():
            new_word = word.replace(bigram, replacement)
            new_words[new_word] = freq
        
        return new_words
    
    def train(self, texts):
        """
        Train BPE on a list of texts.
        """
        # Initialize vocabulary with characters
        word_freqs = defaultdict(int)
        for text in texts:
            words = text.split()
            for word in words:
                word_freqs[' '.join(list(word)) + ' </w>'] += 1
        
        # Learn merges
        for i in range(self.vocab_size - len(self.vocab)):
            pairs = self.get_stats(word_freqs)
            if not pairs:
                break
            
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            word_freqs = self.merge_pair(best_pair, word_freqs)
            self.merges[best_pair] = i
    
    def encode(self, text):
        """
        Encode text using learned BPE merges.
        """
        words = text.split()
        encoded_words = []
        
        for word in words:
            word = ' '.join(list(word)) + ' </w>'
            
            # Apply merges
            while True:
                pairs = self.get_stats({word: 1})
                if not pairs:
                    break
                
                # Find the highest ranked pair
                bigram = min(pairs.keys(), key=lambda p: self.merges.get(p, float('inf')))
                if bigram not in self.merges:
                    break
                
                word = self.merge_pair(bigram, {word: 1})
                word = list(word.keys())[0]
            
            encoded_words.append(word)
        
        return ' '.join(encoded_words)

# Example usage
bpe = BytePairEncoder(vocab_size=50)
texts = [
    "the quick brown fox",
    "the lazy dog",
    "quick brown fox jumps",
    "lazy dog sleeps"
]
bpe.train(texts)
encoded = bpe.encode("the quick fox")
print("BPE encoded:", encoded)
```

### WordPiece Tokenization

```python
import re
from collections import Counter

class WordPieceTokenizer:
    """
    Implementation of WordPiece tokenization.
    """
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3}
        self.unk_token = '[UNK]'
    
    def compute_pair_scores(self, words):
        """
        Compute scores for all adjacent pairs.
        """
        pairs = defaultdict(int)
        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        
        scores = {}
        for pair, freq in pairs.items():
            first, second = pair
            score = freq / (words.get(first, 0) * words.get(second, 0))
            scores[pair] = score
        
        return scores
    
    def train(self, texts):
        """
        Train WordPiece on a list of texts.
        """
        # Initialize vocabulary with characters
        word_freqs = Counter()
        for text in texts:
            words = text.split()
            for word in words:
                word_freqs[' '.join(list(word)) + ' </w>'] += 1
        
        # Learn vocabulary
        while len(self.vocab) < self.vocab_size:
            scores = self.compute_pair_scores(word_freqs)
            if not scores:
                break
            
            # Find best pair
            best_pair = max(scores, key=scores.get)
            new_token = ''.join(best_pair)
            
            # Add to vocabulary
            self.vocab[new_token] = len(self.vocab)
            
            # Update word frequencies
            new_word_freqs = Counter()
            for word, freq in word_freqs.items():
                new_word = word.replace(' '.join(best_pair), new_token)
                new_word_freqs[new_word] += freq
            word_freqs = new_word_freqs
    
    def tokenize(self, text):
        """
        Tokenize text using WordPiece.
        """
        words = text.split()
        tokens = []
        
        for word in words:
            word = ' '.join(list(word)) + ' </w>'
            sub_tokens = []
            
            while word:
                # Find longest matching subword
                i = len(word)
                while i > 0:
                    subword = word[:i]
                    if subword in self.vocab:
                        sub_tokens.append(subword)
                        word = word[i:]
                        break
                    i -= 1
                
                if i == 0:
                    # Unknown word
                    sub_tokens.append(self.unk_token)
                    break
            
            tokens.extend(sub_tokens)
        
        return tokens

# Example usage
wp = WordPieceTokenizer(vocab_size=50)
texts = [
    "the quick brown fox",
    "the lazy dog",
    "quick brown fox jumps",
    "lazy dog sleeps"
]
wp.train(texts)
tokens = wp.tokenize("the quick fox")
print("WordPiece tokens:", tokens)
```

## Character-Level Tokenization

```python
class CharacterTokenizer:
    """
    Character-level tokenization.
    """
    
    def __init__(self, lowercase=True, include_special=True):
        self.lowercase = lowercase
        self.include_special = include_special
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
    
    def fit(self, texts):
        """
        Build vocabulary from texts.
        """
        chars = set()
        for text in texts:
            if self.lowercase:
                text = text.lower()
            chars.update(text)
        
        if self.include_special:
            special_chars = ['<PAD>', '<UNK>', '<START>', '<END>']
            chars.update(special_chars)
        
        # Create mappings
        for i, char in enumerate(sorted(chars)):
            self.char_to_idx[char] = i
            self.idx_to_char[i] = char
        
        self.vocab_size = len(self.char_to_idx)
    
    def encode(self, text):
        """
        Encode text to character indices.
        """
        if self.lowercase:
            text = text.lower()
        
        indices = []
        for char in text:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                indices.append(self.char_to_idx.get('<UNK>', 0))
        
        return indices
    
    def decode(self, indices):
        """
        Decode character indices back to text.
        """
        chars = []
        for idx in indices:
            if idx in self.idx_to_char:
                chars.append(self.idx_to_char[idx])
        
        return ''.join(chars)

# Example usage
char_tokenizer = CharacterTokenizer()
texts = ["Hello world!", "Python programming", "NLP is fun"]
char_tokenizer.fit(texts)

encoded = char_tokenizer.encode("Hello")
print("Encoded:", encoded)
decoded = char_tokenizer.decode(encoded)
print("Decoded:", decoded)
```

## Sentence Tokenization

```python
import nltk
from nltk.tokenize import sent_tokenize
import spacy

class SentenceTokenizer:
    """
    Sentence-level tokenization.
    """
    
    def __init__(self, method='nltk'):
        self.method = method
        if method == 'spacy':
            self.nlp = spacy.load('en_core_web_sm')
    
    def tokenize(self, text):
        """
        Split text into sentences.
        """
        if self.method == 'nltk':
            return sent_tokenize(text)
        elif self.method == 'spacy':
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents]
        elif self.method == 'simple':
            # Simple regex-based sentence splitting
            import re
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def tokenize_with_metadata(self, text):
        """
        Tokenize with additional sentence metadata.
        """
        if self.method != 'spacy':
            raise ValueError("Metadata only available with spaCy")
        
        doc = self.nlp(text)
        sentences = []
        
        for sent in doc.sents:
            sentences.append({
                'text': sent.text.strip(),
                'start': sent.start_char,
                'end': sent.end_char,
                'length': len(sent),
                'word_count': len([token for token in sent if not token.is_space])
            })
        
        return sentences

# Example usage
sent_tokenizer = SentenceTokenizer('nltk')
text = "Hello world! This is a test. How are you today?"
sentences = sent_tokenizer.tokenize(text)
print("Sentences:", sentences)
```

## Implementation Examples

### Complete Tokenization Pipeline

```python
from typing import List, Dict, Union
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy

class NLPTokenizer:
    """
    Comprehensive NLP tokenization pipeline.
    """
    
    def __init__(self, 
                 word_method='nltk',
                 sent_method='nltk',
                 lowercase=True,
                 remove_punctuation=False,
                 remove_stopwords=False):
        
        self.word_method = word_method
        self.sent_method = sent_method
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        
        # Initialize tokenizers
        if word_method == 'spacy':
            self.nlp = spacy.load('en_core_web_sm')
        
        if remove_stopwords:
            nltk.download('stopwords')
            self.stop_words = set(nltk.corpus.stopwords.words('english'))
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text before tokenization.
        """
        if self.lowercase:
            text = text.lower()
        
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        return text
    
    def tokenize_words(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        """
        text = self.preprocess_text(text)
        
        if self.word_method == 'nltk':
            tokens = word_tokenize(text)
        elif self.word_method == 'spacy':
            doc = self.nlp(text)
            tokens = [token.text for token in doc if not token.is_space]
        elif self.word_method == 'simple':
            tokens = text.split()
        else:
            raise ValueError(f"Unknown word method: {self.word_method}")
        
        # Remove stopwords if requested
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        return tokens
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Tokenize text into sentences.
        """
        if self.sent_method == 'nltk':
            return sent_tokenize(text)
        elif self.sent_method == 'spacy':
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents]
        elif self.sent_method == 'simple':
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
        else:
            raise ValueError(f"Unknown sentence method: {self.sent_method}")
    
    def tokenize_document(self, text: str) -> Dict[str, Union[List[str], List[List[str]]]]:
        """
        Complete document tokenization.
        """
        sentences = self.tokenize_sentences(text)
        words_by_sentence = [self.tokenize_words(sent) for sent in sentences]
        
        return {
            'sentences': sentences,
            'words_by_sentence': words_by_sentence,
            'all_words': [word for sent in words_by_sentence for word in sent]
        }
    
    def get_statistics(self, text: str) -> Dict[str, int]:
        """
        Get tokenization statistics.
        """
        result = self.tokenize_document(text)
        
        return {
            'sentence_count': len(result['sentences']),
            'word_count': len(result['all_words']),
            'unique_words': len(set(result['all_words'])),
            'avg_sentence_length': len(result['all_words']) / len(result['sentences'])
        }

# Example usage
tokenizer = NLPTokenizer(
    word_method='nltk',
    sent_method='nltk',
    lowercase=True,
    remove_punctuation=False,
    remove_stopwords=False
)

text = "Hello world! This is a test document. It contains multiple sentences."
result = tokenizer.tokenize_document(text)
stats = tokenizer.get_statistics(text)

print("Sentences:", result['sentences'])
print("Words by sentence:", result['words_by_sentence'])
print("Statistics:", stats)
```

### Subword Tokenization with Hugging Face

```python
from transformers import AutoTokenizer

def demonstrate_subword_tokenization():
    """
    Demonstrate subword tokenization with Hugging Face transformers.
    """
    
    # Load different tokenizers
    tokenizers = {
        'BERT': AutoTokenizer.from_pretrained('bert-base-uncased'),
        'GPT-2': AutoTokenizer.from_pretrained('gpt2'),
        'T5': AutoTokenizer.from_pretrained('t5-base'),
        'RoBERTa': AutoTokenizer.from_pretrained('roberta-base')
    }
    
    test_texts = [
        "Hello world!",
        "The quick brown fox jumps over the lazy dog.",
        "Natural language processing is fascinating.",
        "Supercalifragilisticexpialidocious"
    ]
    
    for name, tokenizer in tokenizers.items():
        print(f"\n{name} Tokenization:")
        print("-" * 40)
        
        for text in test_texts:
            tokens = tokenizer.tokenize(text)
            token_ids = tokenizer.encode(text)
            
            print(f"Text: {text}")
            print(f"Tokens: {tokens}")
            print(f"Token IDs: {token_ids}")
            print(f"Vocabulary size: {tokenizer.vocab_size}")
            print()
    
    return tokenizers

# Run demonstration
tokenizers = demonstrate_subword_tokenization()
```

## Best Practices

### 1. Choose the Right Tokenization Method

```python
def choose_tokenization_method(task_type: str, language: str, model_type: str) -> str:
    """
    Choose appropriate tokenization method based on requirements.
    """
    
    if model_type == 'transformer':
        if language == 'multilingual':
            return 'sentencepiece'  # or 'mBART'
        else:
            return 'wordpiece'  # or 'BPE'
    
    elif task_type == 'classification':
        return 'word'
    
    elif task_type == 'generation':
        return 'subword'
    
    elif task_type == 'language_modeling':
        return 'character' if small_vocab else 'subword'
    
    else:
        return 'word'  # default
```

### 2. Handle Special Cases

```python
class RobustTokenizer:
    """
    Robust tokenization with special case handling.
    """
    
    def __init__(self):
        self.contractions = {
            "n't": " not",
            "'ll": " will",
            "'re": " are",
            "'ve": " have",
            "'m": " am",
            "'d": " would",
            "'s": " is"  # Note: this is simplified
        }
        
        self.special_patterns = [
            (r'[^\w\s]', ' '),  # Replace punctuation with space
            (r'\s+', ' '),      # Normalize whitespace
        ]
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text before tokenization.
        """
        # Handle contractions
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        
        # Apply special patterns
        for pattern, replacement in self.special_patterns:
            text = re.sub(pattern, replacement, text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Robust tokenization with normalization.
        """
        normalized = self.normalize_text(text)
        tokens = normalized.split()
        
        # Filter out empty tokens
        tokens = [token for token in tokens if token]
        
        return tokens
```

### 3. Vocabulary Management

```python
class VocabularyManager:
    """
    Manage vocabulary for tokenization.
    """
    
    def __init__(self, min_freq=1, max_vocab_size=50000):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.word_freq = Counter()
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.unk_token = '<UNK>'
        self.pad_token = '<PAD>'
    
    def build_vocab(self, texts: List[str]):
        """
        Build vocabulary from texts.
        """
        # Count word frequencies
        for text in texts:
            words = text.split()
            self.word_freq.update(words)
        
        # Filter by minimum frequency
        filtered_words = {word: freq for word, freq in self.word_freq.items() 
                         if freq >= self.min_freq}
        
        # Sort by frequency and limit size
        sorted_words = sorted(filtered_words.items(), 
                             key=lambda x: x[1], reverse=True)
        
        # Add special tokens
        vocab_words = [self.unk_token, self.pad_token] + \
                     [word for word, _ in sorted_words[:self.max_vocab_size-2]]
        
        # Create mappings
        for idx, word in enumerate(vocab_words):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to indices.
        """
        words = text.split()
        indices = []
        
        for word in words:
            if word in self.word_to_idx:
                indices.append(self.word_to_idx[word])
            else:
                indices.append(self.word_to_idx[self.unk_token])
        
        return indices
    
    def decode(self, indices: List[int]) -> str:
        """
        Decode indices back to text.
        """
        words = []
        for idx in indices:
            if idx in self.idx_to_word:
                word = self.idx_to_word[idx]
                if word not in [self.unk_token, self.pad_token]:
                    words.append(word)
        
        return ' '.join(words)
```

## Advanced Techniques

### 1. Multilingual Tokenization

```python
from transformers import AutoTokenizer

class MultilingualTokenizer:
    """
    Handle multilingual tokenization.
    """
    
    def __init__(self, model_name='xlm-roberta-base'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.language_codes = {
            'en': 'english',
            'es': 'spanish',
            'fr': 'french',
            'de': 'german',
            'zh': 'chinese',
            'ja': 'japanese'
        }
    
    def detect_language(self, text: str) -> str:
        """
        Simple language detection (in practice, use langdetect or similar).
        """
        # This is a simplified version
        # In practice, use proper language detection libraries
        return 'en'  # Default to English
    
    def tokenize_multilingual(self, texts: List[str]) -> List[List[str]]:
        """
        Tokenize texts in multiple languages.
        """
        tokenized_texts = []
        
        for text in texts:
            # Add language-specific tokens if needed
            tokens = self.tokenizer.tokenize(text)
            tokenized_texts.append(tokens)
        
        return tokenized_texts
```

### 2. Domain-Specific Tokenization

```python
class DomainSpecificTokenizer:
    """
    Tokenization for specific domains (medical, legal, technical).
    """
    
    def __init__(self, domain='general'):
        self.domain = domain
        self.domain_patterns = self._load_domain_patterns()
    
    def _load_domain_patterns(self):
        """
        Load domain-specific patterns.
        """
        patterns = {
            'medical': [
                (r'\b\d+\.\d+\s*mg\b', 'DOSAGE'),
                (r'\b\d+\.\d+\s*ml\b', 'VOLUME'),
                (r'\b[A-Z]{2,}\b', 'ABBREVIATION'),
            ],
            'legal': [
                (r'\b\d+\.\d+\.\d+\b', 'SECTION_REFERENCE'),
                (r'\b[A-Z]+\s+v\.\s+[A-Z]+\b', 'CASE_REFERENCE'),
            ],
            'technical': [
                (r'\b[A-Z]{2,}\b', 'ACRONYM'),
                (r'\b\d+\.\d+\.\d+\b', 'VERSION_NUMBER'),
            ]
        }
        return patterns.get(self.domain, [])
    
    def tokenize(self, text: str) -> List[str]:
        """
        Domain-specific tokenization.
        """
        # Apply domain patterns
        for pattern, replacement in self.domain_patterns:
            text = re.sub(pattern, replacement, text)
        
        # Standard tokenization
        tokens = text.split()
        return tokens
```

### 3. Streaming Tokenization

```python
class StreamingTokenizer:
    """
    Tokenization for streaming text data.
    """
    
    def __init__(self, buffer_size=1000):
        self.buffer_size = buffer_size
        self.buffer = ""
    
    def add_text(self, text: str) -> List[str]:
        """
        Add text to buffer and return complete tokens.
        """
        self.buffer += text
        tokens = []
        
        # Process complete words
        while ' ' in self.buffer:
            space_idx = self.buffer.find(' ')
            word = self.buffer[:space_idx]
            if word:
                tokens.append(word)
            self.buffer = self.buffer[space_idx + 1:]
        
        # Keep buffer size manageable
        if len(self.buffer) > self.buffer_size:
            # Force tokenization of remaining text
            if self.buffer:
                tokens.append(self.buffer)
                self.buffer = ""
        
        return tokens
    
    def flush(self) -> List[str]:
        """
        Flush remaining text in buffer.
        """
        tokens = []
        if self.buffer:
            tokens.append(self.buffer)
            self.buffer = ""
        return tokens
```

## Conclusion

Tokenization is a fundamental step in NLP that significantly impacts model performance. The choice of tokenization method depends on:

1. **Task Requirements**: Different tasks benefit from different tokenization strategies
2. **Language Characteristics**: Some languages require specialized tokenization
3. **Model Architecture**: Transformers work best with subword tokenization
4. **Computational Constraints**: Vocabulary size affects memory and speed

### Key Takeaways

- **Word-level tokenization** is simple but limited by vocabulary size
- **Subword tokenization** handles unknown words and reduces vocabulary size
- **Character-level tokenization** is universal but loses semantic information
- **Sentence tokenization** is crucial for document-level tasks
- **Domain-specific tokenization** improves performance on specialized texts

### Next Steps

1. Experiment with different tokenization methods on your specific task
2. Consider the trade-offs between vocabulary size and tokenization quality
3. Implement robust error handling for edge cases
4. Monitor tokenization statistics to optimize performance

---

**Happy Tokenizing!** 