# Word Embeddings Guide

A comprehensive guide to word embeddings, their types, implementations, and applications in NLP.

## Table of Contents
1. [Introduction](#introduction)
2. [Traditional Methods](#traditional-methods)
3. [Neural Word Embeddings](#neural-word-embeddings)
4. [Word2Vec](#word2vec)
5. [GloVe](#glove)
6. [FastText](#fasttext)
7. [Contextual Embeddings](#contextual-embeddings)
8. [Implementation Examples](#implementation-examples)
9. [Best Practices](#best-practices)

## Introduction

Word embeddings are dense vector representations of words that capture semantic and syntactic relationships. They form the foundation of modern NLP systems.

### Why Word Embeddings Matter
- **Semantic Understanding**: Words with similar meanings have similar vectors
- **Mathematical Operations**: Can perform arithmetic on word vectors
- **Dimensionality Reduction**: Compact representation of large vocabularies
- **Transfer Learning**: Pre-trained embeddings improve model performance

## Traditional Methods

### One-Hot Encoding

```python
import numpy as np
from collections import Counter

def create_one_hot_encoding(texts, max_vocab_size=10000):
    """
    Create one-hot encoding for a collection of texts.
    """
    # Build vocabulary
    word_counts = Counter()
    for text in texts:
        words = text.lower().split()
        word_counts.update(words)
    
    # Get most common words
    vocab = ['<UNK>'] + [word for word, count in word_counts.most_common(max_vocab_size - 1)]
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    # Create one-hot encoding
    def text_to_one_hot(text):
        words = text.lower().split()
        encoding = np.zeros(len(vocab))
        for word in words:
            if word in word_to_idx:
                encoding[word_to_idx[word]] = 1
            else:
                encoding[word_to_idx['<UNK>']] = 1
        return encoding
    
    return vocab, word_to_idx, text_to_one_hot

# Example usage
texts = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "a bird flew over the tree"
]

vocab, word_to_idx, encoder = create_one_hot_encoding(texts)
print("Vocabulary:", vocab[:10])
print("One-hot encoding of 'the cat sat':", encoder("the cat sat")[:10])
```

### Bag of Words (BoW)

```python
from sklearn.feature_extraction.text import CountVectorizer

def create_bow_representation(texts, max_features=1000):
    """
    Create Bag of Words representation.
    """
    vectorizer = CountVectorizer(
        max_features=max_features,
        stop_words='english',
        lowercase=True,
        ngram_range=(1, 1)
    )
    
    bow_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    return bow_matrix, feature_names, vectorizer

# Example usage
texts = [
    "I love machine learning",
    "Machine learning is fascinating",
    "I enjoy studying machine learning algorithms"
]

bow_matrix, features, vectorizer = create_bow_representation(texts)
print("Feature names:", features[:10])
print("BoW matrix shape:", bow_matrix.shape)
```

### TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_representation(texts, max_features=1000):
    """
    Create TF-IDF representation.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        lowercase=True,
        ngram_range=(1, 2)
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    return tfidf_matrix, feature_names, vectorizer

# Example usage
texts = [
    "Natural language processing is a subfield of artificial intelligence",
    "Machine learning algorithms can process natural language",
    "Deep learning has revolutionized natural language processing"
]

tfidf_matrix, features, vectorizer = create_tfidf_representation(texts)
print("TF-IDF matrix shape:", tfidf_matrix.shape)
```

## Neural Word Embeddings

### Simple Neural Embedding

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleEmbedding(nn.Module):
    """
    Simple neural network for learning word embeddings.
    """
    
    def __init__(self, vocab_size, embedding_dim):
        super(SimpleEmbedding, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output = self.linear(embedded)
        return output

def create_training_data(texts, window_size=2):
    """
    Create training data for word embedding learning.
    """
    # Build vocabulary
    word_counts = Counter()
    for text in texts:
        words = text.lower().split()
        word_counts.update(words)
    
    vocab = ['<UNK>'] + [word for word, count in word_counts.most_common(1000)]
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    # Create training pairs
    training_pairs = []
    for text in texts:
        words = text.lower().split()
        word_indices = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in words]
        
        for i, target_word in enumerate(word_indices):
            # Get context words
            start = max(0, i - window_size)
            end = min(len(word_indices), i + window_size + 1)
            
            for j in range(start, end):
                if j != i:
                    training_pairs.append((target_word, word_indices[j]))
    
    return training_pairs, vocab, word_to_idx

def train_simple_embedding(texts, embedding_dim=50, epochs=100, learning_rate=0.01):
    """
    Train a simple word embedding model.
    """
    # Create training data
    training_pairs, vocab, word_to_idx = create_training_data(texts)
    
    # Create model
    model = SimpleEmbedding(len(vocab), embedding_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for target, context in training_pairs:
            target_tensor = torch.tensor([target], dtype=torch.long)
            context_tensor = torch.tensor([context], dtype=torch.long)
            
            optimizer.zero_grad()
            output = model(target_tensor)
            loss = criterion(output, context_tensor.squeeze())
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(training_pairs):.4f}')
    
    return model, vocab, word_to_idx
```

## Word2Vec

### CBOW Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class CBOWDataset(Dataset):
    """
    Dataset for CBOW (Continuous Bag of Words) training.
    """
    
    def __init__(self, texts, window_size=2, min_count=5):
        self.window_size = window_size
        self.min_count = min_count
        
        # Build vocabulary
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        # Filter by minimum count
        self.vocab = ['<UNK>'] + [word for word, count in word_counts.items() 
                                 if count >= min_count]
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        # Create training pairs
        self.training_pairs = []
        for text in texts:
            words = text.lower().split()
            word_indices = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) 
                           for word in words]
            
            for i, target_word in enumerate(word_indices):
                # Get context words
                context_words = []
                for j in range(max(0, i - window_size), min(len(word_indices), i + window_size + 1)):
                    if j != i:
                        context_words.append(word_indices[j])
                
                # Create multiple training examples
                for context_word in context_words:
                    self.training_pairs.append((context_word, target_word))
    
    def __len__(self):
        return len(self.training_pairs)
    
    def __getitem__(self, idx):
        context, target = self.training_pairs[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)

class CBOW(nn.Module):
    """
    CBOW (Continuous Bag of Words) model.
    """
    
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = self.linear(embeds)
        return out

def train_cbow(texts, embedding_dim=100, window_size=2, epochs=10, batch_size=32):
    """
    Train CBOW word embeddings.
    """
    # Create dataset
    dataset = CBOWDataset(texts, window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = CBOW(len(dataset.vocab), embedding_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for context, target in dataloader:
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
    
    return model, dataset.vocab, dataset.word_to_idx

# Example usage
texts = [
    "the quick brown fox jumps over the lazy dog",
    "a quick brown dog jumps over the lazy fox",
    "the lazy fox sleeps while the quick brown dog runs",
    "a dog and a fox are both animals",
    "the quick brown fox is fast and clever"
]

model, vocab, word_to_idx = train_cbow(texts, embedding_dim=50, epochs=50)
```

### Skip-gram Implementation

```python
class SkipGramDataset(Dataset):
    """
    Dataset for Skip-gram training.
    """
    
    def __init__(self, texts, window_size=2, min_count=5):
        self.window_size = window_size
        self.min_count = min_count
        
        # Build vocabulary (same as CBOW)
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        self.vocab = ['<UNK>'] + [word for word, count in word_counts.items() 
                                 if count >= min_count]
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        # Create training pairs (target -> context)
        self.training_pairs = []
        for text in texts:
            words = text.lower().split()
            word_indices = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) 
                           for word in words]
            
            for i, target_word in enumerate(word_indices):
                # Get context words
                for j in range(max(0, i - window_size), min(len(word_indices), i + window_size + 1)):
                    if j != i:
                        self.training_pairs.append((target_word, word_indices[j]))
    
    def __len__(self):
        return len(self.training_pairs)
    
    def __getitem__(self, idx):
        target, context = self.training_pairs[idx]
        return torch.tensor(target, dtype=torch.long), torch.tensor(context, dtype=torch.long)

class SkipGram(nn.Module):
    """
    Skip-gram model.
    """
    
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = self.linear(embeds)
        return out

def train_skipgram(texts, embedding_dim=100, window_size=2, epochs=10, batch_size=32):
    """
    Train Skip-gram word embeddings.
    """
    # Create dataset
    dataset = SkipGramDataset(texts, window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = SkipGram(len(dataset.vocab), embedding_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for target, context in dataloader:
            optimizer.zero_grad()
            output = model(target)
            loss = criterion(output, context)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
    
    return model, dataset.vocab, dataset.word_to_idx
```

## GloVe

### GloVe Implementation

```python
import numpy as np
from scipy.sparse import csr_matrix

class GloVe:
    """
    GloVe (Global Vectors for Word Representation) implementation.
    """
    
    def __init__(self, embedding_dim=100, window_size=10, min_count=5):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.min_count = min_count
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.cooccurrence_matrix = None
        self.embeddings = None
    
    def build_vocabulary(self, texts):
        """
        Build vocabulary from texts.
        """
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        # Filter by minimum count
        vocab = [word for word, count in word_counts.items() if count >= self.min_count]
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        return vocab
    
    def build_cooccurrence_matrix(self, texts):
        """
        Build co-occurrence matrix.
        """
        vocab_size = len(self.word_to_idx)
        self.cooccurrence_matrix = np.zeros((vocab_size, vocab_size))
        
        for text in texts:
            words = text.lower().split()
            word_indices = [self.word_to_idx.get(word) for word in words 
                           if word in self.word_to_idx]
            
            for i, word_idx in enumerate(word_indices):
                if word_idx is None:
                    continue
                
                # Count co-occurrences within window
                start = max(0, i - self.window_size)
                end = min(len(word_indices), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i != j and word_indices[j] is not None:
                        distance = abs(i - j)
                        weight = 1.0 / distance
                        self.cooccurrence_matrix[word_idx, word_indices[j]] += weight
    
    def train(self, texts, epochs=50, learning_rate=0.05, x_max=100, alpha=0.75):
        """
        Train GloVe embeddings.
        """
        # Build vocabulary and co-occurrence matrix
        self.build_vocabulary(texts)
        self.build_cooccurrence_matrix(texts)
        
        vocab_size = len(self.word_to_idx)
        
        # Initialize embeddings
        self.W = np.random.randn(vocab_size, self.embedding_dim) * 0.1
        self.W_tilde = np.random.randn(vocab_size, self.embedding_dim) * 0.1
        self.b = np.zeros(vocab_size)
        self.b_tilde = np.zeros(vocab_size)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(vocab_size):
                for j in range(vocab_size):
                    if self.cooccurrence_matrix[i, j] > 0:
                        # Calculate f(x)
                        x = self.cooccurrence_matrix[i, j]
                        if x < x_max:
                            f_x = (x / x_max) ** alpha
                        else:
                            f_x = 1.0
                        
                        # Calculate prediction
                        pred = (self.W[i] * self.W_tilde[j]).sum() + self.b[i] + self.b_tilde[j]
                        
                        # Calculate loss
                        loss = f_x * (pred - np.log(x)) ** 2
                        total_loss += loss
                        
                        # Calculate gradients
                        grad = 2 * f_x * (pred - np.log(x))
                        
                        # Update parameters
                        self.W[i] -= learning_rate * grad * self.W_tilde[j]
                        self.W_tilde[j] -= learning_rate * grad * self.W[i]
                        self.b[i] -= learning_rate * grad
                        self.b_tilde[j] -= learning_rate * grad
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}')
        
        # Final embeddings are the sum of W and W_tilde
        self.embeddings = self.W + self.W_tilde
    
    def get_embedding(self, word):
        """
        Get embedding for a word.
        """
        if word in self.word_to_idx:
            return self.embeddings[self.word_to_idx[word]]
        else:
            return None
    
    def find_similar_words(self, word, top_k=5):
        """
        Find most similar words.
        """
        if word not in self.word_to_idx:
            return []
        
        target_embedding = self.embeddings[self.word_to_idx[word]]
        similarities = []
        
        for idx, other_word in self.idx_to_word.items():
            if other_word != word:
                other_embedding = self.embeddings[idx]
                similarity = np.dot(target_embedding, other_embedding) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(other_embedding))
                similarities.append((other_word, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

# Example usage
texts = [
    "the quick brown fox jumps over the lazy dog",
    "a quick brown dog jumps over the lazy fox",
    "the lazy fox sleeps while the quick brown dog runs",
    "a dog and a fox are both animals",
    "the quick brown fox is fast and clever",
    "dogs and foxes are different types of animals",
    "the fox is clever and the dog is loyal",
    "quick animals like foxes and dogs can run fast"
]

glove = GloVe(embedding_dim=50, window_size=5)
glove.train(texts, epochs=100)

# Find similar words
print("Words similar to 'fox':", glove.find_similar_words('fox'))
print("Words similar to 'quick':", glove.find_similar_words('quick'))
```

## FastText

### FastText Implementation

```python
class FastTextDataset(Dataset):
    """
    Dataset for FastText training.
    """
    
    def __init__(self, texts, window_size=2, min_count=5, ngrams=3):
        self.window_size = window_size
        self.min_count = min_count
        self.ngrams = ngrams
        
        # Build vocabulary
        word_counts = Counter()
        ngram_counts = Counter()
        
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
            
            # Generate n-grams
            for word in words:
                ngrams_list = self.get_ngrams(word, ngrams)
                ngram_counts.update(ngrams_list)
        
        # Create vocabulary with words and n-grams
        self.vocab = ['<UNK>'] + [word for word, count in word_counts.items() 
                                 if count >= min_count]
        self.ngram_vocab = ['<UNK>'] + [ngram for ngram, count in ngram_counts.items() 
                                       if count >= min_count]
        
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.ngram_to_idx = {ngram: idx for idx, ngram in enumerate(self.ngram_vocab)}
        
        # Create training pairs
        self.training_pairs = []
        for text in texts:
            words = text.lower().split()
            word_indices = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) 
                           for word in words]
            
            for i, target_word in enumerate(word_indices):
                # Get context words
                for j in range(max(0, i - window_size), min(len(word_indices), i + window_size + 1)):
                    if j != i:
                        self.training_pairs.append((target_word, word_indices[j]))
    
    def get_ngrams(self, word, n):
        """
        Generate n-grams for a word.
        """
        ngrams = []
        word = '<' + word + '>'
        for i in range(len(word) - n + 1):
            ngrams.append(word[i:i+n])
        return ngrams
    
    def __len__(self):
        return len(self.training_pairs)
    
    def __getitem__(self, idx):
        target, context = self.training_pairs[idx]
        return torch.tensor(target, dtype=torch.long), torch.tensor(context, dtype=torch.long)

class FastText(nn.Module):
    """
    FastText model.
    """
    
    def __init__(self, vocab_size, ngram_vocab_size, embedding_dim):
        super(FastText, self).__init__()
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.ngram_embeddings = nn.Embedding(ngram_vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, word_indices, ngram_indices):
        # Word embeddings
        word_embeds = self.word_embeddings(word_indices)
        
        # N-gram embeddings
        ngram_embeds = self.ngram_embeddings(ngram_indices)
        
        # Combine embeddings
        combined_embeds = word_embeds + ngram_embeds.mean(dim=1)
        
        output = self.linear(combined_embeds)
        return output

def train_fasttext(texts, embedding_dim=100, window_size=2, epochs=10, batch_size=32):
    """
    Train FastText embeddings.
    """
    # Create dataset
    dataset = FastTextDataset(texts, window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = FastText(len(dataset.vocab), len(dataset.ngram_vocab), embedding_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for target, context in dataloader:
            optimizer.zero_grad()
            
            # Get n-grams for target words
            ngram_indices = []
            for idx in target:
                word = dataset.idx_to_word[idx.item()]
                ngrams = dataset.get_ngrams(word, 3)
                ngram_idx = [dataset.ngram_to_idx.get(ngram, 0) for ngram in ngrams]
                ngram_indices.append(ngram_idx)
            
            # Pad n-gram sequences
            max_ngrams = max(len(ngrams) for ngrams in ngram_indices)
            padded_ngrams = []
            for ngrams in ngram_indices:
                padded = ngrams + [0] * (max_ngrams - len(ngrams))
                padded_ngrams.append(padded)
            
            ngram_tensor = torch.tensor(padded_ngrams, dtype=torch.long)
            
            output = model(target, ngram_tensor)
            loss = criterion(output, context)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
    
    return model, dataset.vocab, dataset.word_to_idx
```

## Contextual Embeddings

### Using Pre-trained Embeddings

```python
from transformers import AutoTokenizer, AutoModel
import torch

def get_bert_embeddings(texts, model_name='bert-base-uncased'):
    """
    Get BERT embeddings for texts.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    embeddings = []
    
    for text in texts:
        # Tokenize
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token embedding or mean pooling
            embedding = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
            embeddings.append(embedding.squeeze().numpy())
    
    return embeddings

def get_word_embeddings_from_bert(text, word, model_name='bert-base-uncased'):
    """
    Get contextual embedding for a specific word in a text.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt')
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state
    
    # Find the word tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    word_embeddings = []
    
    for i, token in enumerate(tokens):
        if word.lower() in token.lower():
            word_embeddings.append(hidden_states[0, i].numpy())
    
    return word_embeddings

# Example usage
texts = [
    "The bank is near the river.",
    "I need to go to the bank to withdraw money."
]

# Get sentence embeddings
sentence_embeddings = get_bert_embeddings(texts)

# Get contextual word embeddings
bank_embeddings_1 = get_word_embeddings_from_bert(texts[0], "bank")
bank_embeddings_2 = get_word_embeddings_from_bert(texts[1], "bank")

print("Sentence embeddings shape:", sentence_embeddings[0].shape)
print("Contextual 'bank' embeddings:", len(bank_embeddings_1), len(bank_embeddings_2))
```

## Implementation Examples

### Word Similarity and Analogies

```python
def word_similarity(word1, word2, embeddings, word_to_idx):
    """
    Calculate cosine similarity between two words.
    """
    if word1 not in word_to_idx or word2 not in word_to_idx:
        return 0.0
    
    emb1 = embeddings[word_to_idx[word1]]
    emb2 = embeddings[word_to_idx[word2]]
    
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity

def word_analogies(word1, word2, word3, embeddings, word_to_idx, top_k=5):
    """
    Solve word analogies: word1 : word2 :: word3 : ?
    """
    if not all(word in word_to_idx for word in [word1, word2, word3]):
        return []
    
    # Calculate analogy vector
    vec1 = embeddings[word_to_idx[word1]]
    vec2 = embeddings[word_to_idx[word2]]
    vec3 = embeddings[word_to_idx[word3]]
    
    analogy_vec = vec2 - vec1 + vec3
    
    # Find most similar words
    similarities = []
    for word, idx in word_to_idx.items():
        if word not in [word1, word2, word3]:
            word_vec = embeddings[idx]
            similarity = np.dot(analogy_vec, word_vec) / (
                np.linalg.norm(analogy_vec) * np.linalg.norm(word_vec))
            similarities.append((word, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# Example usage
def example_analogies():
    """
    Example of word analogies.
    """
    # Train embeddings
    texts = [
        "king is to queen as man is to woman",
        "paris is to france as tokyo is to japan",
        "cat is to kitten as dog is to puppy",
        "happy is to sad as good is to bad",
        "big is to small as tall is to short"
    ]
    
    model, vocab, word_to_idx = train_cbow(texts, embedding_dim=50, epochs=100)
    embeddings = model.embeddings.weight.detach().numpy()
    
    # Test analogies
    analogies = [
        ("king", "queen", "man"),
        ("paris", "france", "tokyo"),
        ("cat", "kitten", "dog")
    ]
    
    for word1, word2, word3 in analogies:
        results = word_analogies(word1, word2, word3, embeddings, word_to_idx)
        print(f"{word1} : {word2} :: {word3} : {results[0][0] if results else 'unknown'}")
```

## Best Practices

### 1. **Choosing Embedding Type**
- **Word2Vec**: Good for general-purpose embeddings
- **GloVe**: Better for semantic relationships
- **FastText**: Good for morphologically rich languages
- **Contextual**: Best for modern NLP tasks

### 2. **Training Parameters**
- **Embedding Dimension**: 50-300 for most tasks
- **Window Size**: 2-10 depending on task
- **Minimum Count**: 5-10 to filter rare words
- **Training Epochs**: 10-100 depending on data size

### 3. **Data Quality**
- Use large, diverse text corpora
- Clean and preprocess text properly
- Consider domain-specific data for specialized tasks

### 4. **Evaluation**
- Use word similarity datasets
- Test on analogy tasks
- Evaluate on downstream tasks

### 5. **Common Pitfalls**
- Insufficient training data
- Poor text preprocessing
- Inappropriate hyperparameters
- Ignoring out-of-vocabulary words

## Conclusion

Word embeddings are fundamental to modern NLP. Choose the right embedding type based on your task requirements and data characteristics. Pre-trained embeddings often provide excellent performance, but training custom embeddings can be beneficial for domain-specific applications.

Remember: **Good embeddings lead to better NLP models!** 