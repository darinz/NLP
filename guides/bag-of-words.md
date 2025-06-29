# Bag of Words and TF-IDF in Natural Language Processing

A comprehensive guide to traditional text representation methods including Bag of Words, TF-IDF, and their applications.

## Table of Contents
1. [Introduction](#introduction)
2. [Bag of Words (BoW)](#bag-of-words-bow)
3. [Term Frequency (TF)](#term-frequency-tf)
4. [Inverse Document Frequency (IDF)](#inverse-document-frequency-idf)
5. [TF-IDF](#tf-idf)
6. [Implementation Examples](#implementation-examples)
7. [Advanced Techniques](#advanced-techniques)
8. [Applications](#applications)
9. [Best Practices](#best-practices)
10. [Limitations and Alternatives](#limitations-and-alternatives)

## Introduction

Bag of Words (BoW) and TF-IDF are fundamental text representation techniques in Natural Language Processing. They convert text documents into numerical vectors that can be used by machine learning algorithms.

### Why These Methods Matter
- **Text Vectorization**: Convert text to numerical representations
- **Document Similarity**: Measure similarity between documents
- **Feature Extraction**: Extract meaningful features from text
- **Classification**: Enable text classification algorithms
- **Information Retrieval**: Power search and recommendation systems

## Bag of Words (BoW)

### Basic Concept

Bag of Words represents text as a vector where each dimension corresponds to a word in the vocabulary, and the value indicates the frequency of that word in the document.

### Simple Implementation

```python
from collections import Counter
import re

def create_bow_simple(text: str) -> dict:
    """
    Create a simple bag of words representation.
    """
    # Clean and tokenize
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    
    # Count word frequencies
    word_counts = Counter(words)
    
    return dict(word_counts)

# Example usage
text = "The quick brown fox jumps over the lazy dog. The fox is quick."
bow = create_bow_simple(text)
print("Bag of Words:")
for word, count in bow.items():
    print(f"  {word}: {count}")
```

### Advanced BoW Implementation

```python
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

class BagOfWords:
    """
    Advanced Bag of Words implementation.
    """
    
    def __init__(self, 
                 max_features: int = 1000,
                 min_df: int = 1,
                 max_df: float = 1.0,
                 ngram_range: tuple = (1, 1),
                 stop_words: str = 'english'):
        """
        Initialize Bag of Words vectorizer.
        
        Args:
            max_features: Maximum number of features
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            ngram_range: Range of n-grams to consider
            stop_words: Stop words to remove
        """
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            stop_words=stop_words,
            lowercase=True
        )
        self.vocabulary = None
        self.feature_names = None
    
    def fit(self, texts: list) -> 'BagOfWords':
        """
        Fit the vectorizer on training texts.
        """
        self.vectorizer.fit(texts)
        self.vocabulary = self.vectorizer.vocabulary_
        self.feature_names = self.vectorizer.get_feature_names_out()
        return self
    
    def transform(self, texts: list) -> np.ndarray:
        """
        Transform texts to BoW vectors.
        """
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts: list) -> np.ndarray:
        """
        Fit and transform texts.
        """
        return self.vectorizer.fit_transform(texts)
    
    def get_feature_names(self) -> list:
        """
        Get feature names (words).
        """
        return self.feature_names.tolist()
    
    def get_vocabulary(self) -> dict:
        """
        Get vocabulary mapping.
        """
        return self.vocabulary
    
    def analyze_document(self, text: str) -> dict:
        """
        Analyze a single document.
        """
        vector = self.transform([text])
        feature_names = self.get_feature_names()
        
        # Get non-zero features
        nonzero_indices = vector.nonzero()[1]
        nonzero_values = vector.data
        
        analysis = {}
        for idx, value in zip(nonzero_indices, nonzero_values):
            word = feature_names[idx]
            analysis[word] = value
        
        return analysis

# Example usage
texts = [
    "The quick brown fox jumps over the lazy dog",
    "A quick brown dog jumps over the lazy fox",
    "The lazy fox sleeps while the quick dog runs"
]

bow = BagOfWords(max_features=20)
bow_matrix = bow.fit_transform(texts)
feature_names = bow.get_feature_names()

print("Bag of Words Matrix:")
print("Features:", feature_names)
print("Matrix shape:", bow_matrix.shape)
print("Matrix:")
print(bow_matrix.toarray())
```

## Term Frequency (TF)

### TF Calculation Methods

```python
def calculate_tf(word: str, document: str, method: str = 'raw') -> float:
    """
    Calculate term frequency using different methods.
    
    Args:
        word: Target word
        document: Document text
        method: TF calculation method ('raw', 'log', 'double_log', 'normalized')
    """
    # Tokenize document
    words = document.lower().split()
    total_words = len(words)
    
    if total_words == 0:
        return 0.0
    
    # Count word frequency
    word_count = words.count(word.lower())
    
    if method == 'raw':
        return word_count
    elif method == 'log':
        return 1 + np.log(word_count) if word_count > 0 else 0
    elif method == 'double_log':
        return 1 + np.log(1 + word_count)
    elif method == 'normalized':
        return word_count / total_words
    else:
        raise ValueError(f"Unknown method: {method}")

def calculate_tf_all_words(document: str, method: str = 'raw') -> dict:
    """
    Calculate TF for all words in a document.
    """
    words = document.lower().split()
    word_counts = Counter(words)
    total_words = len(words)
    
    tf_scores = {}
    for word, count in word_counts.items():
        if method == 'raw':
            tf_scores[word] = count
        elif method == 'log':
            tf_scores[word] = 1 + np.log(count) if count > 0 else 0
        elif method == 'double_log':
            tf_scores[word] = 1 + np.log(1 + count)
        elif method == 'normalized':
            tf_scores[word] = count / total_words
    
    return tf_scores

# Example usage
document = "The quick brown fox jumps over the lazy dog. The fox is quick."
tf_scores = calculate_tf_all_words(document, method='normalized')
print("Term Frequencies (Normalized):")
for word, score in sorted(tf_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"  {word}: {score:.3f}")
```

## Inverse Document Frequency (IDF)

### IDF Calculation

```python
def calculate_idf(word: str, documents: list, method: str = 'standard') -> float:
    """
    Calculate inverse document frequency.
    
    Args:
        word: Target word
        documents: List of document texts
        method: IDF calculation method ('standard', 'smooth', 'max')
    """
    total_docs = len(documents)
    docs_with_word = sum(1 for doc in documents if word.lower() in doc.lower().split())
    
    if docs_with_word == 0:
        return 0.0
    
    if method == 'standard':
        return np.log(total_docs / docs_with_word)
    elif method == 'smooth':
        return np.log((total_docs + 1) / (docs_with_word + 1))
    elif method == 'max':
        return np.log(total_docs / docs_with_word)
    else:
        raise ValueError(f"Unknown method: {method}")

def calculate_idf_all_words(documents: list, method: str = 'standard') -> dict:
    """
    Calculate IDF for all unique words in documents.
    """
    # Get all unique words
    all_words = set()
    for doc in documents:
        words = doc.lower().split()
        all_words.update(words)
    
    # Calculate IDF for each word
    idf_scores = {}
    for word in all_words:
        idf_scores[word] = calculate_idf(word, documents, method)
    
    return idf_scores

# Example usage
documents = [
    "The quick brown fox jumps over the lazy dog",
    "A quick brown dog jumps over the lazy fox",
    "The lazy fox sleeps while the quick dog runs",
    "A document about machine learning algorithms",
    "Machine learning is a subset of artificial intelligence"
]

idf_scores = calculate_idf_all_words(documents, method='standard')
print("Inverse Document Frequencies:")
for word, score in sorted(idf_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"  {word}: {score:.3f}")
```

## TF-IDF

### Manual TF-IDF Implementation

```python
class TFIDFVectorizer:
    """
    Manual TF-IDF implementation.
    """
    
    def __init__(self, 
                 tf_method: str = 'normalized',
                 idf_method: str = 'standard',
                 max_features: int = None):
        """
        Initialize TF-IDF vectorizer.
        
        Args:
            tf_method: TF calculation method
            idf_method: IDF calculation method
            max_features: Maximum number of features
        """
        self.tf_method = tf_method
        self.idf_method = idf_method
        self.max_features = max_features
        self.vocabulary = None
        self.idf_scores = None
        self.feature_names = None
    
    def fit(self, documents: list) -> 'TFIDFVectorizer':
        """
        Fit the vectorizer on training documents.
        """
        # Calculate IDF scores
        self.idf_scores = calculate_idf_all_words(documents, self.idf_method)
        
        # Create vocabulary
        all_words = set()
        for doc in documents:
            words = doc.lower().split()
            all_words.update(words)
        
        # Limit vocabulary if max_features is specified
        if self.max_features and len(all_words) > self.max_features:
            # Sort by IDF score and take top features
            sorted_words = sorted(all_words, key=lambda w: self.idf_scores.get(w, 0), reverse=True)
            all_words = set(sorted_words[:self.max_features])
        
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(all_words))}
        self.feature_names = list(self.vocabulary.keys())
        
        return self
    
    def transform(self, documents: list) -> np.ndarray:
        """
        Transform documents to TF-IDF vectors.
        """
        if self.vocabulary is None:
            raise ValueError("Vectorizer must be fitted before transform")
        
        n_docs = len(documents)
        n_features = len(self.vocabulary)
        
        # Initialize matrix
        tfidf_matrix = np.zeros((n_docs, n_features))
        
        for doc_idx, document in enumerate(documents):
            # Calculate TF scores
            tf_scores = calculate_tf_all_words(document, self.tf_method)
            
            # Calculate TF-IDF scores
            for word, tf_score in tf_scores.items():
                if word in self.vocabulary:
                    feature_idx = self.vocabulary[word]
                    idf_score = self.idf_scores.get(word, 0)
                    tfidf_matrix[doc_idx, feature_idx] = tf_score * idf_score
        
        return tfidf_matrix
    
    def fit_transform(self, documents: list) -> np.ndarray:
        """
        Fit and transform documents.
        """
        return self.fit(documents).transform(documents)
    
    def get_feature_names(self) -> list:
        """
        Get feature names.
        """
        return self.feature_names
    
    def get_idf_scores(self) -> dict:
        """
        Get IDF scores.
        """
        return self.idf_scores

# Example usage
tfidf = TFIDFVectorizer(tf_method='normalized', max_features=20)
tfidf_matrix = tfidf.fit_transform(documents)
feature_names = tfidf.get_feature_names()

print("TF-IDF Matrix:")
print("Features:", feature_names)
print("Matrix shape:", tfidf_matrix.shape)
print("Matrix:")
print(tfidf_matrix)
```

### Using Scikit-learn TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_sklearn(documents: list, 
                        max_features: int = 1000,
                        min_df: int = 1,
                        max_df: float = 1.0,
                        ngram_range: tuple = (1, 1),
                        stop_words: str = 'english') -> tuple:
    """
    Create TF-IDF vectors using scikit-learn.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        stop_words=stop_words,
        lowercase=True
    )
    
    # Fit and transform
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    
    return tfidf_matrix, feature_names, vectorizer

# Example usage
tfidf_matrix, feature_names, vectorizer = create_tfidf_sklearn(
    documents, 
    max_features=20,
    ngram_range=(1, 2)
)

print("Scikit-learn TF-IDF Matrix:")
print("Features:", feature_names.tolist())
print("Matrix shape:", tfidf_matrix.shape)
print("Matrix:")
print(tfidf_matrix.toarray())
```

## Implementation Examples

### Document Similarity

```python
from sklearn.metrics.pairwise import cosine_similarity

def calculate_document_similarity(documents: list, method: str = 'tfidf') -> np.ndarray:
    """
    Calculate document similarity using different methods.
    
    Args:
        documents: List of document texts
        method: Similarity method ('bow', 'tfidf')
    """
    if method == 'bow':
        vectorizer = CountVectorizer(max_features=1000, stop_words='english')
        vectors = vectorizer.fit_transform(documents)
    elif method == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        vectors = vectorizer.fit_transform(documents)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(vectors)
    
    return similarity_matrix

# Example usage
similarity_matrix = calculate_document_similarity(documents, method='tfidf')
print("Document Similarity Matrix:")
print(similarity_matrix)
```

### Text Classification

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

def classify_texts_tfidf(texts: list, labels: list) -> dict:
    """
    Perform text classification using TF-IDF features.
    """
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    X = vectorizer.fit_transform(texts)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Train classifier
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = classifier.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'vectorizer': vectorizer,
        'classifier': classifier
    }

# Example usage
texts = [
    "Python programming tutorial for beginners",
    "Delicious recipe for chocolate cake",
    "Latest news about climate change",
    "Movie review: The new action film",
    "Health tips for better sleep",
    "JavaScript web development guide",
    "Easy pasta cooking instructions",
    "Sports news: Football match results",
    "Book review: Science fiction novel",
    "Fitness workout routine for beginners"
]

labels = [
    'technology', 'food', 'news', 'entertainment', 'health',
    'technology', 'food', 'news', 'entertainment', 'health'
]

results = classify_texts_tfidf(texts, labels)
print(f"Classification Accuracy: {results['accuracy']:.3f}")
```

### Keyword Extraction

```python
def extract_keywords_tfidf(document: str, documents: list, top_k: int = 10) -> list:
    """
    Extract keywords from a document using TF-IDF scores.
    """
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    # Add the target document to the corpus
    all_docs = documents + [document]
    tfidf_matrix = vectorizer.fit_transform(all_docs)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get TF-IDF scores for the target document
    doc_scores = tfidf_matrix[-1].toarray()[0]
    
    # Create word-score pairs and sort
    word_scores = list(zip(feature_names, doc_scores))
    word_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top keywords
    keywords = [word for word, score in word_scores[:top_k] if score > 0]
    
    return keywords

# Example usage
target_doc = "Machine learning algorithms process large amounts of data to find patterns."
keywords = extract_keywords_tfidf(target_doc, documents, top_k=5)
print("Extracted Keywords:", keywords)
```

## Advanced Techniques

### N-gram Features

```python
def create_ngram_features(texts: list, ngram_range: tuple = (1, 3)) -> tuple:
    """
    Create n-gram features using TF-IDF.
    """
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=ngram_range,
        stop_words='english'
    )
    
    features = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    return features, feature_names

# Example usage
ngram_features, ngram_names = create_ngram_features(documents, ngram_range=(1, 2))
print("N-gram Features:")
print("Feature names:", ngram_names[:10])  # Show first 10 features
print("Matrix shape:", ngram_features.shape)
```

### Sublinear TF Scaling

```python
def create_sublinear_tfidf(texts: list) -> tuple:
    """
    Create TF-IDF with sublinear TF scaling.
    """
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        sublinear_tf=True  # Apply sublinear TF scaling
    )
    
    features = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    return features, feature_names

# Example usage
sublinear_features, sublinear_names = create_sublinear_tfidf(documents)
print("Sublinear TF-IDF Matrix Shape:", sublinear_features.shape)
```

## Applications

### Information Retrieval

```python
def search_documents(query: str, documents: list, top_k: int = 5) -> list:
    """
    Search documents using TF-IDF similarity.
    """
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    doc_vectors = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])
    
    # Calculate similarities
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    
    # Get top matches
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0:
            results.append({
                'document': documents[idx],
                'similarity': similarities[idx],
                'index': idx
            })
    
    return results

# Example usage
query = "quick fox"
search_results = search_documents(query, documents, top_k=3)
print("Search Results:")
for result in search_results:
    print(f"  Similarity: {result['similarity']:.3f}")
    print(f"  Document: {result['document']}")
    print()
```

### Document Clustering

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def cluster_documents_tfidf(documents: list, n_clusters: int = 3) -> dict:
    """
    Cluster documents using TF-IDF features.
    """
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english'
    )
    
    features = vectorizer.fit_transform(documents)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features)
    
    # Get cluster centers
    cluster_centers = kmeans.cluster_centers_
    
    # Get top words for each cluster
    feature_names = vectorizer.get_feature_names_out()
    top_words_per_cluster = []
    
    for center in cluster_centers:
        top_indices = center.argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        top_words_per_cluster.append(top_words)
    
    return {
        'clusters': clusters,
        'cluster_centers': cluster_centers,
        'top_words_per_cluster': top_words_per_cluster,
        'kmeans': kmeans,
        'vectorizer': vectorizer
    }

# Example usage
clustering_results = cluster_documents_tfidf(documents, n_clusters=3)
print("Document Clustering Results:")
for i, top_words in enumerate(clustering_results['top_words_per_cluster']):
    print(f"  Cluster {i+1} top words: {', '.join(top_words)}")
```

## Best Practices

### Feature Selection

```python
def select_important_features(tfidf_matrix, feature_names, threshold: float = 0.1) -> tuple:
    """
    Select important features based on variance.
    """
    from sklearn.feature_selection import VarianceThreshold
    
    # Remove low-variance features
    selector = VarianceThreshold(threshold=threshold)
    selected_features = selector.fit_transform(tfidf_matrix)
    
    # Get selected feature names
    selected_indices = selector.get_support()
    selected_names = feature_names[selected_indices]
    
    return selected_features, selected_names

# Example usage
tfidf_matrix, feature_names, _ = create_tfidf_sklearn(documents)
selected_features, selected_names = select_important_features(
    tfidf_matrix, feature_names, threshold=0.01
)
print(f"Selected {len(selected_names)} features out of {len(feature_names)}")
```

### Dimensionality Reduction

```python
def reduce_dimensions_tfidf(tfidf_matrix, n_components: int = 100) -> tuple:
    """
    Reduce TF-IDF dimensions using PCA.
    """
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(tfidf_matrix.toarray())
    
    return reduced_features, pca

# Example usage
tfidf_matrix, _, _ = create_tfidf_sklearn(documents)
reduced_features, pca = reduce_dimensions_tfidf(tfidf_matrix, n_components=50)
print(f"Reduced from {tfidf_matrix.shape[1]} to {reduced_features.shape[1]} dimensions")
print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
```

## Limitations and Alternatives

### Limitations of BoW and TF-IDF

1. **No Semantic Understanding**: Doesn't capture word meanings
2. **Sparse Representations**: High-dimensional sparse vectors
3. **No Word Order**: Ignores word sequence information
4. **Vocabulary Size**: Can become very large
5. **Out-of-Vocabulary**: Can't handle new words

### Modern Alternatives

```python
# Word Embeddings (Word2Vec, GloVe)
from gensim.models import Word2Vec

def create_word_embeddings(texts: list) -> Word2Vec:
    """
    Create word embeddings using Word2Vec.
    """
    # Tokenize texts
    tokenized_texts = [text.lower().split() for text in texts]
    
    # Train Word2Vec model
    model = Word2Vec(
        tokenized_texts,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4
    )
    
    return model

# Example usage
w2v_model = create_word_embeddings(documents)
print("Word2Vec model vocabulary size:", len(w2v_model.wv.key_to_index))
```

### Comparison Summary

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Bag of Words** | Simple, interpretable | No semantic meaning, sparse | Basic text classification |
| **TF-IDF** | Weights important words | Still sparse, no semantics | Information retrieval |
| **Word Embeddings** | Semantic meaning, dense | Requires training | Semantic similarity |
| **BERT Embeddings** | Contextual, powerful | Computationally expensive | Advanced NLP tasks |

## Conclusion

Bag of Words and TF-IDF remain fundamental techniques in NLP, providing a solid foundation for text analysis and machine learning applications.

### Key Takeaways

- **BoW**: Simple frequency-based representation
- **TF-IDF**: Weights words by importance across documents
- **Applications**: Classification, similarity, information retrieval
- **Limitations**: No semantic understanding, sparse representations
- **Alternatives**: Word embeddings, contextual embeddings

### When to Use

- **Use BoW/TF-IDF for**: Simple text classification, information retrieval, baseline models
- **Use Word Embeddings for**: Semantic similarity, advanced NLP tasks
- **Use BERT for**: State-of-the-art performance, contextual understanding

---

**Happy Text Processing!** 