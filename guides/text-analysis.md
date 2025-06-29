# Text Analysis in Natural Language Processing

A comprehensive guide to text analysis techniques, from basic statistics to advanced linguistic analysis.

## Table of Contents
1. [Introduction](#introduction)
2. [Basic Text Statistics](#basic-text-statistics)
3. [Linguistic Analysis](#linguistic-analysis)
4. [Sentiment Analysis](#sentiment-analysis)
5. [Topic Modeling](#topic-modeling)
6. [Text Similarity](#text-similarity)
7. [Named Entity Recognition](#named-entity-recognition)
8. [Text Classification](#text-classification)
9. [Advanced Analysis](#advanced-analysis)
10. [Implementation Examples](#implementation-examples)

## Introduction

Text analysis is the process of extracting meaningful information and insights from text data. It encompasses a wide range of techniques from simple statistical analysis to complex linguistic understanding.

### Why Text Analysis Matters
- **Data Understanding**: Gain insights from large text collections
- **Information Extraction**: Extract structured data from unstructured text
- **Content Organization**: Categorize and organize text content
- **Trend Analysis**: Identify patterns and trends in text data
- **Quality Assessment**: Evaluate text quality and characteristics

## Basic Text Statistics

### Character-Level Statistics

```python
def analyze_characters(text: str) -> dict:
    """
    Analyze character-level statistics.
    """
    stats = {
        'total_chars': len(text),
        'alphabetic_chars': sum(c.isalpha() for c in text),
        'numeric_chars': sum(c.isdigit() for c in text),
        'whitespace_chars': sum(c.isspace() for c in text),
        'punctuation_chars': sum(c in string.punctuation for c in text),
        'uppercase_chars': sum(c.isupper() for c in text),
        'lowercase_chars': sum(c.islower() for c in text),
        'unique_chars': len(set(text)),
        'char_diversity': len(set(text)) / len(text) if text else 0
    }
    
    return stats

# Example usage
text = "Hello, World! This is a test. 123"
char_stats = analyze_characters(text)
print("Character Statistics:")
for key, value in char_stats.items():
    print(f"  {key}: {value}")
```

### Word-Level Statistics

```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter

def analyze_words(text: str) -> dict:
    """
    Analyze word-level statistics.
    """
    # Tokenize
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    
    # Filter out punctuation
    words = [word for word in words if word.isalpha()]
    
    stats = {
        'total_words': len(words),
        'unique_words': len(set(words)),
        'total_sentences': len(sentences),
        'avg_words_per_sentence': len(words) / len(sentences) if sentences else 0,
        'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
        'vocabulary_diversity': len(set(words)) / len(words) if words else 0,
        'word_frequencies': Counter(words).most_common(10)
    }
    
    return stats

# Example usage
text = "Hello world! This is a test sentence. It contains multiple words."
word_stats = analyze_words(text)
print("Word Statistics:")
for key, value in word_stats.items():
    if key != 'word_frequencies':
        print(f"  {key}: {value}")
print("  Most common words:", word_stats['word_frequencies'])
```

### Readability Metrics

```python
def calculate_readability(text: str) -> dict:
    """
    Calculate various readability metrics.
    """
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha()]
    
    # Count syllables (simplified)
    def count_syllables(word):
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
        return count
    
    total_syllables = sum(count_syllables(word) for word in words)
    
    # Calculate metrics
    avg_sentence_length = len(words) / len(sentences) if sentences else 0
    avg_syllables_per_word = total_syllables / len(words) if words else 0
    
    # Flesch Reading Ease
    flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    flesch_score = max(0, min(100, flesch_score))
    
    # Flesch-Kincaid Grade Level
    fk_grade = 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59
    
    # Gunning Fog Index
    complex_words = sum(1 for word in words if count_syllables(word) > 2)
    fog_index = 0.4 * (avg_sentence_length + (100 * complex_words / len(words))) if words else 0
    
    return {
        'flesch_reading_ease': flesch_score,
        'flesch_kincaid_grade': fk_grade,
        'gunning_fog_index': fog_index,
        'avg_sentence_length': avg_sentence_length,
        'avg_syllables_per_word': avg_syllables_per_word,
        'complex_word_ratio': complex_words / len(words) if words else 0
    }

# Example usage
text = "This is a sample text for readability analysis. It contains multiple sentences with varying complexity."
readability = calculate_readability(text)
print("Readability Metrics:")
for key, value in readability.items():
    print(f"  {key}: {value:.2f}")
```

## Linguistic Analysis

### Part-of-Speech Analysis

```python
import nltk
from nltk import pos_tag

def analyze_pos(text: str) -> dict:
    """
    Analyze part-of-speech distribution.
    """
    # Download required data
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')
    
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    
    # Count POS tags
    pos_counts = Counter(tag for word, tag in pos_tags)
    
    # Group by major categories
    pos_categories = {
        'nouns': sum(pos_counts.get(tag, 0) for tag in ['NN', 'NNS', 'NNP', 'NNPS']),
        'verbs': sum(pos_counts.get(tag, 0) for tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']),
        'adjectives': sum(pos_counts.get(tag, 0) for tag in ['JJ', 'JJR', 'JJS']),
        'adverbs': sum(pos_counts.get(tag, 0) for tag in ['RB', 'RBR', 'RBS']),
        'pronouns': sum(pos_counts.get(tag, 0) for tag in ['PRP', 'PRP$']),
        'determiners': sum(pos_counts.get(tag, 0) for tag in ['DT']),
        'prepositions': sum(pos_counts.get(tag, 0) for tag in ['IN']),
        'conjunctions': sum(pos_counts.get(tag, 0) for tag in ['CC'])
    }
    
    return {
        'pos_distribution': pos_categories,
        'detailed_pos': dict(pos_counts),
        'total_words': len(words)
    }

# Example usage
text = "The quick brown fox jumps over the lazy dog."
pos_analysis = analyze_pos(text)
print("POS Analysis:")
print("  Distribution:", pos_analysis['pos_distribution'])
print("  Total words:", pos_analysis['total_words'])
```

### Named Entity Recognition

```python
def analyze_named_entities(text: str) -> dict:
    """
    Analyze named entities in text.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('chunkers/maxent_ne_chunker')
    except LookupError:
        nltk.download('maxent_ne_chunker')
    
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('words')
    
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    ne_tree = nltk.ne_chunk(pos_tags)
    
    # Extract named entities
    entities = []
    entity_counts = Counter()
    
    for chunk in ne_tree:
        if hasattr(chunk, 'label'):
            entity_text = ' '.join(c[0] for c in chunk)
            entity_type = chunk.label()
            entities.append((entity_text, entity_type))
            entity_counts[entity_type] += 1
    
    return {
        'entities': entities,
        'entity_counts': dict(entity_counts),
        'total_entities': len(entities)
    }

# Example usage
text = "John Smith works at Google in New York City."
ne_analysis = analyze_named_entities(text)
print("Named Entity Analysis:")
print("  Entities:", ne_analysis['entities'])
print("  Entity counts:", ne_analysis['entity_counts'])
```

## Sentiment Analysis

### Lexicon-Based Sentiment Analysis

```python
def analyze_sentiment_lexicon(text: str) -> dict:
    """
    Perform lexicon-based sentiment analysis.
    """
    # Simple sentiment lexicons (in practice, use comprehensive lexicons)
    positive_words = {
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
        'love', 'like', 'happy', 'joy', 'beautiful', 'perfect', 'best'
    }
    
    negative_words = {
        'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike',
        'sad', 'angry', 'frustrated', 'disappointed', 'worst', 'ugly'
    }
    
    # Tokenize and normalize
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha()]
    
    # Count sentiment words
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    # Calculate sentiment score
    total_words = len(words)
    if total_words == 0:
        sentiment_score = 0
    else:
        sentiment_score = (positive_count - negative_count) / total_words
    
    # Determine sentiment
    if sentiment_score > 0.05:
        sentiment = 'positive'
    elif sentiment_score < -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return {
        'sentiment': sentiment,
        'sentiment_score': sentiment_score,
        'positive_words': positive_count,
        'negative_words': negative_count,
        'total_words': total_words
    }

# Example usage
text = "I love this product! It's amazing and works perfectly."
sentiment = analyze_sentiment_lexicon(text)
print("Sentiment Analysis:")
for key, value in sentiment.items():
    print(f"  {key}: {value}")
```

### Advanced Sentiment Analysis with VADER

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_sentiment_vader(text: str) -> dict:
    """
    Perform sentiment analysis using VADER.
    """
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    
    # Determine sentiment
    compound_score = scores['compound']
    if compound_score >= 0.05:
        sentiment = 'positive'
    elif compound_score <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return {
        'sentiment': sentiment,
        'compound_score': compound_score,
        'positive_score': scores['pos'],
        'negative_score': scores['neg'],
        'neutral_score': scores['neu']
    }

# Example usage
text = "This movie is absolutely fantastic! I love it!"
vader_sentiment = analyze_sentiment_vader(text)
print("VADER Sentiment Analysis:")
for key, value in vader_sentiment.items():
    print(f"  {key}: {value}")
```

## Topic Modeling

### Latent Dirichlet Allocation (LDA)

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

def perform_lda_analysis(texts: list, n_topics: int = 5, n_top_words: int = 10) -> dict:
    """
    Perform LDA topic modeling.
    """
    # Create document-term matrix
    vectorizer = CountVectorizer(
        max_features=1000,
        stop_words='english',
        min_df=2,
        max_df=0.95
    )
    
    doc_term_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Perform LDA
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=100
    )
    
    lda_output = lda.fit_transform(doc_term_matrix)
    
    # Get topic words
    topic_words = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topic_words.append(top_words)
    
    # Get document-topic distribution
    doc_topics = lda_output.tolist()
    
    return {
        'topic_words': topic_words,
        'doc_topics': doc_topics,
        'feature_names': feature_names.tolist(),
        'lda_model': lda
    }

# Example usage
texts = [
    "Machine learning algorithms are used in artificial intelligence.",
    "Deep learning neural networks process large amounts of data.",
    "Natural language processing helps computers understand text.",
    "Computer vision algorithms analyze images and videos.",
    "Data science involves statistics and machine learning."
]

lda_results = perform_lda_analysis(texts, n_topics=3)
print("LDA Topic Modeling:")
for i, topic_words in enumerate(lda_results['topic_words']):
    print(f"  Topic {i+1}: {', '.join(topic_words)}")
```

## Text Similarity

### Cosine Similarity

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_text_similarity(texts: list) -> np.ndarray:
    """
    Calculate cosine similarity between texts.
    """
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    return similarity_matrix

# Example usage
texts = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks for pattern recognition.",
    "Natural language processing helps computers understand text.",
    "Computer vision processes visual information from images."
]

similarity_matrix = calculate_text_similarity(texts)
print("Text Similarity Matrix:")
print(similarity_matrix)
```

### Jaccard Similarity

```python
def calculate_jaccard_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity between two texts.
    """
    # Tokenize and create sets
    words1 = set(word_tokenize(text1.lower()))
    words2 = set(word_tokenize(text2.lower()))
    
    # Remove non-alphabetic tokens
    words1 = {word for word in words1 if word.isalpha()}
    words2 = {word for word in words2 if word.isalpha()}
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0

# Example usage
text1 = "Machine learning algorithms process data"
text2 = "Deep learning neural networks analyze information"
jaccard_sim = calculate_jaccard_similarity(text1, text2)
print(f"Jaccard Similarity: {jaccard_sim:.3f}")
```

## Text Classification

### Feature Extraction

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

def classify_texts(texts: list, labels: list) -> dict:
    """
    Perform text classification using TF-IDF and Naive Bayes.
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

classification_results = classify_texts(texts, labels)
print(f"Classification Accuracy: {classification_results['accuracy']:.3f}")
```

## Advanced Analysis

### Text Complexity Analysis

```python
def analyze_text_complexity(text: str) -> dict:
    """
    Analyze text complexity using multiple metrics.
    """
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha()]
    
    # Basic statistics
    avg_sentence_length = len(words) / len(sentences) if sentences else 0
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    
    # Vocabulary richness
    unique_words = set(words)
    type_token_ratio = len(unique_words) / len(words) if words else 0
    
    # Long word ratio (words with more than 6 characters)
    long_words = [word for word in words if len(word) > 6]
    long_word_ratio = len(long_words) / len(words) if words else 0
    
    # Complex word ratio (words with more than 2 syllables)
    def count_syllables(word):
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
        return count
    
    complex_words = [word for word in words if count_syllables(word) > 2]
    complex_word_ratio = len(complex_words) / len(words) if words else 0
    
    # Calculate overall complexity score
    complexity_score = (
        0.3 * (avg_sentence_length / 20) +  # Normalize sentence length
        0.2 * (avg_word_length / 5) +       # Normalize word length
        0.2 * (1 - type_token_ratio) +      # Vocabulary diversity (inverted)
        0.15 * long_word_ratio +            # Long word ratio
        0.15 * complex_word_ratio           # Complex word ratio
    )
    
    return {
        'avg_sentence_length': avg_sentence_length,
        'avg_word_length': avg_word_length,
        'type_token_ratio': type_token_ratio,
        'long_word_ratio': long_word_ratio,
        'complex_word_ratio': complex_word_ratio,
        'complexity_score': complexity_score,
        'complexity_level': 'high' if complexity_score > 0.6 else 'medium' if complexity_score > 0.3 else 'low'
    }

# Example usage
text = "The sophisticated implementation of advanced algorithms necessitates comprehensive understanding of complex mathematical principles."
complexity = analyze_text_complexity(text)
print("Text Complexity Analysis:")
for key, value in complexity.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.3f}")
    else:
        print(f"  {key}: {value}")
```

## Implementation Examples

### Complete Text Analysis Pipeline

```python
class TextAnalyzer:
    """
    Comprehensive text analysis pipeline.
    """
    
    def __init__(self):
        """Initialize text analyzer."""
        self.vader_analyzer = SentimentIntensityAnalyzer()
    
    def analyze_text(self, text: str) -> dict:
        """
        Perform comprehensive text analysis.
        """
        analysis = {}
        
        # Basic statistics
        analysis['character_stats'] = analyze_characters(text)
        analysis['word_stats'] = analyze_words(text)
        analysis['readability'] = calculate_readability(text)
        
        # Linguistic analysis
        analysis['pos_analysis'] = analyze_pos(text)
        analysis['named_entities'] = analyze_named_entities(text)
        
        # Sentiment analysis
        analysis['sentiment_lexicon'] = analyze_sentiment_lexicon(text)
        analysis['sentiment_vader'] = analyze_sentiment_vader(text)
        
        # Complexity analysis
        analysis['complexity'] = analyze_text_complexity(text)
        
        return analysis
    
    def generate_report(self, text: str) -> str:
        """
        Generate a comprehensive text analysis report.
        """
        analysis = self.analyze_text(text)
        
        report = f"""
TEXT ANALYSIS REPORT
{'='*50}

BASIC STATISTICS:
- Total characters: {analysis['character_stats']['total_chars']}
- Total words: {analysis['word_stats']['total_words']}
- Total sentences: {analysis['word_stats']['total_sentences']}
- Average words per sentence: {analysis['word_stats']['avg_words_per_sentence']:.2f}
- Average word length: {analysis['word_stats']['avg_word_length']:.2f}

READABILITY:
- Flesch Reading Ease: {analysis['readability']['flesch_reading_ease']:.1f}
- Flesch-Kincaid Grade: {analysis['readability']['flesch_kincaid_grade']:.1f}
- Gunning Fog Index: {analysis['readability']['gunning_fog_index']:.1f}

LINGUISTIC ANALYSIS:
- Nouns: {analysis['pos_analysis']['pos_distribution']['nouns']}
- Verbs: {analysis['pos_analysis']['pos_distribution']['verbs']}
- Adjectives: {analysis['pos_analysis']['pos_distribution']['adjectives']}
- Named entities: {analysis['named_entities']['total_entities']}

SENTIMENT ANALYSIS:
- Lexicon-based sentiment: {analysis['sentiment_lexicon']['sentiment']}
- VADER sentiment: {analysis['sentiment_vader']['sentiment']}
- VADER compound score: {analysis['sentiment_vader']['compound_score']:.3f}

COMPLEXITY:
- Complexity level: {analysis['complexity']['complexity_level']}
- Complexity score: {analysis['complexity']['complexity_score']:.3f}
        """
        
        return report

# Example usage
analyzer = TextAnalyzer()
sample_text = """
Natural language processing (NLP) is a subfield of linguistics, computer science, 
and artificial intelligence concerned with the interactions between computers and 
human language, in particular how to program computers to process and analyze 
large amounts of natural language data.
"""

report = analyzer.generate_report(sample_text)
print(report)
```

## Conclusion

Text analysis provides powerful tools for understanding and extracting insights from text data. The techniques covered in this guide range from simple statistical analysis to advanced linguistic understanding.

### Key Takeaways

- **Start Simple**: Begin with basic statistics and gradually add complexity
- **Choose Appropriate Metrics**: Select analysis methods based on your goals
- **Validate Results**: Always validate analysis results with domain knowledge
- **Consider Context**: Text analysis should consider the context and domain
- **Combine Methods**: Use multiple analysis techniques for comprehensive understanding

### Next Steps

1. **Explore Advanced Techniques**: Dive deeper into specific analysis methods
2. **Apply to Your Data**: Use these techniques on your own text data
3. **Customize Analysis**: Adapt methods to your specific use case
4. **Evaluate Performance**: Measure the effectiveness of your analysis
5. **Iterate and Improve**: Continuously refine your analysis approach

---

**Happy Analyzing!** 