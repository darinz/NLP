# Named Entity Recognition (NER) in Natural Language Processing

A comprehensive guide to Named Entity Recognition, from basic concepts to advanced implementations.

## Table of Contents
1. [Introduction](#introduction)
2. [Types of Named Entities](#types-of-named-entities)
3. [NER Approaches](#ner-approaches)
4. [Rule-Based NER](#rule-based-ner)
5. [Machine Learning NER](#machine-learning-ner)
6. [Deep Learning NER](#deep-learning-ner)
7. [Implementation Examples](#implementation-examples)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Advanced Techniques](#advanced-techniques)
10. [Applications](#applications)

## Introduction

Named Entity Recognition (NER) is a subtask of information extraction that identifies and classifies named entities in text into predefined categories such as person names, organizations, locations, dates, and more.

### Why NER Matters
- **Information Extraction**: Extract structured data from unstructured text
- **Knowledge Graphs**: Build knowledge bases and relationships
- **Search and Retrieval**: Improve search functionality
- **Question Answering**: Identify entities in questions and answers
- **Document Classification**: Categorize documents based on entities

## Types of Named Entities

### Standard Entity Types

```python
# Common NER entity types
ENTITY_TYPES = {
    'PERSON': 'Names of people',
    'ORGANIZATION': 'Companies, institutions, agencies',
    'LOCATION': 'Countries, cities, geographical locations',
    'DATE': 'Dates, times, periods',
    'MONEY': 'Monetary values',
    'PERCENT': 'Percentages',
    'TIME': 'Time expressions',
    'FACILITY': 'Buildings, airports, highways',
    'GPE': 'Countries, cities, states',
    'PRODUCT': 'Products, brands',
    'EVENT': 'Named events',
    'WORK_OF_ART': 'Titles of books, songs, etc.',
    'LAW': 'Named documents, laws',
    'LANGUAGE': 'Named languages',
    'QUANTITY': 'Measurements, amounts'
}
```

### Custom Entity Types

```python
# Domain-specific entity types
DOMAIN_ENTITIES = {
    'MEDICAL': {
        'DISEASE': 'Medical conditions and diseases',
        'DRUG': 'Medications and drugs',
        'SYMPTOM': 'Medical symptoms',
        'TREATMENT': 'Medical procedures and treatments'
    },
    'LEGAL': {
        'COURT': 'Court names and jurisdictions',
        'CASE': 'Legal case names',
        'STATUTE': 'Laws and regulations',
        'PARTY': 'Legal parties involved'
    },
    'FINANCIAL': {
        'STOCK': 'Stock symbols and companies',
        'CURRENCY': 'Currency names and codes',
        'ACCOUNT': 'Account numbers and identifiers',
        'TRANSACTION': 'Financial transactions'
    }
}
```

## NER Approaches

### 1. Rule-Based NER
- **Pattern Matching**: Use regular expressions and linguistic patterns
- **Dictionary Lookup**: Match against predefined entity lists
- **Pros**: Fast, interpretable, no training data needed
- **Cons**: Limited coverage, requires manual maintenance

### 2. Machine Learning NER
- **Supervised Learning**: Train on labeled data
- **Sequence Labeling**: Treat as sequence labeling problem
- **Pros**: Better accuracy, learns from data
- **Cons**: Requires labeled training data

### 3. Deep Learning NER
- **Neural Networks**: Use RNNs, CNNs, or Transformers
- **Pre-trained Models**: Leverage BERT, RoBERTa, etc.
- **Pros**: State-of-the-art performance
- **Cons**: Computationally expensive, requires large datasets

## Rule-Based NER

### Basic Pattern Matching

```python
import re
from typing import List, Tuple, Dict

class RuleBasedNER:
    """
    Basic rule-based Named Entity Recognition.
    """
    
    def __init__(self):
        """Initialize rule-based NER."""
        self.patterns = self._create_patterns()
    
    def _create_patterns(self) -> Dict[str, List[str]]:
        """
        Create regex patterns for different entity types.
        """
        patterns = {
            'PERSON': [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
                r'\b[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+\b',  # First M. Last
                r'\b[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+\b'  # First Middle Last
            ],
            'ORGANIZATION': [
                r'\b[A-Z][a-z]+ (Inc|Corp|LLC|Ltd|Company|Organization)\b',
                r'\b[A-Z][a-z]+ [A-Z][a-z]+ (University|College|Institute)\b',
                r'\b[A-Z]+ [A-Z]+\b'  # Acronyms like IBM, NASA
            ],
            'LOCATION': [
                r'\b[A-Z][a-z]+, [A-Z]{2}\b',  # City, State
                r'\b[A-Z][a-z]+ (Street|Avenue|Road|Boulevard)\b',
                r'\b[A-Z][a-z]+ (Park|Mountain|River|Lake)\b'
            ],
            'DATE': [
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
                r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
                r'\b(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2},? \d{4}\b'
            ],
            'MONEY': [
                r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b',  # $1,234.56
                r'\b\d+(?:,\d{3})*(?:\.\d{2})? (dollars|USD)\b'
            ],
            'PERCENT': [
                r'\b\d+(?:\.\d+)?%\b'  # 25% or 25.5%
            ]
        }
        return patterns
    
    def extract_entities(self, text: str) -> List[Dict[str, any]]:
        """
        Extract entities from text using rule-based patterns.
        """
        entities = []
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    entity = {
                        'text': match.group(),
                        'type': entity_type,
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.8  # Rule-based confidence
                    }
                    entities.append(entity)
        
        # Remove overlapping entities (keep the longest)
        entities = self._remove_overlaps(entities)
        
        return entities
    
    def _remove_overlaps(self, entities: List[Dict]) -> List[Dict]:
        """
        Remove overlapping entities, keeping the longest ones.
        """
        if not entities:
            return entities
        
        # Sort by length (longest first)
        entities.sort(key=lambda x: len(x['text']), reverse=True)
        
        filtered_entities = []
        for entity in entities:
            # Check if this entity overlaps with any already selected entity
            overlaps = False
            for selected in filtered_entities:
                if (entity['start'] < selected['end'] and 
                    entity['end'] > selected['start']):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_entities.append(entity)
        
        # Sort by position in text
        filtered_entities.sort(key=lambda x: x['start'])
        
        return filtered_entities

# Example usage
ner = RuleBasedNER()
text = "John Smith works at IBM in New York, NY. He earns $75,000 per year and started on January 15, 2020."
entities = ner.extract_entities(text)

print("Rule-based NER Results:")
for entity in entities:
    print(f"  {entity['text']} ({entity['type']}) at position {entity['start']}-{entity['end']}")
```

### Dictionary-Based NER

```python
class DictionaryNER:
    """
    Dictionary-based Named Entity Recognition.
    """
    
    def __init__(self):
        """Initialize dictionary-based NER."""
        self.entity_dictionaries = self._load_dictionaries()
    
    def _load_dictionaries(self) -> Dict[str, set]:
        """
        Load entity dictionaries for different types.
        """
        dictionaries = {
            'PERSON': {
                'John Smith', 'Jane Doe', 'Michael Johnson', 'Sarah Wilson',
                'David Brown', 'Emily Davis', 'Robert Miller', 'Lisa Garcia'
            },
            'ORGANIZATION': {
                'Google', 'Microsoft', 'Apple', 'Amazon', 'Facebook',
                'IBM', 'Intel', 'Oracle', 'Cisco', 'Netflix'
            },
            'LOCATION': {
                'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
                'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose',
                'United States', 'Canada', 'United Kingdom', 'Germany', 'France'
            },
            'DISEASE': {
                'diabetes', 'cancer', 'heart disease', 'hypertension',
                'asthma', 'arthritis', 'depression', 'anxiety'
            }
        }
        return dictionaries
    
    def extract_entities(self, text: str) -> List[Dict[str, any]]:
        """
        Extract entities using dictionary lookup.
        """
        entities = []
        text_lower = text.lower()
        
        for entity_type, dictionary in self.entity_dictionaries.items():
            for entity_name in dictionary:
                # Case-insensitive search
                entity_lower = entity_name.lower()
                start = 0
                
                while True:
                    pos = text_lower.find(entity_lower, start)
                    if pos == -1:
                        break
                    
                    # Check word boundaries
                    if self._is_word_boundary(text, pos, len(entity_name)):
                        entity = {
                            'text': text[pos:pos + len(entity_name)],
                            'type': entity_type,
                            'start': pos,
                            'end': pos + len(entity_name),
                            'confidence': 0.9
                        }
                        entities.append(entity)
                    
                    start = pos + 1
        
        # Remove overlapping entities
        entities = self._remove_overlaps(entities)
        
        return entities
    
    def _is_word_boundary(self, text: str, start: int, length: int) -> bool:
        """
        Check if the entity is at a word boundary.
        """
        # Check start boundary
        if start > 0 and text[start - 1].isalnum():
            return False
        
        # Check end boundary
        end = start + length
        if end < len(text) and text[end].isalnum():
            return False
        
        return True
    
    def _remove_overlaps(self, entities: List[Dict]) -> List[Dict]:
        """
        Remove overlapping entities.
        """
        if not entities:
            return entities
        
        entities.sort(key=lambda x: len(x['text']), reverse=True)
        
        filtered_entities = []
        for entity in entities:
            overlaps = False
            for selected in filtered_entities:
                if (entity['start'] < selected['end'] and 
                    entity['end'] > selected['start']):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_entities.append(entity)
        
        filtered_entities.sort(key=lambda x: x['start'])
        return filtered_entities

# Example usage
dict_ner = DictionaryNER()
text = "John Smith works at Google in New York. He has diabetes and takes medication."
entities = dict_ner.extract_entities(text)

print("Dictionary-based NER Results:")
for entity in entities:
    print(f"  {entity['text']} ({entity['type']}) at position {entity['start']}-{entity['end']}")
```

## Machine Learning NER

### CRF-based NER

```python
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report
import nltk

class CRFNER:
    """
    Conditional Random Field-based Named Entity Recognition.
    """
    
    def __init__(self):
        """Initialize CRF NER."""
        self.crf = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        self.feature_extractor = None
    
    def word2features(self, sent: List[Tuple[str, str]], i: int) -> Dict[str, any]:
        """
        Extract features for a word in a sentence.
        """
        word = sent[i][0]
        postag = sent[i][1]
        
        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': postag,
            'postag[:2]': postag[:2],
        }
        
        if i > 0:
            word1 = sent[i-1][0]
            postag1 = sent[i-1][1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True
        
        if i < len(sent)-1:
            word1 = sent[i+1][0]
            postag1 = sent[i+1][1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
            })
        else:
            features['EOS'] = True
        
        return features
    
    def sent2features(self, sent: List[Tuple[str, str]]) -> List[Dict[str, any]]:
        """
        Extract features for all words in a sentence.
        """
        return [self.word2features(sent, i) for i in range(len(sent))]
    
    def sent2labels(self, sent: List[Tuple[str, str, str]]) -> List[str]:
        """
        Extract labels from a sentence.
        """
        return [label for token, postag, label in sent]
    
    def sent2tokens(self, sent: List[Tuple[str, str, str]]) -> List[str]:
        """
        Extract tokens from a sentence.
        """
        return [token for token, postag, label in sent]
    
    def train(self, training_data: List[List[Tuple[str, str, str]]]):
        """
        Train the CRF model.
        """
        X_train = [self.sent2features(sent) for sent in training_data]
        y_train = [self.sent2labels(sent) for sent in training_data]
        
        self.crf.fit(X_train, y_train)
    
    def predict(self, text: str) -> List[Dict[str, any]]:
        """
        Predict entities in text.
        """
        # Tokenize and POS tag
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        
        # Extract features
        features = self.sent2features(pos_tags)
        
        # Predict labels
        labels = self.crf.predict([features])[0]
        
        # Extract entities
        entities = []
        current_entity = None
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            if label.startswith('B-'):  # Beginning of entity
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'text': token,
                    'type': label[2:],
                    'start': sum(len(t) + 1 for t in tokens[:i]),
                    'end': sum(len(t) + 1 for t in tokens[:i+1]) - 1
                }
            elif label.startswith('I-') and current_entity and label[2:] == current_entity['type']:
                # Inside of entity
                current_entity['text'] += ' ' + token
                current_entity['end'] = sum(len(t) + 1 for t in tokens[:i+1]) - 1
            elif label == 'O':  # Outside of entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities

# Example training data (simplified)
training_data = [
    [('John', 'NNP', 'B-PERSON'), ('Smith', 'NNP', 'I-PERSON')],
    [('Google', 'NNP', 'B-ORGANIZATION')],
    [('New', 'NNP', 'B-LOCATION'), ('York', 'NNP', 'I-LOCATION')]
]

# Example usage (would need more training data in practice)
# crf_ner = CRFNER()
# crf_ner.train(training_data)
# entities = crf_ner.predict("John Smith works at Google in New York.")
```

## Deep Learning NER

### LSTM-CRF NER

```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader

class LSTMCRFNER(nn.Module):
    """
    LSTM-CRF model for Named Entity Recognition.
    """
    
    def __init__(self, 
                 vocab_size: int,
                 embedding_dim: int = 128,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 num_tags: int = 9,  # B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC, O
                 dropout: float = 0.1):
        """
        Initialize LSTM-CRF model.
        """
        super(LSTMCRFNER, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_tags = num_tags
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,  # Bidirectional
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        
        # CRF layer
        self.crf = CRF(num_tags)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def _get_lstm_features(self, input_ids: torch.Tensor, 
                          attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Get LSTM features for input sequence.
        """
        # Embedding
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        
        # Pack sequence
        lengths = attention_mask.sum(dim=1)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(packed)
        
        # Unpack sequence
        unpacked, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        # Output layer
        emissions = self.hidden2tag(unpacked)
        
        return emissions
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor,
                labels: torch.Tensor = None) -> dict:
        """
        Forward pass through the model.
        """
        emissions = self._get_lstm_features(input_ids, attention_mask)
        
        if labels is not None:
            # Training mode
            loss = self.crf(emissions, labels, mask=attention_mask.bool())
            return {'loss': loss}
        else:
            # Inference mode
            predictions = self.crf.decode(emissions, mask=attention_mask.bool())
            return {'predictions': predictions}

class CRF(nn.Module):
    """
    Conditional Random Field layer.
    """
    
    def __init__(self, num_tags: int):
        """
        Initialize CRF layer.
        """
        super(CRF, self).__init__()
        self.num_tags = num_tags
        
        # Transition matrix
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        
        # Start and end transitions
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
    
    def forward(self, emissions: torch.Tensor, 
                tags: torch.Tensor, 
                mask: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log likelihood loss.
        """
        # Forward algorithm
        forward_score = self._forward_alg(emissions, mask)
        
        # Score for given tags
        gold_score = self._score_sentence(emissions, tags, mask)
        
        return forward_score - gold_score
    
    def decode(self, emissions: torch.Tensor, 
               mask: torch.Tensor) -> List[List[int]]:
        """
        Viterbi decoding.
        """
        return [self._viterbi_decode(emission, mask_seq) 
                for emission, mask_seq in zip(emissions, mask)]
    
    def _forward_alg(self, emissions: torch.Tensor, 
                     mask: torch.Tensor) -> torch.Tensor:
        """
        Forward algorithm for CRF.
        """
        batch_size, seq_len, num_tags = emissions.size()
        
        # Initialize forward variables
        forward_vars = emissions[:, 0] + self.start_transitions.unsqueeze(0)
        
        for t in range(1, seq_len):
            # Broadcast forward variables
            forward_vars = forward_vars.unsqueeze(2)
            
            # Add transition scores
            transition_scores = self.transitions.unsqueeze(0)
            forward_vars = forward_vars + transition_scores
            
            # Add emission scores
            emission_scores = emissions[:, t].unsqueeze(1)
            forward_vars = forward_vars + emission_scores
            
            # Log-sum-exp
            forward_vars = torch.logsumexp(forward_vars, dim=1)
            
            # Apply mask
            mask_t = mask[:, t].unsqueeze(1)
            forward_vars = forward_vars * mask_t + forward_vars * (1 - mask_t)
        
        # Add end transitions
        forward_vars = forward_vars + self.end_transitions.unsqueeze(0)
        
        return torch.logsumexp(forward_vars, dim=1).sum()
    
    def _score_sentence(self, emissions: torch.Tensor, 
                       tags: torch.Tensor, 
                       mask: torch.Tensor) -> torch.Tensor:
        """
        Score a sentence with given tags.
        """
        batch_size, seq_len, num_tags = emissions.size()
        
        # Start transitions
        score = self.start_transitions[tags[:, 0]]
        
        # Emission and transition scores
        for t in range(seq_len - 1):
            score += emissions[:, t, tags[:, t]]
            score += self.transitions[tags[:, t], tags[:, t + 1]]
        
        # Last emission
        score += emissions[:, seq_len - 1, tags[:, seq_len - 1]]
        
        # End transitions
        score += self.end_transitions[tags[:, seq_len - 1]]
        
        return score.sum()
    
    def _viterbi_decode(self, emissions: torch.Tensor, 
                       mask: torch.Tensor) -> List[int]:
        """
        Viterbi decoding for a single sequence.
        """
        seq_len, num_tags = emissions.size()
        
        # Initialize viterbi variables
        viterbi = emissions[0] + self.start_transitions
        
        # Forward pass
        for t in range(1, seq_len):
            viterbi = viterbi.unsqueeze(1) + self.transitions
            viterbi = emissions[t] + viterbi.max(dim=0)[0]
        
        # Add end transitions
        viterbi = viterbi + self.end_transitions
        
        # Backward pass
        best_path = []
        best_tag = viterbi.argmax().item()
        best_path.append(best_tag)
        
        for t in range(seq_len - 1, 0, -1):
            best_tag = (viterbi[t] - emissions[t, best_tag] - 
                       self.end_transitions[best_tag]).argmax().item()
            best_path.append(best_tag)
        
        return best_path[::-1]

# Example usage
vocab_size = 10000
num_tags = 9  # B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC, O

model = LSTMCRFNER(
    vocab_size=vocab_size,
    embedding_dim=128,
    hidden_dim=256,
    num_tags=num_tags
)

print(f"LSTM-CRF model created with {sum(p.numel() for p in model.parameters())} parameters")
```

## Implementation Examples

### Using spaCy for NER

```python
import spacy

def extract_entities_spacy(text: str) -> List[Dict[str, any]]:
    """
    Extract entities using spaCy.
    """
    # Load English model
    nlp = spacy.load("en_core_web_sm")
    
    # Process text
    doc = nlp(text)
    
    # Extract entities
    entities = []
    for ent in doc.ents:
        entity = {
            'text': ent.text,
            'type': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char,
            'confidence': 0.8  # spaCy doesn't provide confidence scores
        }
        entities.append(entity)
    
    return entities

# Example usage
text = "Apple Inc. CEO Tim Cook announced new products in Cupertino, California on January 15, 2023."
entities = extract_entities_spacy(text)

print("spaCy NER Results:")
for entity in entities:
    print(f"  {entity['text']} ({entity['type']}) at position {entity['start']}-{entity['end']}")
```

### Using Transformers for NER

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

class TransformerNER:
    """
    Transformer-based Named Entity Recognition.
    """
    
    def __init__(self, model_name: str = "dbmdz/bert-large-cased-finetuned-conll03-english"):
        """
        Initialize transformer NER model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.model.eval()
        
        # Entity label mapping
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
    
    def extract_entities(self, text: str) -> List[Dict[str, any]]:
        """
        Extract entities from text.
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True
        )
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        # Extract entities
        entities = []
        current_entity = None
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        offset_mapping = inputs['offset_mapping'][0]
        
        for i, (token, pred_id, offset) in enumerate(zip(tokens, predictions[0], offset_mapping)):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            label = self.id2label[pred_id.item()]
            
            if label.startswith('B-'):  # Beginning of entity
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'text': token,
                    'type': label[2:],
                    'start': offset[0].item(),
                    'end': offset[1].item()
                }
            elif label.startswith('I-') and current_entity and label[2:] == current_entity['type']:
                # Inside of entity
                current_entity['text'] += token.replace('##', '')
                current_entity['end'] = offset[1].item()
            elif label == 'O':  # Outside of entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities

# Example usage
transformer_ner = TransformerNER()
text = "Apple Inc. CEO Tim Cook announced new products in Cupertino, California on January 15, 2023."
entities = transformer_ner.extract_entities(text)

print("Transformer NER Results:")
for entity in entities:
    print(f"  {entity['text']} ({entity['type']}) at position {entity['start']}-{entity['end']}")
```

## Evaluation Metrics

### NER Evaluation

```python
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

def evaluate_ner(predictions: List[Dict], gold_standard: List[Dict]) -> Dict[str, float]:
    """
    Evaluate NER performance.
    """
    # Convert to sets for comparison
    pred_entities = set()
    gold_entities = set()
    
    for entity in predictions:
        pred_entities.add((entity['text'], entity['type'], entity['start'], entity['end']))
    
    for entity in gold_standard:
        gold_entities.add((entity['text'], entity['type'], entity['start'], entity['end']))
    
    # Calculate metrics
    true_positives = len(pred_entities.intersection(gold_entities))
    false_positives = len(pred_entities - gold_entities)
    false_negatives = len(gold_entities - pred_entities)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

# Example usage
predictions = [
    {'text': 'John Smith', 'type': 'PERSON', 'start': 0, 'end': 10},
    {'text': 'Google', 'type': 'ORGANIZATION', 'start': 20, 'end': 26}
]

gold_standard = [
    {'text': 'John Smith', 'type': 'PERSON', 'start': 0, 'end': 10},
    {'text': 'Microsoft', 'type': 'ORGANIZATION', 'start': 20, 'end': 29}
]

metrics = evaluate_ner(predictions, gold_standard)
print("NER Evaluation Metrics:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.3f}")
```

## Applications

### Information Extraction Pipeline

```python
class InformationExtractor:
    """
    Complete information extraction pipeline using NER.
    """
    
    def __init__(self):
        """Initialize information extractor."""
        self.ner_model = TransformerNER()
    
    def extract_information(self, text: str) -> Dict[str, List[str]]:
        """
        Extract structured information from text.
        """
        entities = self.ner_model.extract_entities(text)
        
        # Group entities by type
        information = {
            'persons': [],
            'organizations': [],
            'locations': [],
            'dates': [],
            'money': [],
            'other': []
        }
        
        for entity in entities:
            entity_type = entity['type'].lower()
            if entity_type in information:
                information[entity_type + 's'].append(entity['text'])
            else:
                information['other'].append(entity['text'])
        
        return information
    
    def create_knowledge_graph(self, texts: List[str]) -> Dict[str, set]:
        """
        Create a simple knowledge graph from entities.
        """
        knowledge_graph = {
            'persons': set(),
            'organizations': set(),
            'locations': set(),
            'relationships': set()
        }
        
        for text in texts:
            info = self.extract_information(text)
            
            # Add entities
            knowledge_graph['persons'].update(info['persons'])
            knowledge_graph['organizations'].update(info['organizations'])
            knowledge_graph['locations'].update(info['locations'])
            
            # Add relationships (simplified)
            for person in info['persons']:
                for org in info['organizations']:
                    knowledge_graph['relationships'].add(f"{person} works at {org}")
        
        return knowledge_graph

# Example usage
extractor = InformationExtractor()
text = "Tim Cook is the CEO of Apple Inc. in Cupertino, California. He earns $15 million annually."
information = extractor.extract_information(text)

print("Extracted Information:")
for category, entities in information.items():
    if entities:
        print(f"  {category}: {', '.join(entities)}")
```

## Conclusion

Named Entity Recognition is a fundamental NLP task that enables the extraction of structured information from unstructured text.

### Key Takeaways

- **Multiple Approaches**: Rule-based, ML, and deep learning methods
- **Entity Types**: Standard and domain-specific entity categories
- **Evaluation**: Precision, recall, and F1-score metrics
- **Applications**: Information extraction, knowledge graphs, search

### Best Practices

1. **Choose Appropriate Method**: Select based on data availability and requirements
2. **Domain Adaptation**: Adapt models to specific domains
3. **Evaluation**: Use proper evaluation metrics and test sets
4. **Post-processing**: Clean and validate extracted entities
5. **Integration**: Combine with other NLP components

### Next Steps

1. **Explore Advanced Models**: Try BERT, RoBERTa, and other transformers
2. **Domain-Specific NER**: Adapt to medical, legal, or financial domains
3. **Multilingual NER**: Extend to multiple languages
4. **Entity Linking**: Connect entities to knowledge bases
5. **Real-time Processing**: Optimize for production systems

---

**Happy Entity Recognition!** 