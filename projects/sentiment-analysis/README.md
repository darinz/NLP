# Sentiment Analysis Project

A complete sentiment analysis system using LSTM and BERT models for classifying text sentiment.

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Data Preparation](#data-preparation)
4. [Model Implementation](#model-implementation)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Deployment](#deployment)
8. [Usage Examples](#usage-examples)

## Overview

This project implements sentiment analysis using both traditional LSTM models and modern BERT-based approaches. It provides a complete pipeline from data preprocessing to model deployment.

### Features
- **LSTM-based sentiment classifier**
- **BERT fine-tuning for sentiment analysis**
- **Data preprocessing and augmentation**
- **Model evaluation and comparison**
- **API deployment with FastAPI**
- **Interactive web interface**

## Project Structure

```
sentiment-analysis/
├── data/
│   ├── raw/                    # Raw datasets
│   ├── processed/              # Processed datasets
│   └── external/               # External datasets
├── models/
│   ├── lstm_sentiment.py       # LSTM model implementation
│   ├── bert_sentiment.py       # BERT model implementation
│   └── saved_models/           # Trained model files
├── src/
│   ├── data_preprocessing.py   # Data preprocessing utilities
│   ├── training.py             # Training scripts
│   ├── evaluation.py           # Evaluation utilities
│   └── utils.py                # Utility functions
├── api/
│   ├── main.py                 # FastAPI application
│   ├── models.py               # API models
│   └── requirements.txt        # API dependencies
├── web/
│   ├── app.py                  # Streamlit web app
│   └── requirements.txt        # Web app dependencies
├── notebooks/
│   ├── data_exploration.ipynb  # Data exploration
│   ├── model_comparison.ipynb  # Model comparison
│   └── results_analysis.ipynb  # Results analysis
├── tests/
│   ├── test_models.py          # Model tests
│   └── test_preprocessing.py   # Preprocessing tests
├── config/
│   └── config.yaml             # Configuration file
├── requirements.txt            # Main dependencies
└── README.md                   # This file
```

## Data Preparation

### Dataset Loading

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_sentiment_data(file_path):
    """
    Load sentiment analysis dataset.
    """
    df = pd.read_csv(file_path)
    
    # Clean data
    df = df.dropna()
    df = df[df['text'].str.len() > 10]  # Remove very short texts
    
    return df

def prepare_data(df, test_size=0.2, val_size=0.1):
    """
    Prepare data for training.
    """
    # Encode labels
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['sentiment'])
    
    # Split data
    train_df, temp_df = train_test_split(df, test_size=test_size + val_size, 
                                        random_state=42, stratify=df['label_encoded'])
    
    val_size_adjusted = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(temp_df, test_size=val_size_adjusted, 
                                      random_state=42, stratify=temp_df['label_encoded'])
    
    return train_df, val_df, test_df, label_encoder

# Example usage
df = load_sentiment_data('data/raw/sentiment_dataset.csv')
train_df, val_df, test_df, label_encoder = prepare_data(df)
```

### Text Preprocessing

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor:
    """
    Text preprocessing for sentiment analysis.
    """
    
    def __init__(self, remove_stopwords=True, lemmatize=True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """
        Clean and preprocess text.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_clean(self, text):
        """
        Tokenize and clean text.
        """
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatize
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def preprocess_batch(self, texts):
        """
        Preprocess a batch of texts.
        """
        processed_texts = []
        for text in texts:
            tokens = self.tokenize_and_clean(text)
            processed_texts.append(' '.join(tokens))
        
        return processed_texts

# Example usage
preprocessor = TextPreprocessor()
processed_texts = preprocessor.preprocess_batch(train_df['text'].tolist())
```

## Model Implementation

### LSTM Sentiment Classifier

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class SentimentDataset(Dataset):
    """
    Dataset for sentiment analysis.
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class LSTMSentimentClassifier(nn.Module):
    """
    LSTM-based sentiment classifier.
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, 
                 num_layers=2, dropout=0.1, pretrained_embeddings=None):
        super(LSTMSentimentClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention layer
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.output = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask=None):
        # Embedding
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out: (batch_size, seq_len, hidden_size * 2)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        # attention_weights: (batch_size, seq_len, 1)
        
        # Apply attention
        context = torch.sum(attention_weights * lstm_out, dim=1)
        # context: (batch_size, hidden_size * 2)
        
        # Classification
        out = F.relu(self.fc(context))
        out = self.dropout(out)
        out = self.output(out)
        
        return out

def create_lstm_model(vocab_size, embedding_dim=128, hidden_size=256, 
                     num_classes=3, num_layers=2, dropout=0.1):
    """
    Create LSTM sentiment classifier.
    """
    model = LSTMSentimentClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout
    )
    
    return model
```

### BERT Sentiment Classifier

```python
from transformers import AutoModel, AutoTokenizer

class BERTSentimentClassifier(nn.Module):
    """
    BERT-based sentiment classifier.
    """
    
    def __init__(self, model_name, num_classes, dropout=0.1):
        super(BERTSentimentClassifier, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask=None):
        # BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

def create_bert_model(model_name='bert-base-uncased', num_classes=3, dropout=0.1):
    """
    Create BERT sentiment classifier.
    """
    model = BERTSentimentClassifier(model_name, num_classes, dropout)
    return model
```

## Training

### Training Script

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
import wandb
import yaml

def train_model(model, train_loader, val_loader, config):
    """
    Train sentiment analysis model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler_step'], gamma=0.1)
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_true_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_predictions.extend(predicted.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss / len(train_loader),
            'val_loss': val_loss / len(val_loader),
            'train_acc': train_acc,
            'val_acc': val_acc,
            'learning_rate': scheduler.get_last_lr()[0]
        })
        
        print(f'Epoch [{epoch+1}/{config["epochs"]}]')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'models/saved_models/best_model.pth')
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
        
        scheduler.step()
    
    return model

def main():
    """
    Main training function.
    """
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize wandb
    wandb.init(project="sentiment-analysis", config=config)
    
    # Load data
    train_df, val_df, test_df, label_encoder = prepare_data(load_sentiment_data(config['data_path']))
    
    # Preprocess data
    preprocessor = TextPreprocessor()
    train_texts = preprocessor.preprocess_batch(train_df['text'].tolist())
    val_texts = preprocessor.preprocess_batch(val_df['text'].tolist())
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_df['label_encoded'].tolist(), tokenizer)
    val_dataset = SentimentDataset(val_texts, val_df['label_encoded'].tolist(), tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    # Create model
    if config['model_type'] == 'bert':
        model = create_bert_model(config['model_name'], len(label_encoder.classes_))
    else:
        model = create_lstm_model(len(tokenizer), config['embedding_dim'], 
                                config['hidden_size'], len(label_encoder.classes_))
    
    # Train model
    trained_model = train_model(model, train_loader, val_loader, config)
    
    wandb.finish()

if __name__ == "__main__":
    main()
```

### Configuration

```yaml
# config/config.yaml
model_type: "bert"  # "lstm" or "bert"
model_name: "bert-base-uncased"
data_path: "data/raw/sentiment_dataset.csv"

# Training parameters
epochs: 10
batch_size: 16
learning_rate: 2e-5
scheduler_step: 3

# Model parameters
embedding_dim: 128
hidden_size: 256
num_layers: 2
dropout: 0.1

# Data parameters
max_length: 128
test_size: 0.2
val_size: 0.1
```

## Evaluation

### Evaluation Script

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, test_loader, label_encoder, device):
    """
    Evaluate model performance.
    """
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs.data, 1)
            
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, 
                                 target_names=label_encoder.classes_)
    conf_matrix = confusion_matrix(true_labels, predictions)
    
    return accuracy, report, conf_matrix, predictions, true_labels

def plot_confusion_matrix(conf_matrix, label_encoder, save_path=None):
    """
    Plot confusion matrix.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_training_history(history):
    """
    Plot training history.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
```

## Deployment

### FastAPI Application

```python
# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
import numpy as np

app = FastAPI(title="Sentiment Analysis API", version="1.0.0")

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: dict

# Load model and tokenizer
model = None
tokenizer = None
label_encoder = None

@app.on_event("startup")
async def load_model():
    global model, tokenizer, label_encoder
    
    # Load your trained model here
    model = create_bert_model('bert-base-uncased', 3)
    model.load_state_dict(torch.load('models/saved_models/best_model.pth'))
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    # Load label encoder

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    try:
        # Preprocess text
        inputs = tokenizer(
            request.text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        # Map prediction to sentiment
        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        sentiment = sentiment_map[prediction]
        
        # Create response
        response = SentimentResponse(
            sentiment=sentiment,
            confidence=confidence,
            probabilities={
                "negative": probabilities[0][0].item(),
                "neutral": probabilities[0][1].item(),
                "positive": probabilities[0][2].item()
            }
        )
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}
```

### Streamlit Web App

```python
# web/app.py
import streamlit as st
import requests
import json

st.title("Sentiment Analysis App")

# API endpoint
API_URL = "http://localhost:8000/predict"

# Text input
text_input = st.text_area("Enter text for sentiment analysis:", height=100)

if st.button("Analyze Sentiment"):
    if text_input.strip():
        # Make API request
        response = requests.post(API_URL, json={"text": text_input})
        
        if response.status_code == 200:
            result = response.json()
            
            # Display results
            st.subheader("Results")
            
            # Sentiment
            sentiment = result["sentiment"]
            confidence = result["confidence"]
            
            st.write(f"**Sentiment:** {sentiment.title()}")
            st.write(f"**Confidence:** {confidence:.2%}")
            
            # Probabilities
            st.subheader("Probabilities")
            probabilities = result["probabilities"]
            
            for sentiment, prob in probabilities.items():
                st.progress(prob)
                st.write(f"{sentiment.title()}: {prob:.2%}")
        else:
            st.error("Error analyzing sentiment. Please try again.")
    else:
        st.warning("Please enter some text to analyze.")

# Example texts
st.sidebar.subheader("Example Texts")
example_texts = [
    "I love this product! It's amazing and works perfectly.",
    "This is terrible. I'm very disappointed with the quality.",
    "It's okay, nothing special but gets the job done."
]

for i, text in enumerate(example_texts):
    if st.sidebar.button(f"Example {i+1}"):
        st.text_area("Selected text:", text, key=f"example_{i}")
```

## Usage Examples

### Basic Usage

```python
# Load trained model
model = create_bert_model('bert-base-uncased', 3)
model.load_state_dict(torch.load('models/saved_models/best_model.pth'))
model.eval()

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, truncation=True, padding=True, 
                      max_length=128, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
    return sentiment_map[prediction], confidence

# Example predictions
texts = [
    "I absolutely love this movie! It's fantastic!",
    "This product is terrible. I want my money back.",
    "The food was okay, nothing special."
]

for text in texts:
    sentiment, confidence = predict_sentiment(text)
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment} (confidence: {confidence:.2%})")
    print("-" * 50)
```

### Batch Processing

```python
def predict_batch_sentiments(texts, batch_size=32):
    """
    Predict sentiments for a batch of texts.
    """
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize batch
        inputs = tokenizer(batch_texts, truncation=True, padding=True, 
                          max_length=128, return_tensors="pt")
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            confidences = torch.max(probabilities, dim=1)[0]
        
        # Process results
        for j, (pred, conf) in enumerate(zip(predictions, confidences)):
            sentiment = sentiment_map[pred.item()]
            results.append({
                'text': batch_texts[j],
                'sentiment': sentiment,
                'confidence': conf.item()
            })
    
    return results

# Example batch processing
texts = [
    "Great product, highly recommended!",
    "Disappointed with the service.",
    "Average quality, expected better.",
    "Excellent customer support!",
    "Poor quality, waste of money."
]

results = predict_batch_sentiments(texts)
for result in results:
    print(f"Text: {result['text']}")
    print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.2%})")
    print("-" * 50)
```

## Conclusion

This sentiment analysis project provides a complete pipeline for building and deploying sentiment analysis models. The implementation includes both traditional LSTM approaches and modern BERT-based methods, with comprehensive evaluation and deployment options.

Key features:
- **Flexible model architecture** supporting both LSTM and BERT
- **Complete training pipeline** with validation and early stopping
- **Comprehensive evaluation** with multiple metrics
- **Production-ready deployment** with FastAPI and Streamlit
- **Easy-to-use interface** for both API and web applications

The project can be extended with additional features such as:
- Multi-language support
- Aspect-based sentiment analysis
- Real-time streaming analysis
- Advanced visualization dashboards 