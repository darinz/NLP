# Neural Networks for NLP

A comprehensive guide to neural network architectures commonly used in Natural Language Processing (NLP) tasks.

## Table of Contents
1. [Introduction](#introduction)
2. [Feedforward Neural Networks](#feedforward-neural-networks)
3. [Recurrent Neural Networks (RNNs)](#recurrent-neural-networks-rnns)
4. [Long Short-Term Memory (LSTM)](#long-short-term-memory-lstm)
5. [Gated Recurrent Units (GRU)](#gated-recurrent-units-gru)
6. [Attention Mechanisms](#attention-mechanisms)
7. [Transformer Architecture](#transformer-architecture)
8. [Convolutional Neural Networks (CNNs)](#convolutional-neural-networks-cnns)
9. [Implementation Examples](#implementation-examples)
10. [Best Practices](#best-practices)

## Introduction

Neural networks have revolutionized NLP by enabling models to learn complex patterns in text data. This guide covers the most important architectures and their implementations.

### Key Concepts
- **Sequential Data**: Text is inherently sequential
- **Variable Length**: Sentences have different lengths
- **Context Dependencies**: Words depend on previous words
- **Semantic Understanding**: Capturing meaning and relationships

## Feedforward Neural Networks

### Basic Structure

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedforwardNN(nn.Module):
    """
    Basic feedforward neural network for NLP tasks.
    """
    
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super(FeedforwardNN, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Example usage for text classification
class TextClassifier(nn.Module):
    """
    Text classifier using feedforward network.
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, dropout=0.1):
        super(TextClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.classifier = FeedforwardNN(embedding_dim, hidden_size, num_classes, dropout)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Average pooling over sequence length
        pooled = torch.mean(embedded, dim=1)  # (batch_size, embedding_dim)
        
        output = self.classifier(pooled)
        return output
```

## Recurrent Neural Networks (RNNs)

### Basic RNN Implementation

```python
class SimpleRNN(nn.Module):
    """
    Simple RNN implementation.
    """
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward pass through RNN
        out, hidden = self.rnn(x, h0)
        
        # Use the last output for classification
        out = self.fc(out[:, -1, :])
        return out

# Character-level RNN for name classification
class CharRNN(nn.Module):
    """
    Character-level RNN for name classification.
    """
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(CharRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, hidden_size)
        
        batch_size = embedded.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, hidden = self.rnn(embedded, h0)
        out = self.fc(out[:, -1, :])
        return out
```

## Long Short-Term Memory (LSTM)

### LSTM Implementation

```python
class LSTMClassifier(nn.Module):
    """
    LSTM-based text classifier.
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, num_layers=1, dropout=0.1):
        super(LSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        batch_size = embedded.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward pass through LSTM
        out, (hidden, cell) = self.lstm(embedded, (h0, c0))
        
        # Use the last hidden state
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# Bidirectional LSTM
class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM classifier.
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, num_layers=1, dropout=0.1):
        super(BiLSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, 
                           batch_first=True, bidirectional=True, 
                           dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        embedded = self.embedding(x)
        
        batch_size = embedded.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        
        out, (hidden, cell) = self.lstm(embedded, (h0, c0))
        
        # Concatenate forward and backward hidden states
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out
```

## Gated Recurrent Units (GRU)

### GRU Implementation

```python
class GRUClassifier(nn.Module):
    """
    GRU-based text classifier.
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, num_layers=1, dropout=0.1):
        super(GRUClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        embedded = self.embedding(x)
        
        batch_size = embedded.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, hidden = self.gru(embedded, h0)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out
```

## Attention Mechanisms

### Basic Attention

```python
class Attention(nn.Module):
    """
    Basic attention mechanism.
    """
    
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, hidden_states):
        # hidden_states shape: (batch_size, seq_len, hidden_size)
        
        # Calculate attention weights
        energy = torch.tanh(self.attention(hidden_states))
        attention_weights = torch.softmax(self.v(energy), dim=1)
        
        # Apply attention weights
        context = torch.sum(attention_weights * hidden_states, dim=1)
        
        return context, attention_weights

# LSTM with Attention
class LSTMWithAttention(nn.Module):
    """
    LSTM classifier with attention mechanism.
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, num_layers=1, dropout=0.1):
        super(LSTMWithAttention, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        embedded = self.embedding(x)
        
        batch_size = embedded.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(embedded, (h0, c0))
        
        # Apply attention
        context, attention_weights = self.attention(lstm_out)
        
        # Classification
        out = self.dropout(context)
        out = self.fc(out)
        
        return out, attention_weights
```

## Transformer Architecture

### Multi-Head Attention

```python
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Scaled dot-product attention.
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        # Final linear transformation
        output = self.w_o(attention_output)
        
        return output, attention_weights

# Positional Encoding
class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.
    """
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer.
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

# Complete Transformer
class TransformerClassifier(nn.Module):
    """
    Transformer-based text classifier.
    """
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, num_classes, 
                 max_len=512, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len)
        seq_len = x.size(1)
        
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        # Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        
        # Global average pooling
        if mask is not None:
            x = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            x = x.mean(dim=1)
        
        # Classification
        output = self.fc(x)
        return output
```

## Convolutional Neural Networks (CNNs)

### CNN for Text Classification

```python
class TextCNN(nn.Module):
    """
    CNN for text classification.
    """
    
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_classes, dropout=0.1):
        super(TextCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim))
            for k in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = embedded.unsqueeze(1)  # (batch_size, 1, seq_len, embedding_dim)
        
        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # (batch_size, num_filters, seq_len-k+1, 1)
            conv_out = conv_out.squeeze(3)  # (batch_size, num_filters, seq_len-k+1)
            conv_out = F.max_pool1d(conv_out, conv_out.size(2))  # (batch_size, num_filters, 1)
            conv_outputs.append(conv_out.squeeze(2))  # (batch_size, num_filters)
        
        # Concatenate and classify
        concatenated = torch.cat(conv_outputs, dim=1)  # (batch_size, len(filter_sizes) * num_filters)
        concatenated = self.dropout(concatenated)
        output = self.fc(concatenated)
        
        return output
```

## Implementation Examples

### Training Loop

```python
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    """
    Training loop for NLP models.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        accuracy = 100 * correct / total
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Validation Accuracy: {accuracy:.2f}%')
        print('-' * 50)
    
    return train_losses, val_losses

# Example usage
def example_training():
    """
    Example of training an LSTM classifier.
    """
    # Model parameters
    vocab_size = 10000
    embedding_dim = 128
    hidden_size = 256
    num_classes = 5
    num_layers = 2
    
    # Create model
    model = LSTMClassifier(vocab_size, embedding_dim, hidden_size, num_classes, num_layers)
    
    # Create dummy data (replace with real data)
    batch_size = 32
    seq_len = 50
    
    # Dummy tensors
    train_data = torch.randint(0, vocab_size, (1000, seq_len))
    train_labels = torch.randint(0, num_classes, (1000,))
    val_data = torch.randint(0, vocab_size, (200, seq_len))
    val_labels = torch.randint(0, num_classes, (200,))
    
    # Create data loaders
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader)
    
    return model, train_losses, val_losses
```

## Best Practices

### 1. **Model Selection**
- **RNN**: Good for short sequences, simple tasks
- **LSTM**: Better for longer sequences, vanishing gradient problem
- **GRU**: Faster training, similar performance to LSTM
- **Transformer**: Best for most modern NLP tasks
- **CNN**: Good for text classification, parallel processing

### 2. **Hyperparameter Tuning**
- **Hidden Size**: Start with 128-512, increase for complex tasks
- **Number of Layers**: 1-4 layers usually sufficient
- **Dropout**: 0.1-0.5 to prevent overfitting
- **Learning Rate**: 0.001-0.0001 for Adam optimizer

### 3. **Data Preprocessing**
- Use appropriate tokenization (word, subword, character)
- Handle variable sequence lengths with padding
- Apply proper text cleaning and normalization

### 4. **Training Strategies**
- Use early stopping to prevent overfitting
- Implement learning rate scheduling
- Use gradient clipping for RNNs
- Monitor training and validation metrics

### 5. **Performance Optimization**
- Use batch processing for efficiency
- Implement proper data loading with DataLoader
- Use GPU acceleration when available
- Profile memory usage for large models

## Conclusion

Neural networks have transformed NLP by enabling models to learn complex patterns in text data. The choice of architecture depends on your specific task, data characteristics, and computational resources. Start with simpler models and gradually move to more complex architectures as needed.

Remember: **The best model is the one that solves your problem effectively!** 