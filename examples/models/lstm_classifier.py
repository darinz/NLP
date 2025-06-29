"""
LSTM Classifier for NLP Tasks

This module provides a complete LSTM-based classifier implementation
for various NLP tasks like sentiment analysis, text classification,
and sequence labeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import Counter


class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for NLP tasks.
    
    This model uses an LSTM to process text sequences and a
    classification head to predict labels.
    """
    
    def __init__(self, 
                 vocab_size: int,
                 embedding_dim: int = 128,
                 hidden_size: int = 256,
                 num_layers: int = 2,
                 num_classes: int = 2,
                 dropout: float = 0.1,
                 bidirectional: bool = True,
                 pretrained_embeddings: Optional[np.ndarray] = None):
        """
        Initialize LSTM classifier.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_size: Size of LSTM hidden layers
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            pretrained_embeddings: Pre-trained embedding weights
        """
        super(LSTMClassifier, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        if bidirectional:
            attention_input_size = hidden_size * 2
        else:
            attention_input_size = hidden_size
        
        self.attention = nn.Linear(attention_input_size, 1)
        
        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(attention_input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token indices (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            lengths: Sequence lengths (batch_size,)
            
        Returns:
            Logits for classification (batch_size, num_classes)
        """
        batch_size, seq_len = input_ids.size()
        
        # Embedding
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        
        # Pack sequence if lengths are provided
        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Unpack sequence if it was packed
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=seq_len
            )
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
            lstm_out = lstm_out * attention_mask
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        # attention_weights: (batch_size, seq_len, 1)
        
        # Apply attention
        context = torch.sum(attention_weights * lstm_out, dim=1)
        # context: (batch_size, hidden_size * 2) if bidirectional
        
        # Classification
        out = F.relu(self.fc1(context))
        out = self.dropout(out)
        logits = self.fc2(out)
        
        return logits
    
    def get_attention_weights(self, input_ids: torch.Tensor,
                            attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get attention weights for interpretability.
        
        Args:
            input_ids: Input token indices
            attention_mask: Attention mask
            
        Returns:
            Attention weights (batch_size, seq_len)
        """
        batch_size, seq_len = input_ids.size()
        
        # Embedding
        embedded = self.embedding(input_ids)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(embedded)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1)
            lstm_out = lstm_out * attention_mask
        
        # Get attention weights
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        
        return attention_weights.squeeze(-1)  # (batch_size, seq_len)


class TextDataset(Dataset):
    """
    Dataset for text classification tasks.
    """
    
    def __init__(self, texts: List[str], labels: List[int], 
                 tokenizer, max_length: int = 128):
        """
        Initialize dataset.
        
        Args:
            texts: List of text strings
            labels: List of label integers
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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


class LSTMClassifierTrainer:
    """
    Trainer class for LSTM classifier.
    """
    
    def __init__(self, model: LSTMClassifier, device: str = 'cpu'):
        """
        Initialize trainer.
        
        Args:
            model: LSTM classifier model
            device: Device to train on
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, train_loader: DataLoader, 
                   optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100 * correct / total
        }
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': 100 * correct / total,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def predict(self, text: str, tokenizer) -> Dict[str, any]:
        """
        Make prediction on single text.
        
        Args:
            text: Input text
            tokenizer: Tokenizer for text processing
            
        Returns:
            Dictionary with prediction results
        """
        self.model.eval()
        
        # Tokenize
        encoding = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy()
        }


def create_lstm_classifier(vocab_size: int, 
                          embedding_dim: int = 128,
                          hidden_size: int = 256,
                          num_layers: int = 2,
                          num_classes: int = 2,
                          dropout: float = 0.1,
                          bidirectional: bool = True) -> LSTMClassifier:
    """
    Create LSTM classifier with specified parameters.
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_size: Size of LSTM hidden layers
        num_layers: Number of LSTM layers
        num_classes: Number of output classes
        dropout: Dropout rate
        bidirectional: Whether to use bidirectional LSTM
        
    Returns:
        LSTM classifier model
    """
    model = LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        bidirectional=bidirectional
    )
    
    return model


# Example usage
if __name__ == "__main__":
    # Example parameters
    vocab_size = 10000
    embedding_dim = 128
    hidden_size = 256
    num_layers = 2
    num_classes = 3  # Positive, Negative, Neutral
    
    # Create model
    model = create_lstm_classifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Example forward pass
    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    outputs = model(input_ids, attention_mask)
    print(f"Output shape: {outputs.shape}")
    
    # Get attention weights
    attention_weights = model.get_attention_weights(input_ids, attention_mask)
    print(f"Attention weights shape: {attention_weights.shape}") 