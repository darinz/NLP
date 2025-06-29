# Model Deployment Guide

A comprehensive guide to deploying NLP models to production environments.

## Table of Contents
1. [Introduction](#introduction)
2. [Model Serialization](#model-serialization)
3. [API Development](#api-development)
4. [Docker Containerization](#docker-containerization)
5. [Cloud Deployment](#cloud-deployment)
6. [Performance Optimization](#performance-optimization)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Best Practices](#best-practices)

## Introduction

Deploying NLP models to production requires careful consideration of scalability, performance, and maintainability. This guide covers the essential aspects of model deployment.

### Deployment Challenges
- **Model Size**: Large models require efficient serving
- **Latency**: Real-time applications need fast inference
- **Scalability**: Handle varying load demands
- **Monitoring**: Track model performance and health
- **Versioning**: Manage model updates and rollbacks

## Model Serialization

### PyTorch Model Serialization

```python
import torch
import torch.nn as nn
import pickle
import json

class SentimentClassifier(nn.Module):
    """
    Example sentiment classifier for deployment.
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super(SentimentClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        pooled = torch.mean(lstm_out, dim=1)
        pooled = self.dropout(pooled)
        output = self.fc(pooled)
        return output

def save_model(model, tokenizer, config, model_path, tokenizer_path, config_path):
    """
    Save model, tokenizer, and configuration.
    """
    # Save PyTorch model
    torch.save(model.state_dict(), model_path)
    
    # Save tokenizer
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Save configuration
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Model saved to {model_path}")
    print(f"Tokenizer saved to {tokenizer_path}")
    print(f"Config saved to {config_path}")

def load_model(model_class, model_path, tokenizer_path, config_path):
    """
    Load model, tokenizer, and configuration.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize model
    model = model_class(**config['model_params'])
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Load tokenizer
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    return model, tokenizer, config

# Example usage
def example_serialization():
    """
    Example of model serialization.
    """
    # Model parameters
    config = {
        'model_params': {
            'vocab_size': 10000,
            'embedding_dim': 128,
            'hidden_size': 256,
            'num_classes': 3
        },
        'training_params': {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 10
        }
    }
    
    # Create model
    model = SentimentClassifier(**config['model_params'])
    
    # Create dummy tokenizer
    class SimpleTokenizer:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size
            self.word_to_idx = {f'word_{i}': i for i in range(vocab_size)}
        
        def encode(self, text):
            words = text.lower().split()
            return [self.word_to_idx.get(word, 0) for word in words]
    
    tokenizer = SimpleTokenizer(config['model_params']['vocab_size'])
    
    # Save model
    save_model(
        model, tokenizer, config,
        'model.pth', 'tokenizer.pkl', 'config.json'
    )
    
    # Load model
    loaded_model, loaded_tokenizer, loaded_config = load_model(
        SentimentClassifier, 'model.pth', 'tokenizer.pkl', 'config.json'
    )
    
    return loaded_model, loaded_tokenizer, loaded_config
```

### ONNX Export

```python
import onnx
import onnxruntime

def export_to_onnx(model, dummy_input, onnx_path):
    """
    Export PyTorch model to ONNX format.
    """
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'sequence_length'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"Model exported to {onnx_path}")

def load_onnx_model(onnx_path):
    """
    Load ONNX model for inference.
    """
    session = onnxruntime.InferenceSession(onnx_path)
    return session

def predict_with_onnx(session, input_data):
    """
    Make predictions with ONNX model.
    """
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    result = session.run([output_name], {input_name: input_data})
    return result[0]

# Example usage
def example_onnx_export():
    """
    Example of ONNX export and inference.
    """
    # Create model
    model = SentimentClassifier(10000, 128, 256, 3)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randint(0, 10000, (1, 50))
    
    # Export to ONNX
    export_to_onnx(model, dummy_input, 'model.onnx')
    
    # Load and use ONNX model
    session = load_onnx_model('model.onnx')
    
    # Make prediction
    input_data = dummy_input.numpy()
    prediction = predict_with_onnx(session, input_data)
    
    print("ONNX prediction shape:", prediction.shape)
    return session
```

## API Development

### FastAPI Implementation

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from typing import List, Dict, Any

app = FastAPI(title="NLP Model API", version="1.0.0")

# Request/Response models
class TextRequest(BaseModel):
    text: str
    max_length: int = 512

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    probabilities: List[float]
    processing_time: float

class BatchRequest(BaseModel):
    texts: List[str]
    max_length: int = 512

class BatchResponse(BaseModel):
    predictions: List[int]
    confidences: List[float]
    processing_time: float

# Global model variables
model = None
tokenizer = None
config = None

@app.on_event("startup")
async def load_model():
    """
    Load model on startup.
    """
    global model, tokenizer, config
    
    try:
        model, tokenizer, config = load_model(
            SentimentClassifier, 'model.pth', 'tokenizer.pkl', 'config.json'
        )
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

def preprocess_text(text: str, max_length: int) -> torch.Tensor:
    """
    Preprocess text for model input.
    """
    # Tokenize
    tokens = tokenizer.encode(text)
    
    # Pad or truncate
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    else:
        tokens = tokens + [0] * (max_length - len(tokens))
    
    return torch.tensor([tokens], dtype=torch.long)

@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(request: TextRequest):
    """
    Predict sentiment for a single text.
    """
    import time
    start_time = time.time()
    
    try:
        # Preprocess
        input_tensor = preprocess_text(request.text, request.max_length)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            probabilities=probabilities[0].tolist(),
            processing_time=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=BatchResponse)
async def predict_sentiment_batch(request: BatchRequest):
    """
    Predict sentiment for multiple texts.
    """
    import time
    start_time = time.time()
    
    try:
        predictions = []
        confidences = []
        
        for text in request.texts:
            # Preprocess
            input_tensor = preprocess_text(text, request.max_length)
            
            # Predict
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                prediction = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][prediction].item()
            
            predictions.append(prediction)
            confidences.append(confidence)
        
        processing_time = time.time() - start_time
        
        return BatchResponse(
            predictions=predictions,
            confidences=confidences,
            processing_time=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/model_info")
async def model_info():
    """
    Get model information.
    """
    if config is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": "SentimentClassifier",
        "vocab_size": config['model_params']['vocab_size'],
        "embedding_dim": config['model_params']['embedding_dim'],
        "hidden_size": config['model_params']['hidden_size'],
        "num_classes": config['model_params']['num_classes']
    }

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
```

### Flask Implementation

```python
from flask import Flask, request, jsonify
import torch
import time
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variables
model = None
tokenizer = None

def load_model_on_startup():
    """
    Load model when Flask starts.
    """
    global model, tokenizer
    try:
        model, tokenizer, _ = load_model(
            SentimentClassifier, 'model.pth', 'tokenizer.pkl', 'config.json'
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e

@app.before_first_request
def before_first_request():
    """
    Load model before first request.
    """
    load_model_on_startup()

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict sentiment endpoint.
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        max_length = data.get('max_length', 512)
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        start_time = time.time()
        
        # Preprocess
        input_tensor = preprocess_text(text, max_length)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        processing_time = time.time() - start_time
        
        response = {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probabilities[0].tolist(),
            'processing_time': processing_time
        }
        
        logger.info(f"Prediction completed in {processing_time:.4f}s")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/model_info', methods=['GET'])
def model_info():
    """
    Model information endpoint.
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_type': 'SentimentClassifier',
        'parameters': sum(p.numel() for p in model.parameters())
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

## Docker Containerization

### Dockerfile

```dockerfile
# Use official Python runtime as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Requirements.txt

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
torch==2.1.0
numpy==1.24.3
pydantic==2.5.0
python-multipart==0.0.6
requests==2.31.0
```

### Docker Compose

```yaml
version: '3.8'

services:
  nlp-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - nlp-api
    restart: unless-stopped
```

### Nginx Configuration

```nginx
events {
    worker_connections 1024;
}

http {
    upstream nlp_api {
        server nlp-api:8000;
    }

    server {
        listen 80;
        server_name localhost;

        location / {
            proxy_pass http://nlp_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

## Cloud Deployment

### AWS Lambda with API Gateway

```python
import json
import torch
import base64
import boto3
from io import BytesIO

# Global variables
model = None
tokenizer = None

def load_model():
    """
    Load model from S3 or local storage.
    """
    global model, tokenizer
    
    if model is None:
        # Load model (implement based on your storage solution)
        model, tokenizer, _ = load_model_from_s3()
    
    return model, tokenizer

def lambda_handler(event, context):
    """
    AWS Lambda handler function.
    """
    try:
        # Load model
        model, tokenizer = load_model()
        
        # Parse request
        body = json.loads(event['body'])
        text = body.get('text', '')
        
        if not text:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Text is required'})
            }
        
        # Preprocess and predict
        input_tensor = preprocess_text(text, 512)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        response = {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probabilities[0].tolist()
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(response)
        }
    
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

### Google Cloud Run

```python
# main.py for Google Cloud Run
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your existing FastAPI endpoints here...

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

```dockerfile
# Dockerfile for Google Cloud Run
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "main.py"]
```

## Performance Optimization

### Model Optimization

```python
import torch
import torch.nn as nn
from torch.jit import script

def optimize_model(model, dummy_input):
    """
    Optimize model for production.
    """
    # Set to evaluation mode
    model.eval()
    
    # TorchScript optimization
    scripted_model = script(model)
    
    # Quantization (for CPU inference)
    quantized_model = torch.quantization.quantize_dynamic(
        scripted_model, {nn.Linear}, dtype=torch.qint8
    )
    
    return quantized_model

def benchmark_model(model, input_data, num_runs=100):
    """
    Benchmark model performance.
    """
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_data)
    
    # Benchmark
    import time
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_data)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    
    return avg_time

# Example usage
def optimize_for_production():
    """
    Optimize model for production deployment.
    """
    # Load model
    model, tokenizer, config = load_model(
        SentimentClassifier, 'model.pth', 'tokenizer.pkl', 'config.json'
    )
    
    # Create dummy input
    dummy_input = torch.randint(0, 10000, (1, 512))
    
    # Optimize model
    optimized_model = optimize_model(model, dummy_input)
    
    # Benchmark
    original_time = benchmark_model(model, dummy_input)
    optimized_time = benchmark_model(optimized_model, dummy_input)
    
    print(f"Original model: {original_time:.4f}s per inference")
    print(f"Optimized model: {optimized_time:.4f}s per inference")
    print(f"Speedup: {original_time/optimized_time:.2f}x")
    
    # Save optimized model
    torch.save(optimized_model.state_dict(), 'optimized_model.pth')
    
    return optimized_model
```

### Batch Processing

```python
from concurrent.futures import ThreadPoolExecutor
import asyncio

class BatchProcessor:
    """
    Batch processor for efficient inference.
    """
    
    def __init__(self, model, tokenizer, batch_size=32, max_workers=4):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def process_batch(self, texts):
        """
        Process a batch of texts.
        """
        # Preprocess batch
        batch_tensors = []
        for text in texts:
            tensor = preprocess_text(text, 512)
            batch_tensors.append(tensor)
        
        # Stack tensors
        batch_input = torch.cat(batch_tensors, dim=0)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(batch_input)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            confidences = torch.max(probabilities, dim=1)[0]
        
        return {
            'predictions': predictions.tolist(),
            'confidences': confidences.tolist(),
            'probabilities': probabilities.tolist()
        }
    
    async def process_async(self, texts):
        """
        Process texts asynchronously.
        """
        loop = asyncio.get_event_loop()
        
        # Split into batches
        batches = [texts[i:i+self.batch_size] 
                  for i in range(0, len(texts), self.batch_size)]
        
        # Process batches in parallel
        tasks = []
        for batch in batches:
            task = loop.run_in_executor(self.executor, self.process_batch, batch)
            tasks.append(task)
        
        # Wait for all batches
        results = await asyncio.gather(*tasks)
        
        # Combine results
        all_predictions = []
        all_confidences = []
        all_probabilities = []
        
        for result in results:
            all_predictions.extend(result['predictions'])
            all_confidences.extend(result['confidences'])
            all_probabilities.extend(result['probabilities'])
        
        return {
            'predictions': all_predictions,
            'confidences': all_confidences,
            'probabilities': all_probabilities
        }

# Example usage
async def example_batch_processing():
    """
    Example of batch processing.
    """
    # Load model
    model, tokenizer, config = load_model(
        SentimentClassifier, 'model.pth', 'tokenizer.pkl', 'config.json'
    )
    
    # Create batch processor
    processor = BatchProcessor(model, tokenizer, batch_size=16)
    
    # Sample texts
    texts = [
        "I love this product!",
        "This is terrible.",
        "It's okay, nothing special.",
        "Amazing experience!",
        "Disappointed with the quality."
    ] * 10  # 50 texts total
    
    # Process asynchronously
    results = await processor.process_async(texts)
    
    print(f"Processed {len(texts)} texts")
    print(f"Predictions: {results['predictions'][:5]}")
    print(f"Confidences: {results['confidences'][:5]}")
    
    return results
```

## Monitoring and Logging

### Structured Logging

```python
import logging
import json
import time
from datetime import datetime
from typing import Dict, Any

class StructuredLogger:
    """
    Structured logger for NLP model monitoring.
    """
    
    def __init__(self, log_file='nlp_model.log'):
        self.logger = logging.getLogger('nlp_model')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_prediction(self, text: str, prediction: int, confidence: float, 
                      processing_time: float, user_id: str = None):
        """
        Log prediction details.
        """
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'prediction',
            'text_length': len(text),
            'prediction': prediction,
            'confidence': confidence,
            'processing_time': processing_time,
            'user_id': user_id
        }
        
        self.logger.info(json.dumps(log_entry))
    
    def log_error(self, error: str, context: Dict[str, Any] = None):
        """
        Log error details.
        """
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'error',
            'error': error,
            'context': context or {}
        }
        
        self.logger.error(json.dumps(log_entry))
    
    def log_model_metrics(self, metrics: Dict[str, float]):
        """
        Log model performance metrics.
        """
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'metrics',
            'metrics': metrics
        }
        
        self.logger.info(json.dumps(log_entry))

# Example usage
def example_monitoring():
    """
    Example of model monitoring.
    """
    # Initialize logger
    logger = StructuredLogger()
    
    # Load model
    model, tokenizer, config = load_model(
        SentimentClassifier, 'model.pth', 'tokenizer.pkl', 'config.json'
    )
    
    # Sample predictions
    texts = [
        "I love this product!",
        "This is terrible.",
        "It's okay, nothing special."
    ]
    
    for i, text in enumerate(texts):
        start_time = time.time()
        
        try:
            # Predict
            input_tensor = preprocess_text(text, 512)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                prediction = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][prediction].item()
            
            processing_time = time.time() - start_time
            
            # Log prediction
            logger.log_prediction(
                text=text,
                prediction=prediction,
                confidence=confidence,
                processing_time=processing_time,
                user_id=f"user_{i}"
            )
            
        except Exception as e:
            logger.log_error(str(e), {'text': text, 'user_id': f"user_{i}"})
    
    # Log model metrics
    logger.log_model_metrics({
        'total_predictions': len(texts),
        'avg_confidence': 0.85,
        'avg_processing_time': 0.02
    })
```

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

app = FastAPI()

# Prometheus metrics
PREDICTION_COUNTER = Counter('nlp_predictions_total', 'Total number of predictions')
PREDICTION_DURATION = Histogram('nlp_prediction_duration_seconds', 'Prediction duration')
MODEL_LOADED = Gauge('nlp_model_loaded', 'Model loaded status')
PREDICTION_CONFIDENCE = Histogram('nlp_prediction_confidence', 'Prediction confidence')

@app.middleware("http")
async def add_metrics(request, call_next):
    """
    Middleware to add metrics.
    """
    start_time = time.time()
    
    response = await call_next(request)
    
    # Record request duration
    duration = time.time() - start_time
    PREDICTION_DURATION.observe(duration)
    
    return response

@app.post("/predict")
async def predict_with_metrics(request: TextRequest):
    """
    Predict with metrics recording.
    """
    start_time = time.time()
    
    try:
        # Your prediction logic here
        result = await predict_sentiment(request)
        
        # Record metrics
        PREDICTION_COUNTER.inc()
        PREDICTION_CONFIDENCE.observe(result.confidence)
        
        return result
    
    except Exception as e:
        # Record error
        PREDICTION_COUNTER.inc()
        raise e

@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    """
    return PlainTextResponse(generate_latest())
```

## Best Practices

### 1. **Model Management**
- Version your models and configurations
- Use model registries (MLflow, DVC)
- Implement A/B testing for model updates
- Maintain model lineage and documentation

### 2. **Performance Optimization**
- Use model quantization for faster inference
- Implement batch processing for efficiency
- Use appropriate hardware (GPU/CPU)
- Optimize input preprocessing pipeline

### 3. **Scalability**
- Use load balancers for multiple instances
- Implement caching for repeated requests
- Use async processing for batch operations
- Monitor resource usage and scale accordingly

### 4. **Security**
- Validate and sanitize all inputs
- Implement rate limiting
- Use HTTPS for all communications
- Secure model storage and access

### 5. **Monitoring**
- Track prediction accuracy and drift
- Monitor system performance metrics
- Implement alerting for failures
- Log all predictions for analysis

### 6. **Testing**
- Unit tests for model components
- Integration tests for API endpoints
- Load testing for performance validation
- A/B testing for model comparison

## Conclusion

Successful NLP model deployment requires careful planning and implementation of production-ready systems. Focus on scalability, performance, monitoring, and maintainability to ensure your models serve users effectively in production environments.

Remember: **Good deployment practices lead to reliable and scalable NLP services!** 