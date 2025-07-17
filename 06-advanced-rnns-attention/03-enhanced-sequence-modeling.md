# 03. Enhanced Sequence Modeling

## Introduction

Enhanced sequence modeling combines advanced RNNs (LSTM, GRU) and attention mechanisms to achieve state-of-the-art performance on complex NLP tasks. These techniques enable models to capture long-range dependencies, focus on relevant information, and handle variable-length sequences.

## Why Enhance Sequence Models?

- Vanilla RNNs struggle with long-term dependencies and information bottlenecks.
- LSTM/GRU mitigate vanishing gradients, but still compress all information into a single vector in seq2seq models.
- Attention allows dynamic focus on different parts of the input, improving performance.

## Combining LSTM/GRU with Attention

A typical enhanced sequence model for tasks like translation or summarization:
- **Encoder:** LSTM/GRU processes input sequence, outputs hidden states $`h_1, ..., h_T`$.
- **Attention:** Computes context vector $`c_t`$ for each decoder step.
- **Decoder:** LSTM/GRU generates output, conditioned on $`c_t`$ and previous outputs.

### Mathematical Formulation

At decoder time step $`t`$:

```math
c_t = \sum_{i=1}^T \alpha_{t,i} h_i
```
where $`\alpha_{t,i}`$ are attention weights.

The decoder hidden state is updated as:

```math
h_t^{dec} = \text{LSTM}(y_{t-1}, h_{t-1}^{dec}, c_t)
```

## Python Example: LSTM with Attention (Keras AdditiveAttention)

```python
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, AdditiveAttention

# Toy data
encoder_input = np.random.rand(100, 10, 8)
decoder_input = np.random.rand(100, 8, 8)
decoder_target = np.random.rand(100, 8, 16)

# Encoder
encoder_inputs = Input(shape=(None, 8))
encoder_lstm = LSTM(32, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None, 8))
decoder_lstm = LSTM(32, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

# Attention
attention = AdditiveAttention()
context = attention([decoder_outputs, encoder_outputs])
concat = Dense(32, activation='tanh')(context)
output = Dense(16, activation='softmax')(concat)

model = Model([encoder_inputs, decoder_inputs], output)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.summary()
model.fit([encoder_input, decoder_input], decoder_target, epochs=2, batch_size=8)
```

## Applications
- Neural machine translation
- Text summarization
- Dialogue systems
- Speech recognition

## Key Takeaways
- Enhanced sequence models combine LSTM/GRU and attention for superior performance.
- Attention enables dynamic focus on relevant input parts.
- Widely used in modern NLP systems.

## References
- [Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2015)](https://arxiv.org/abs/1409.0473)
- [TensorFlow AdditiveAttention Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/AdditiveAttention) 