# 02. Sequence-to-Sequence Modeling

## Introduction

Sequence-to-sequence (seq2seq) models are a class of neural architectures designed to transform one sequence into another, such as translating a sentence from English to French or summarizing a paragraph.

## Why Seq2Seq Models?

Many NLP tasks require mapping input sequences of variable length to output sequences of variable length. Traditional models struggle with this, but seq2seq models handle it naturally.

## Basic Seq2Seq Architecture

A typical seq2seq model consists of two main components:
- **Encoder:** Processes the input sequence and encodes it into a fixed-size context vector.
- **Decoder:** Generates the output sequence from the context vector.

### Encoder
At each time step $`t`$, the encoder RNN updates its hidden state:

```math
h_t^{enc} = \sigma(W_{hh}^{enc} h_{t-1}^{enc} + W_{xh}^{enc} x_t + b_h^{enc})
```

### Context Vector
After processing the input, the final hidden state $`h_T^{enc}`$ is used as the context vector $`c`$:

```math
c = h_T^{enc}
```

### Decoder
The decoder RNN generates the output sequence, one token at a time, using the context vector as its initial hidden state:

```math
h_0^{dec} = c
```
```math
h_t^{dec} = \sigma(W_{hh}^{dec} h_{t-1}^{dec} + W_{xh}^{dec} y_{t-1} + b_h^{dec})
```
where $`y_{t-1}`$ is the previous output token.

## Limitations and Attention Mechanism

- Vanilla seq2seq models struggle with long sequences because the context vector is a bottleneck.
- **Attention** allows the decoder to access all encoder hidden states, improving performance (see advanced guides).

## Python Example: Simple Seq2Seq with Keras

```python
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

# Toy data: input and output sequences (integer-encoded)
encoder_input_data = np.random.rand(100, 10, 8)  # 100 samples, 10 timesteps, 8 features
decoder_input_data = np.random.rand(100, 8, 8)   # 100 samples, 8 timesteps, 8 features

decoder_target_data = np.random.rand(100, 8, 16) # 100 samples, 8 timesteps, 16 features

# Encoder
encoder_inputs = Input(shape=(None, 8))
encoder = LSTM(32, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None, 8))
decoder_lstm = LSTM(32, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(16, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Seq2Seq Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.summary()
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, epochs=2, batch_size=8)
```

## Key Takeaways
- Seq2seq models map input sequences to output sequences of variable length.
- Consist of encoder and decoder RNNs.
- Attention mechanisms improve performance for long sequences.

## References
- [Sequence to Sequence Learning with Neural Networks (Sutskever et al., 2014)](https://arxiv.org/abs/1409.3215)
- [TensorFlow Seq2Seq Tutorial](https://www.tensorflow.org/text/tutorials/nmt_with_attention) 