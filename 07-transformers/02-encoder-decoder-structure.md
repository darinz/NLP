# 02. Encoder-Decoder Structure

## Introduction

The encoder-decoder structure is a neural network architecture designed to map input sequences to output sequences, and is the backbone of the Transformer model. It is widely used in tasks like machine translation, summarization, and question answering.

## Why Encoder-Decoder?

- Many NLP tasks require transforming one sequence into another (e.g., English to French translation).
- The encoder processes the input and creates a rich representation; the decoder generates the output sequence based on this representation.

## Encoder-Decoder in Transformers

The Transformer architecture consists of:
- **Encoder stack:** Processes the input sequence and produces a sequence of hidden representations.
- **Decoder stack:** Generates the output sequence, attending to both previous outputs and the encoder's representations.

## Encoder Block
Each encoder block contains:
- Multi-head self-attention
- Add & Norm
- Feed-forward network
- Add & Norm

### Encoder Block Math

Given input $`X`$:

```math
H' = \text{LayerNorm}(X + \text{MultiHeadSelfAttention}(X))
H = \text{LayerNorm}(H' + \text{FeedForward}(H'))
```

## Decoder Block
Each decoder block contains:
- Masked multi-head self-attention (prevents attending to future tokens)
- Add & Norm
- Multi-head attention over encoder outputs
- Add & Norm
- Feed-forward network
- Add & Norm

### Decoder Block Math

Given previous outputs $`Y`$ and encoder outputs $`E`$:

```math
G' = \text{LayerNorm}(Y + \text{MaskedMultiHeadSelfAttention}(Y))
G'' = \text{LayerNorm}(G' + \text{MultiHeadAttention}(G', E))
G = \text{LayerNorm}(G'' + \text{FeedForward}(G''))
```

## Python Example: Transformer Encoder-Decoder (with TensorFlow)

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout
from tensorflow.keras.models import Model

# Encoder block
def encoder_block(embed_dim, num_heads, ff_dim, rate=0.1):
    inputs = Input(shape=(None, embed_dim))
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    attn = Dropout(rate)(attn)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn)
    ffn = Dense(ff_dim, activation='relu')(out1)
    ffn = Dense(embed_dim)(ffn)
    ffn = Dropout(rate)(ffn)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn)
    return Model(inputs, out2)

# Decoder block (simplified, no masking for brevity)
def decoder_block(embed_dim, num_heads, ff_dim, rate=0.1):
    inputs = Input(shape=(None, embed_dim))
    enc_outputs = Input(shape=(None, embed_dim))
    attn1 = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    attn1 = Dropout(rate)(attn1)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn1)
    attn2 = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(out1, enc_outputs)
    attn2 = Dropout(rate)(attn2)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + attn2)
    ffn = Dense(ff_dim, activation='relu')(out2)
    ffn = Dense(embed_dim)(ffn)
    ffn = Dropout(rate)(ffn)
    out3 = LayerNormalization(epsilon=1e-6)(out2 + ffn)
    return Model([inputs, enc_outputs], out3)

# Example usage
embed_dim = 32
num_heads = 2
ff_dim = 64
encoder = encoder_block(embed_dim, num_heads, ff_dim)
decoder = decoder_block(embed_dim, num_heads, ff_dim)

x = tf.random.normal((1, 10, embed_dim))
enc_out = encoder(x)
y = tf.random.normal((1, 8, embed_dim))
dec_out = decoder([y, enc_out])
print("Encoder output shape:", enc_out.shape)
print("Decoder output shape:", dec_out.shape)
```

## Key Takeaways
- Encoder-decoder structure is fundamental for sequence-to-sequence tasks.
- Transformers use stacks of encoder and decoder blocks with self-attention and feed-forward layers.
- Parallelization and attention enable efficient, powerful models.

## References
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) 