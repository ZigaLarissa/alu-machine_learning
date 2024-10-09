#!/usr/bin/env python3
"""
Attention is all we need.
"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention

class RNNDecoder(tf.keras.layers.Layer):
    """
    RNN Decoder
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        initialized.
        """
        super(RNNDecoder, self).__init__()
        self.units = units
        self.batch = batch

        # Embedding layer to convert words into embedding vectors
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)

        # GRU layer for decoding
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

        # Dense layer to produce logits over the target vocabulary
        self.F = tf.keras.layers.Dense(vocab)

        # Attention mechanism to get the context vector
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        x: Previous word in the target sequence, shape (batch, 1)
        s_prev: Previous decoder hidden state, shape (batch, units)
        hidden_states: Encoder hidden states, shape
        (batch, input_seq_len, units)
        Returns:
          - y: Output word as a one-hot vector in the target
           vocabulary, shape (batch, vocab)
          - s: New decoder hidden state, shape (batch, units)
        """
        # Get the context vector from the attention mechanism
        context, _ = self.attention(s_prev, hidden_states)

        # Embed the previous word (x) in the target sequence
        x = self.embedding(x)  # Shape: (batch, 1, embedding)

        # Concatenate context vector with embedded input (x)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)

        # Pass concatenated input and previous state through the GRU
        output, s = self.gru(x, initial_state=s_prev)

        # Reshape output and pass through the dense layer
        # to get vocab-size logits
        output = tf.reshape(output, (-1, output.shape[2]))
        y = self.F(output)  # Shape: (batch, vocab)

        return y, s
