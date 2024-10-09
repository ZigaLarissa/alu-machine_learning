#!/usr/bin/env python3
"""
Attention is all we need.
"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    RNN Encoder.
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        RNN Encoder init function.
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units

        # Embedding layer to convert word indices to embedding vectors
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)

        # GRU layer with glorot_uniform initialization for recurrent weights
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def initialize_hidden_state(self):
        """
        Initializes hidden states to zeros with shape (batch, units)
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        - x is the input tensor of word indices
        - initial is the initial hidden state tensor
        """
        x = self.embedding(x)  # Convert word indices to embeddings
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
