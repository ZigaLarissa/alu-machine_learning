#!/usr/bin/env python3
"""
Attention is all we need.
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    SelfAttention.
    """
    def __init__(self, units):
        """
        Innitialised
        """
        super(SelfAttention, self).__init__()
        # Dense layer for previous decoder hidden state
        self.W = tf.keras.layers.Dense(units)
        # Dense layer for encoder hidden states
        self.U = tf.keras.layers.Dense(units)
        # Dense layer for scoring (final attention weights)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        s_prev: Previous decoder hidden state, shape (batch, units)
        hidden_states: Encoder hidden states,
        shape (batch, input_seq_len, units)
        Returns:
          - context: Context vector, shape (batch, units)
          - weights: Attention weights,
          shape (batch, input_seq_len, 1)
        """
        # Expand s_prev to (batch, 1, units) so it
        # can be added to hidden_states
        s_prev_expanded = tf.expand_dims(s_prev, 1)

        # Score calculation: V(tanh(W(s_prev) + U(hidden_states)))
        score = self.V(
            tf.nn.tanh(self.W(s_prev_expanded) + self.U(hidden_states))
            )

        # Softmax over the scores to get attention weights
        weights = tf.nn.softmax(score, axis=1)

        # Calculate the context vector as the weighted sum of hidden_states
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
