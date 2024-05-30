#!/usr/bin/env python3
"""
Create a layer of neural network
with dropout regularization
"""


import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
   """
   Creates a new layer in a neural network with dropout regularization.

   Args:
       prev (tf.Tensor): Output tensor from the previous layer.
       n (int): Number of nodes in the new layer.
       activation (callable): Activation function to apply to the new layer.
       keep_prob (float): Probability of keeping a node during dropout.

   Returns:
       tf.Tensor: Output tensor of the new layer.
   """
   # Initialize weights and biases
   init = tf.contrib.layers.xavier_initializer()
   weights = tf.Variable(init(shape=[prev.get_shape().as_list()[-1], n]))
   biases = tf.Variable(tf.zeros([n]))

   # Apply dropout to the previous layer's output
   dropout = tf.nn.dropout(prev, keep_prob)

   # Compute the output of the new layer
   layer = tf.matmul(dropout, weights) + biases
   layer = activation(layer)

   return layer
