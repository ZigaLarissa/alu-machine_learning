#!/usr/bin/env python3
"""
Vanilla Autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    this function creates an autoencoder, where:

    input_dims: is an integer containing dimensions of the model input
    hidden_layers: Is a list containing the number of nodes for
    each hidden layer in the encoder, which is reversed for the decoder
    latent_dims: an integer containing the dimensions of the latent space
    representations

    optimisations:
    - The autoencoder model should be using adam optimisation and
    binary cross-entropy loss
    - All layers should use a relu activation, except for 
    - The last layer in the decoder which should use sigmoid

    Returns
    encoder: the encoder model
    decoder: the decoder model
    auto: the full autoencoder model
    """

    # Encoder model
    input_layer = keras.layers.Input(shape=(input_dims,))
    x = input_layer
    for nodes in hidden_layers:
        x = keras.layers.Dense(
            nodes,
            activation='relu'
        )(x)
    latent_layer = keras.layers.Dense(
        latent_dims,
        activation='relu'
    )(x)
    encoder = keras.models.Model(
        input=input_layer,
        output=latent_layer,
        name="encoder"
    )
