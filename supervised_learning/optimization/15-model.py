#!/usr/bin/env python3
"""
Define a model.
"""

import tensorflow as tf


def model(
        Data_train,
        Data_valid,
        layers,
        activations,
        alpha=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        decay_rate=1,
        batch_size=32,
        epochs=5,
        save_path='/tmp/model.ckpt'
        ):
    
