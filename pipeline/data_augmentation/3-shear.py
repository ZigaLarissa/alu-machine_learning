#!/usr/bin/env python3
"""
   function def shear_image(image, intensity):
   that randomly shears an image:
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

def shear_image(image, intensity):
    """
    Randomly shears an image.

    Params:
        - image: 3D tf.Tensor
        - intensity: Intensity of the shear

    Returns:
        - Sheared image as a 3D tf.Tensor
    """
    # Convert tensor to array
    image = tf.keras.preprocessing.image.img_to_array(image)

    # Randomly shear image
    sheared_image = tf.keras.preprocessing.image.random_shear(image, intensity=intensity, row_axis=0, col_axis=1, channel_axis=2)

    # Convert array back to tensor
    return tf.convert_to_tensor(sheared_image, dtype=tf.float32)
