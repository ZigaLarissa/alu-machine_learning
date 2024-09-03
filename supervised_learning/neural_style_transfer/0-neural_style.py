#!/usr/bin/env python3

import tensorflow as tf
import numpy as np # type: ignore

# Neural Style Transfer (NST) class

class NST:
    # Public class attributes
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    # Class constructor
    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        # Check if style_image is not a np.ndarray with the shape (h, w, 3)
        if not isinstance(style_image, np.ndarray) or len(style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        
        # Check if content_image is not a np.ndarray with the shape (h, w, 3)
        if not isinstance(content_image, np.ndarray) or len(content_image.shape) != 3 or content_image.shape[2] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")
        
        # Check if alpha is not a non-negative number
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        
        # Check if beta is not a non-negative number
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        # Set TensorFlow to execute eagerly
        tf.config.experimental_run_functions_eagerly(True)

        # Set instance attributes
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta


    # Static method to scale image
    @staticmethod
    def scale_image(image):
        # Check if image is not a np.ndarray with the shape (h, w, 3)
        if not isinstance(image, np.ndarray) or len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")
        
        # Get the dimensions of the image
        h, w, _ = image.shape
        
        # Calculate the scale factor
        if h > w:
            new_h = 512
            new_w = int(w * (512 / h))
        else:
            new_w = 512
            new_h = int(h * (512 / w))
        
        # Resize the image using bicubic interpolation
        image = tf.image.resize(image, (new_h, new_w), method='bicubic')
        
        # Rescale pixel values to be between 0 and 1
        image = image / 255.0
        
        # Add a new batch dimension
        image = tf.expand_dims(image, axis=0)
        
        return image
