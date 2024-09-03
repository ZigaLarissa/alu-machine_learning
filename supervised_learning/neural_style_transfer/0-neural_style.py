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

        # Sets Tensorflow to execute eagerly
        tf.enable_eager_execution()

        # Set instance attributes
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta


    # Scale image method
    def scale_image(self, image):
        # Check if image is not a np.ndarray with the shape (h, w, 3)
        if not isinstance(image, np.ndarray) or len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")
        
        # Check if image is not already between 0 and 1
        if np.max(image) > 1:
            image = image / 255

        # Check if image is larger than 512 pixels in the y axis
        h, w, _ = image.shape
        if h > 512:
            h = 512
            w = int(w * (512 / h))
        
        # Check if image is larger than 512 pixels in the x axis
        if w > 512:
            w = 512
            h = int(h * (512 / w))
        
        # Resize the image with inter-cubic interpolation
        image = tf.image.resize_bicubic(tf.expand_dims(image, 0), (h, w))
        image = image / 255
        return image
