import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

def pca_color(image, alphas):
    """
    Performs PCA color augmentation as described in the AlexNet paper.

    Args:
    image: A 3D tf.Tensor containing the image to change.
    alphas: A tuple of length 3 containing the amount that each channel should change.

    Returns:
    A 3D tf.Tensor representing the augmented image.
    """
    # Convert the image to float32
    image = tf.cast(image, tf.float32)
    original_shape = tf.shape(image)

    # Flatten the image to shape (num_pixels, 3)
    flattened_image = tf.reshape(image, [-1, 3])

    # Compute the covariance matrix of the flattened image
    mean = tf.reduce_mean(flattened_image, axis=0)
    centered_image = flattened_image - mean
    cov = tf.matmul(centered_image, centered_image, transpose_a=True) / tf.cast(tf.shape(flattened_image)[0], tf.float32)

    # Perform eigen decomposition of the covariance matrix
    eigvals, eigvecs = tf.linalg.eigh(cov)

    # Perform PCA color augmentation
    delta = tf.matmul(eigvecs, tf.cast(alphas, tf.float32) * eigvals)
    augmented_image = flattened_image + delta

    # Reshape the augmented image back to its original shape
    augmented_image = tf.reshape(augmented_image, original_shape)

    # Clip the augmented image to be in the range [0, 255]
    augmented_image = tf.clip_by_value(augmented_image, 0.0, 255.0)

    return tf.cast(augmented_image, tf.uint8)

# Example usage:
if __name__ == "__main__":
    # Load a sample image
    doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
    for image, _ in doggies.shuffle(10).take(1):
        # Apply PCA color augmentation
        alphas = (0.1, 0.1, 0.1)  # Example alphas
        augmented_image = pca_color(image, alphas)
        plt.imshow(augmented_image)
        plt.show()
