import tensorflow as tf

def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image.

    Args:
    image: A 3D tf.Tensor representing the image to change.
    max_delta: The maximum amount the image should be brightened (or darkened).

    Returns:
    A 3D tf.Tensor representing the altered image.
    """
    return tf.image.random_brightness(image, max_delta)
