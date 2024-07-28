import tensorflow as tf

def flip_image(image):
    """
    Flips an image horizontally.

    Args:
    image: A 3D tf.Tensor representing the image to flip.

    Returns:
    A 3D tf.Tensor representing the horizontally flipped image.
    """
    return tf.image.flip_left_right(image)
