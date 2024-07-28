import tensorflow as tf

def rotate_image(image):
    """
    Rotates an image by 90 degrees counter-clockwise.

    Args:
    image: A 3D tf.Tensor representing the image to rotate.

    Returns:
    A 3D tf.Tensor representing the rotated image.
    """
    return tf.image.rot90(image)
