import tensorflow as tf

def change_hue(image, delta):
    """
    Changes the hue of an image.

    Args:
    image: A 3D tf.Tensor representing the image to change.
    delta: The amount the hue should change.

    Returns:
    A 3D tf.Tensor representing the altered image.
    """
    return tf.image.adjust_hue(image, delta)
