import tensorflow as tf

def crop_image(image, size):
    """
    Performs a random crop of an image.

    Args:
    image: A 3D tf.Tensor representing the image to crop.
    size: A tuple containing the size of the crop (crop_height, crop_width).

    Returns:
    A 3D tf.Tensor representing the cropped image.
    """
    return tf.image.random_crop(image, size)
