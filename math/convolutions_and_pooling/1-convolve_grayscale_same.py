#!/usr/bin/env python3

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Function to perform a same convolution on grayscale images
    Args:
        images: numpy.ndarray with shape (m, h, w) containing multiple
                grayscale images
                m is the number of images
                h is the height in pixels of the images
                w is the width in pixels of the images
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel
                for the convolution
                kh is the height of the kernel
                kw is the width of the kernel
    Returns: numpy.ndarray containing the convolved images
    """
    m, n = kernel.shape
    if m == n:
        i, y, x = images.shape
        y = y - m + 1
        x = x - m + 1
        convolved_images = np.zeros((i, y, x))
        for i in range(y):
            for j in range(x):
                shadow_area = images[:, i:i + m, j:j + n]
                convolved_images[:, i, j] = \
                    np.sum(shadow_area * kernel, axis=(1, 2))
        return convolved_images
