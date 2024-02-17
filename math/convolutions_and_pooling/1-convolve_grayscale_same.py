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

    m, h, w = images.shape
    kh, kw = kernel.shape

    ph = max(int((kh - 1) / 2), int(kh / 2))
    pw = max(int((kw - 1) / 2), int(kw / 2))

    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    convolved_images = np.zeros((m, h, w))

    for i in range(m):
        shadow_region = np.lib.stride_tricks.sliding_window_view(padded_images[i], (kh, kw))
        convolved_images[i] = np.sum(shadow_region * kernel, axis = (1, 2))
    return convolved_images
