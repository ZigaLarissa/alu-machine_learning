#!/usr/bin/env python3
"""
Function to perform a padding convolution on grayscale images
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Function to perform a padding convolution on grayscale images
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
        padding: a tuple of (ph, pw)
                ph is the padding for the height of the image
                pw is the padding for the width of the image
    Returns: numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    if kh % 2 == 1:
        ph = (kh - 1) // 2
    else:
        ph = kh // 2
    if kw % 2 == 1:
        pw = (kw - 1) // 2
    else:
        pw = kw // 2
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                    'constant', constant_values=0)
    conv = np.zeros((m, h, w))
    for i in range(h):
        for j in range(w):
            image = padded[:, i:i+kh, j:j+kw]
            conv[:, i, j] = np.sum(image * kernel, axis=(1, 2))
    return conv