#!/usr/bin/env python3
"""
This module contains a function that creates
a pandas DataFrame from a numpy.ndarray.
"""

import pandas as pd
import numpy as np


def from_numpy(array):
    """
    Function that creates a pandas DataFrame from a numpy.ndarray.

    Args:
        array (numpy.ndarray): The numpy.ndarray to convert.

    Returns:
        pd.DataFrame: The newly created DataFrame.
    """
    return pd.DataFrame(array)
