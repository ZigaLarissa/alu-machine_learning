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
    num_columns = array.shape[1]
    columns = [chr(65 + i) for i in range(num_columns)]
    return pd.DataFrame(array, columns=columns)
