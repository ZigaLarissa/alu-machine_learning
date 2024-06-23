#!/usr/bin/env python3
"""
This module contains a function that loads a pandas
DataFrame from a file."""

import pandas as pd
import numpy as np


def from_file(filename, delimiter):
    """
    Function that loads a pandas DataFrame from a file.

    Args:
        filename (str): The path to the file.
        delimiter (str): The column separator.

    Returns:
        pd.DataFrame: The newly created DataFrame.
    """
    return pd.read_csv(filename, delimiter=delimiter)
