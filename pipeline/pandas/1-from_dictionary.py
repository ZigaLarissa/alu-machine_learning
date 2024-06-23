#!/usr/bin/env python3
"""
This module contains a script that creates
a pandas DataFrame from a dictionary.
"""
import pandas as pd
import numpy as np


dictionary = {
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four']
}

df = pd.DataFrame(dictionary, index=['A', 'B', 'C', 'D'])
