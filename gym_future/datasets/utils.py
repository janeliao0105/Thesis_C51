# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 16:45:17 2022

@author: qwe41
"""

import os
import numpy as np


def load_dataset(name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, 'data', name + '.npy')
    f_array = np.load(path,allow_pickle=True)
    return f_array