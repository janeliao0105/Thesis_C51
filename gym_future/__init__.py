# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 19:56:50 2022

@author: qwe41
"""

from gym.envs.registration import register
from copy import deepcopy
from . import datasets



register(
    id='futures-v1',
    entry_point='gym_future.envs:FuturesTradingEnvrionment',
     kwargs={
        'featue_array': deepcopy(datasets.featue_array),
    }
)
