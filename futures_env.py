# -*- coding: utf-8 -*-
"""
Created on Thu May 26 17:05:57 2022

@author: qwe41
"""

import gym
from gym import spaces
import pandas as pd
import numpy as np
import random
import json


BEAR = 0
BULL = 1


class FuturesTradingEnvrionment(gym.Env):
  """A futures trading envrioment for OpenAI gym"""
  metadata = {'render.modes': ['human']}

  def __init__(self, featue_array):
    super(FuturesTradingEnvrionment, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.featue_array = featue_array
    self.action_space = spaces.Discrete(2)
    self.observation_space = spaces.Box(low=0,high=50000,shape=(2025,),dtype=np.float32)
    self.current_step = 1
    self.shares_held = 0
    self.total_sales_value = 0
    self.total_shares_sold = 0
    self.bull=[]
    self.bear=[]
    self.done = False
  
    
      
  def _next_observation(self):
  # Get the data points for the last 5 days and scale to between 0-1
    obs = self.featue_array[self.current_step].flatten()
    return obs

  def step(self, action):
    # Execute one time step within the environment
    self._take_action(action)
    if self.current_step < len(self.featue_array)-1:
        self.current_step = 0
    else:
        self.current_step += 1
    
    if self.current_step % 405 == 0:
        reward = sum(self.featue_array[self.current_step,404,3] - self.bull)
        self.bear = []
        self.bull = []
    else: reward = 0
    
    if self.current_step > len(self.featue_array):
        self.done = True
    obs = self._next_observation()
    
    return obs, reward, self.done, {}
  
  def _take_action(self,action):
      current_price = self.featue_array[self.current_step,self.current_step%405,3]
      action_type = self.action_space.sample()
      if action_type == 0:
          self.bear.append(current_price)
          
      else:
          self.bull.append(current_price)
      
     
    
  def reset(self):
    # Reset the state of the environment to an initial state
    self.current_step = 1
    self.bull = []
    self.bear = []
    return self._next_observation()


  def render(self, mode='human', close=False):
      print(f'Step: {self.current_step}')

