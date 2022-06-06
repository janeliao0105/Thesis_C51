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
from gym.utils import seeding


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
    self.bull=[]
    self.bear=[]
    self.done = False
    self.total_reward = 0
    self.reward = 0

    
  def seed(self,seed=None):
      self.np_random, seed = seeding.np_random(seed)
      return [seed]
    
  def step(self, action):
    # Execute one time step within the environment
    current_price = self.featue_array[self.current_step,self.current_step%404,3]
    if action == 0:
        self.bear.append(current_price)
    elif action == 1:
        self.bull.append(current_price)
    
    if self.current_step == len(self.featue_array)-2:
        self.done = True
        
    self.current_step += 1
    if self.current_step % 404 == 0:
        self.reward = sum(current_price - self.bull) + sum(self.bear - current_price )
        
        self.bear = []
        self.bull = []
    else: self.reward = 0
    # reward = sum(current_price - self.bull) + sum(self.bear - current_price )
    
    self.total_reward += self.reward
    
    obs = self._next_observation()
    
    return obs, self.reward, self.done, {}  
      
  def _next_observation(self):

    obs = self.featue_array[self.current_step].flatten()
    return obs

  
  
   # def _take_action(self,action):
   #     current_price = self.featue_array[self.current_step,self.current_step%404,3]
   #     action_type = self.action_space.sample()
   #     if action_type == 0:
   #         self.bear.append(current_price)
          
   #     else:
   #         self.bull.append(current_price)
      
     
    
  def reset(self):
    # Reset the state of the environment to an initial state
    self.current_step = 1
    self.bull = []
    self.bear = []
    self.total_reward = 0
    return self._next_observation()


  def render(self, mode='human', close=False):
      print(f'Step: {self.current_step}')
      print(f'Reward: {self.total_reward}')
