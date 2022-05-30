# -*- coding: utf-8 -*-
"""
Created on Fri May 27 14:59:49 2022

@author: qwe41
"""

from futures_env import FuturesTradingEnvrionment
from stable_baselines3.common.env_checker import check_env
import numpy as np
# from stable_baselines3.dqn.policies.DQNPolicy import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3 import PPO

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


features = np.load('FeatureMap.npy',allow_pickle=True)
# test_featue = features[:40500]
# env = FuturesTradingEnvrionment(features)
# check_env(env)
env = DummyVecEnv([lambda: FuturesTradingEnvrionment(features)])
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(100):
  action, _states = model.predict(obs)
  obs, rewards, done, info = env.step(action)
  env.render()