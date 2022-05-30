import gym
import numpy as np
import torch
import pandas as pd
import gym_anytrading
from torch import nn
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import C51Policy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net

from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from futures_env import FuturesTradingEnvrionment
writer = SummaryWriter('log/C51')
logger = TensorboardLogger(writer)

import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
features = np.load('FeatureMap.npy',allow_pickle=True)
env = FuturesTradingEnvrionment(features)
env_maker = DummyVectorEnv([lambda: FuturesTradingEnvrionment(features)])
train_envs = DummyVectorEnv([lambda: FuturesTradingEnvrionment(features) for _ in range(2000)])
test_envs = DummyVectorEnv([lambda: FuturesTradingEnvrionment(features) for _ in range(1000)])


state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
net = Net(state_shape, action_shape,num_atoms=51)
optim = torch.optim.Adam(net.parameters(), lr=1e-3)

policy = C51Policy(net, optim, discount_factor=0.9, estimation_step=3, target_update_freq=320)

train_collector = Collector(policy, train_envs, VectorReplayBuffer(20000, len(train_envs)))
test_collector = Collector(policy, test_envs)
train_collector.collect(n_step=5000, random=True)

# policy.set_eps(0.1)
# for i in range(int(1e6)):  # total step
#     collect_result = train_collector.collect(n_step=10)

#     # once if the collected episodes' mean returns reach the threshold,
#     # or every 1000 steps, we test it on test_collector
#     if  i % 1000 == 0:
#         policy.set_eps(0.05)
#         result = test_collector.collect(n_episode=100)
#         if result['rews'].mean() >= env.spec.reward_threshold:
#             print(f'Finished training! Test mean returns: {result["rews"].mean()}')
#             break
#         else:
#             # back to training eps
#             policy.set_eps(0.1)

#     # train policy with a sampled batch data from buffer
#     losses = policy.update(64, train_collector.buffer)

result = offpolicy_trainer(
    policy, train_collector, test_collector,
    max_epoch=1000, step_per_epoch=10000, step_per_collect=10,
    update_per_step=0.1, episode_per_test=100, batch_size=64,
    train_fn=lambda epoch, env_step: policy.set_eps(0.1),
    test_fn=lambda epoch, env_step: policy.set_eps(0.05))
print(f'Finished training! Use {result["duration"]}')

print(result)

# Let's watch its performance!
policy.eval()
result = test_collector.collect(n_episode=1, render=False)
print("Final reward: {}, length: {}".format(result["rews"].mean(), result["lens"].mean()))