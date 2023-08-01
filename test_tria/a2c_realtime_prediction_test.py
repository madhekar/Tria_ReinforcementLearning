import os
import sys
import numpy as np

import gym
from gym import Env
import torch as th
import torch.nn as nn

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

from realtime_helper import plot


import tria_rl
env_id = 'tria_rl/TriaClimate-v0'
env_s = gym.make(env_id)

print('---- observation space attributes ----')
print('observation space size:   ',env_s.observation_space.shape[0])
print('observation space sample: ',env_s.observation_space.sample)

print('---- action space attributes ----')
print('action space: ', env_s.action_space)
print('action space sample: ', env_s.action_space.sample())

env_s = DummyVecEnv([lambda: env_s])

env_s = VecNormalize(env_s, training=True, norm_obs=True, norm_reward=True, epsilon=1e-08, gamma=0.99)

log_path = os.path.join('test', 'log')

tria_a2c_model_path = os.path.join('test','model', 'tria_a2c_normalized')

print('loading tranined model at: ', tria_a2c_model_path)

model = A2C.load(tria_a2c_model_path, env=env_s)

evaluate_policy(model, env_s, n_eval_episodes=20, render=False)

env_s.close()

print('* * * Tria A2C model for tria 3D environment predictions * * *')

episodes=1000
plot_scores= []
plot_mean_scores=[]
actions = []
score = 0
game=0
total_score=0
for episode in range(1, episodes):
    observation = env_s.reset()
    terminated = False
    score = 0
    norm_score=0
    while not terminated:
        #env.render()
        action, _ = model.predict(observation, deterministic=True)
        observation, norm_reward, terminated , info = env_s.step(action)
        score =  env_s.get_original_reward()
        game +=1

        actions.append(action)
        plot_scores.append(score)
        total_score += score
        mean_score = total_score / game
        plot_mean_scores.append(mean_score)
        plot(plot_scores, plot_mean_scores, actions)
        print('norm_obs: {} observation: {} action: {} norm reward: {} reward: {}'.format(observation, env_s.get_original_obs(), action, norm_reward, env_s.get_original_reward()));
        #print('norm_obs: {} action: {} reward: {}'.format(observation, action, norm_reward));
    print('Model Name: {} Episone:{} Score:{}'.format( "tria_a2c_normalized", episode, score))

env_s.close() 

print('------------------------------------------------------------------')