import os
import numpy as np

import gym
from gym import Env
import torch as th

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

from helper import plots_norm


import tria_rl
#env = gym.make('tria_rl/TriaClimate-v0')
env_id = 'tria_rl/TriaClimate-v0'
env_s = gym.make(env_id)

print('---- observation space attributes ----')
print('observation space size:   ',env_s.observation_space.shape[0])
print('observation space sample: ',env_s.observation_space.sample)

print('---- action space attributes ----')
print('action space: ', env_s.action_space)
print('action space sample: ', env_s.action_space.sample())

#env = make_vec_env(env_id, n_envs=4)

env_s = DummyVecEnv([lambda: env_s])

env_s = VecNormalize(env_s, training=True, norm_obs=True, norm_reward=True, epsilon=1e-08, gamma=0.99)

log_path = os.path.join('train', 'log')

tria_a2c_model_path = os.path.join('train','save', "tria_a2c_normalized")

model = A2C.load(tria_a2c_model_path, env=env_s)

evaluate_policy(model, env_s, n_eval_episodes=20, render=False)

env_s.close()

print('* * * Tria A2C model for tria 3D environment predictions * * *')

episodes=1000
plot_scores= [[0] * episodes for i in range(2)]
plot_mean_scores=[[0] * episodes for i in range(2)]
for episode in range(1, episodes):
    env_s.state = np.array([[81, 61, 201 ]])#env_s.reset()
    observation = np.array([[81, 61, 201 ]])
    print('>>obs>>',observation)
    terminated = False
    score = 0
    norm_score=0
    game=0
    while not terminated:
        #env.render()
        action, _ = model.predict(observation, deterministic=True)
        observation, norm_reward, terminated , info = env_s.step(action)
        norm_score += norm_reward
        score +=env_s.get_original_reward()
        game +=1
        print('norm_obs: {} observation: {} action: {} norm reward: {} reward: {}'.format(observation, env_s.get_original_obs(), action, norm_reward, env_s.get_original_reward()));
    print('Model Name: {} Episone:{} Score:{}'.format( "tria_a2c_normalized", episode, score))
    mean_norm_score = norm_score / game
    mean_score = score / game
    plot_scores[0][episode] =  score 
    plot_mean_scores[0][episode] = mean_score 

    plot_scores[1][episode] =  norm_score 
    plot_mean_scores[1][episode] = mean_norm_score 

env_s.close() 

print('------------------------------------------------------------------')


plots_norm(plot_scores, plot_mean_scores)