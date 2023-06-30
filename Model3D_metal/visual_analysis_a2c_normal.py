import os
import numpy as np
import random
import csv
import gym
from gym import Env
import torch as th

from copy import deepcopy
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from gym.spaces import Discrete, Dict, Box

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

from helper import plots_norm, plots_3d

import tria_rl
env_id = 'tria_rl/TriaClimate-v0'
env_s = gym.make(env_id)
""" class TriaEnv:
    def __init__(self, config=None):
        self.env = gym.make(env_id)
        self.action_space = Discrete(14)
        self.observation_space = self.env.observation_space

    def reset(self):
       return self.env.reset()

    def step(self, action):
       obs, rew, done, info = self.env.step(action)
       return obs, rew, done, info

    def set_state(self, state):
       self.env = deepcopy(state)
       obs = np.array(list(self.env.unwrapped.state))
       return obs

    def get_state(self):
        return deepcopy(self.env)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

env_s = TriaEnv() """

def getColor (reward):
    if reward <=30 and reward >= 10:
        return 'green'
    elif reward < 10 and reward > -10:
        return 'blue'
    else: 
        return 'red'


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

episodes=5

plot_scores= [[0] * episodes for i in range(2)]
plot_mean_scores=[[0] * episodes for i in range(2)]
#act_obs_clr =[[0]* episodes, [[]]* episodes, [''] * episodes]
acts = []
obss = []
clrss = []
for episode in range(1, episodes):
    observation = env_s.reset()
    terminated = False
    score = 0
    norm_score=0
    game=0
    act=[]
    obs=[]
    clrs=[]
    while not terminated:
        action, _ = model.predict(observation, deterministic=True)
        observation, norm_reward, terminated , info = env_s.step(action)
        norm_score += norm_reward
        score +=env_s.get_original_reward()
        game +=1
        #print(act, action.tolist()[0])
        act.append(action.tolist()[0])
        ob = env_s.get_original_obs().ravel().tolist()
        clrs.append(getColor(env_s.get_original_reward()[0]))
        obs.append(ob)
    print('Model Name: {} Episone:{} Score:{}'.format( "tria_a2c_normalized", episode, score))
    
    mean_norm_score = norm_score / game
    mean_score = score / game
    plot_scores[0][episode] =  score 
    plot_mean_scores[0][episode] = mean_score 

    plot_scores[1][episode] =  norm_score 
    plot_mean_scores[1][episode] = mean_norm_score 

    acts.append(act)
    obss.append(obs)
    clrss.append(clrs)

    #print(np.array(obss))
env_s.close() 

print('------------------------------------------------------------------')

#print(np.array(obss)[0][:,0])
plots_3d(np.array(obss),acts,clrss)
plots_norm(plot_scores, plot_mean_scores)