import os
import numpy as np
import random
import csv
import gym
from gym import Env
import torch as th

from copy import deepcopy
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from gym.spaces import Discrete, Dict, Box

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

from helper import plots_3d, plotPredictions, plotAnimation

import tria_rl
env_id = 'tria_rl/TriaClimate-v0'
env_s = gym.make(env_id)

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

tria_ppo_model_path = os.path.join('train','save', "tria_ppo_normalized")

model = PPO.load(tria_ppo_model_path, env=env_s)

#evaluate_policy(model, env_s, n_eval_episodes=20, render=False)

env_s.close()

print('* * * Tria A2C model for tria 3D environment predictions * * *')

episodes=4
rows =2
cols =2

acts = []
obss = []
clrss = []
rwds = [] 
for episode in range(0, episodes):
    observation = env_s.reset()
    terminated = False
    score = 0
    norm_score=0
    act=[]
    obs=[]
    clrs=[]
    rwd = []
    while not terminated:        
        ob = env_s.get_original_obs().ravel().tolist()
        obs.append(ob)
        action, _ = model.predict(observation, deterministic=False)
        observation, norm_reward, terminated , info = env_s.step(action)
        norm_score += norm_reward
        score +=env_s.get_original_reward()

        act.append(action.tolist()[0])
        clrs.append(getColor(env_s.get_original_reward()[0]))
        r = env_s.get_original_reward().tolist()[0]
        rwd.append(r)
        print('E: {} O: {} A: {} R: {}'.format( episode, [str(round(o,2)) for o in ob], action.tolist()[0], r))
    print('Model Name: {} Episone:{} Score:{}'.format( "tria_ppo_normalized", episode, score))

    acts.append(act)
    obss.append(obs)
    clrss.append(clrs)
    rwds.append(rwd)
env_s.close() 

print('------------------------------------------------------------------')

plots_3d(np.array(obss),acts,clrss, rows, cols)
plotPredictions(rwds, obs, acts, clrss, rows, cols)
plotAnimation(obss[3])
#plots_norm(plot_scores, plot_mean_scores)