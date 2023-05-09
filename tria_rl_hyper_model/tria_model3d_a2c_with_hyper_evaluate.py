#!/usr/bin/env python3

import os
import numpy as np
import random

import gym
from gym import Env
from gym.spaces import Discrete, Box, MultiDiscrete
import torch as th

#import tensorflow as tf

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C
#load_ext tf.tensorboard
import tria_rl

env_name = 'tria-3d-rl-model-'

a2c_model_timesteps=2000000

ppo_model_name = env_name + 'ppo'; neural_model_name = env_name + 'ppo-neural'; a2c_model_name = env_name + 'a2c'

''' * * *  tria logging folders * * * '''
log_path = os.path.join('train','log')
print(log_path)

''' * * *  tria model save folder * * * '''
tria_a2c_model_path = os.path.join('train','save', a2c_model_name)
print(tria_a2c_model_path)
''' * * * gym tria environment instance * * * '''


env = gym.make('tria_rl/TriaClimate-v0') #TriaEnv()
print(env.metadata)
print('------------------------------------------------------------------')
print("1. Sample observation space: {}".format(env.observation_space.sample()))
print("1. Sample observation space: {}".format(env.observation_space))
print("1. Sample observation space: {}".format(env.observation_space.dtype))
print("2. Sample action space     : {}".format(env.action_space.sample()))
print("3. Sample state            : {}".format(env.state))    
print('------------------------------------------------------------------')

#print('------------------------------------------------------------------')

print('* * * Tria A2C network model for tria 3D environment * * *')

''' load model in memory to validate model persistance '''
a2c_model = A2C.load(tria_a2c_model_path, env=env)

evaluate_policy(a2c_model, env, n_eval_episodes=50, render=False)

env.close()

print('* * * Tria A2C model for tria 3D environment predictions * * *')

episodes=50
for episode in range(1, episodes+1):
    observation = env.reset()
    terminated = False
    score = 0
    while not terminated:
        #env.render()
        action, _ = a2c_model.predict(observation, deterministic=False)
        observation, reward, terminated , info = env.step(action)
        score += reward
        print('observation: {} action: {}'.format(observation, action));
    print('Model Name: {} Episone:{} Score:{}'.format( a2c_model_name, episode, score))

env.close() 

print('------------------------------------------------------------------')
