#!/usr/bin/env python3

import os
import numpy as np
import random

import gym
from gym import Env
from gym.spaces import Discrete, Box, MultiDiscrete

#import tensorflow as tf

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C
#load_ext tf.tensorboard

env_name = 'tria-3d-rl-model-'

ppo_model_timesteps= 200000; neural_model_timesteps=200000; a2c_model_timesteps=200000

ppo_model_name = env_name + 'ppo'; neural_model_name = env_name + 'ppo-neural'; a2c_model_name = env_name + 'a2c'

''' * * *  tria logging folders * * * '''
log_path = os.path.join('train','log')
print(log_path)

''' * * * gym tria environment instance * * * '''

import tria_rl
env = gym.make('tria_rl/TriaClimate-v0') #TriaEnv()
print(env.metadata)
print('------------------------------------------------------------------')
print("1. Sample observation space: {}".format(env.observation_space.sample()))
print("1. Sample observation space: {}".format(env.observation_space))
print("1. Sample observation space: {}".format(env.observation_space.dtype))
print("2. Sample action space     : {}".format(env.action_space.sample()))
print("3. Sample state            : {}".format(env.state))    
print('------------------------------------------------------------------')

print('------------------------------------------------------------------')

'''  *  Tria A2C  Network  * '''
'''
class stable_baselines.a2c.A2C(policy,         
                               env, 
                               gamma=0.99, 
                               n_steps=5, 
                               vf_coef=0.25, 
                               ent_coef=0.01, 
                               max_grad_norm=0.5, 
                               learning_rate=0.0007, 
                               alpha=0.99, 
                               momentum=0.0, 
                               epsilon=1e-05, 
                               lr_schedule='constant', 
                               verbose=0, 
                               tensorboard_log=None, 
                               _init_setup_model=True, 
                               policy_kwargs=None, 
                               full_tensorboard_log=False, 
                               seed=None, 
                               n_cpu_tf_sess=None)
'''
print('* * * Tria A2C network model for tria 3D environment * * *')

net_arch = dict(pi=[128,128,128,128], vf=[128,128,128,128])

a2c_model = A2C("MlpPolicy", 
                env, 
                verbose=1, 
                learning_rate= 0.0004833166401413716, 
                gamma=0.0016248762308103,
                lr_schedule='constant',
                tensorboard_log=log_path, 
                policy_kwargs={'net_arch':net_arch})

a2c_model.learn(total_timesteps=a2c_model_timesteps)

''' * * *  tria save model folder * * * '''
tria_a2c_model_path = os.path.join('train','save', a2c_model_name)
a2c_model.save(tria_a2c_model_path)

''' delete model in memory to validate model persistance '''
del a2c_model

''' load model in memory to validate model persistance '''
a2c_model = A2C.load(tria_a2c_model_path, env=env)

evaluate_policy(a2c_model, env, n_eval_episodes=20, render=False)

env.close()

print('* * * Tria A2C model for tria 3D environment predictions * * *')

episodes=10
for episode in range(1, episodes+1):
    observation = env.reset()
    terminated = False
    score = 0
    while not terminated:
        #env.render()
        action, _ = a2c_model.predict(observation)
        observation, reward, terminated , info = env.step(action)
        score += reward
    print('Model Name: {} Episone:{} Score:{}'.format( a2c_model_name, episode, score))

env.close() 

print('------------------------------------------------------------------')