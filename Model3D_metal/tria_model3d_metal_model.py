#!/usr/bin/env python3

import os
import numpy as np
import random

import gym
from gym import Env
from gym.spaces import Discrete, Box, MultiDiscrete

#import tensorflow as tf

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
#load_ext tf.tensorboard

from helper import plots

env_name = 'tria-3d-rl-model-'

ppo_model_timesteps= 2000000; neural_model_timesteps=2000000; a2c_model_timesteps=2000000
predict_episodes=10000
ppo_model_name = env_name + 'ppo'; neural_model_name = env_name + 'ppo-neural'; a2c_model_name = env_name + 'a2c'


''' * * * gym tria environment instance * * * '''

import tria_rl
env = gym.make('tria_rl/TriaClimate-v0') # TriaEnv()
#env = DummyVecEnv([lambda: env])
#env = VecNormalize(env, norm_obs=True, norm_reward= False)
plot_scores= [[0] * predict_episodes for i in range(4)]
plot_mean_scores=[[0] * predict_episodes for i in range(4)]

print(env.metadata)
print('------------------------------------------------------------------')
print("1. Sample observation space: {}".format(env.observation_space.sample()))
print("1. Sample observation space: {}".format(env.observation_space))
print("1. type observation space  : {}".format(env.observation_space.dtype))
print("2. Sample action space     : {}".format(env.action_space.sample()))
#print("3. Sample state            : {}".format(env.state))    
print('------------------------------------------------------------------')
''' * * * gym tria environment instance validation before model * * * '''


for episode in range(0, predict_episodes):
    state = env.reset()
    #print(state)
    terminated = False
    score = 0 #[0,0,0] 
    game=0
    total_score = 0
    while not terminated:
        action = env.action_space.sample()
        next_state, rew, terminated, info = env.step(action) 
        score += rew
        game +=1
    #print('Episode: {} Score: {}'.format(episode, score))
    plot_scores[0][episode] = score
    total_score += score
    #print(total_score, game)
    mean_score = total_score / game
    plot_mean_scores[0][episode] = mean_score 
env.close()
print('------------------------------------------------------------------')

''' * * *  tria logging folders * * * '''
log_path = os.path.join('train','log')

print(log_path)

'''  *  Tria Custom PPO Network  * '''
print('* * * Tria PPO model for tria 3D environment * * *')

env = DummyVecEnv([lambda: env])

env = VecNormalize(env, norm_obs=True, norm_reward= False)

#env = VecFrameStack(env, n_stack=4)

ppo_model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

ppo_model.learn(total_timesteps=ppo_model_timesteps)

tria_ppo_model_path = os.path.join('train','save', ppo_model_name)

ppo_model.save(tria_ppo_model_path)

del ppo_model

ppo_model = PPO.load(tria_ppo_model_path, env=env)

evaluate_policy(ppo_model, env, n_eval_episodes=20, render=False)

env.close()

print('* * * Tria PPO model for tria 3D environment predictions * * *')


for episode in range(0, predict_episodes):
    observation = env.reset()
    terminated = False
    score = 0
    game=0
    total_score = 0
    while not terminated:
        action, _ = ppo_model.predict(observation)
        observation, rew, terminated , info = env.step(action)
        #print(rew)
        score += sum(rew)
        game +=1
    #print('Model Name: {} Episone:{} Score:{}'.format( ppo_model_name, episode, score))
    plot_scores[1][episode] = score
    total_score += score
    mean_score = total_score / game
    #print(total_score, game)
    plot_mean_scores[1][episode] = mean_score
env.close()  

print('------------------------------------------------------------------')

'''  *  Tria Custom Neural Network  * '''
print('* * * Tria neural network model for tria 3D environment * * *')

#net_arch = dict(pi=[128,128,128,128], vf=[128,128,128,128])

net_arch = dict(pi=[400,400,400,400], vf=[300,300,300,300])

nn_model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path, policy_kwargs={'net_arch':net_arch})

nn_model.learn(total_timesteps=neural_model_timesteps)

tria_neural_model_path = os.path.join('train','save', neural_model_name)

nn_model.save(tria_neural_model_path)

del nn_model

nn_model = PPO.load(tria_neural_model_path, env=env)

evaluate_policy(nn_model, env, n_eval_episodes=20, render=False)

env.close()


print('* * * Tria neural network model for tria 3D environment predictions * * *')
total_score = 0
for episode in range(0, predict_episodes):
    observation = env.reset()
    terminated = False
    score = 0
    game=0
    total_score = 0
    while not terminated:
        action, _ = nn_model.predict(observation)
        observation, rew, terminated , info = env.step(action)
        #print(rew)
        score += sum(rew)
        game +=1
    #print('Model Name: {} Episone:{} Score:{}'.format( neural_model_name, episode, score))
    plot_scores[2][episode] = score
    total_score += score
    mean_score = total_score / game
    plot_mean_scores[2][episode] = mean_score
env.close()  

print('------------------------------------------------------------------')

'''  *  Tria A2C  Network  * '''
print('* * * Tria A2C network model for tria 3D environment * * *')

from stable_baselines3 import A2C

a2c_model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_path)

a2c_model.learn(total_timesteps=a2c_model_timesteps)

tria_a2c_model_path = os.path.join('train','save', a2c_model_name)

a2c_model.save(tria_a2c_model_path)

del a2c_model

a2c_model = A2C.load(tria_a2c_model_path, env=env)

evaluate_policy(a2c_model, env, n_eval_episodes=20, render=False)

env.close()

print('* * * Tria A2C model for tria 3D environment predictions * * *')


total_score = 0
for episode in range(0, predict_episodes):
    observation = env.reset()
    terminated = False
    score = 0
    game=0    
    total_score = 0
    while not terminated:
        #env.render()
        action, _ = a2c_model.predict(observation)
        observation, rew, terminated , info = env.step(action)
        score += sum(rew)
        game +=1
    #print('Model Name: {} Episone:{} Score:{}'.format( a2c_model_name, episode, score))
    
    plot_scores[3][episode] = score
    total_score += score
    mean_score = total_score / game
    plot_mean_scores[3][episode] = mean_score  
env.close() 

print('------------------------------------------------------------------')

plots(plot_scores, plot_mean_scores)

#tensorboard --logdir './train/log/' --bind_all  # training_log_path
