#!/usr/bin/env python3

import os
import numpy as np
import random

import gym
from gym import Env
from gym.spaces import Discrete, Box, MultiDiscrete

#import tensorflow as tf

from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
#load_ext tf.tensorboard

env_name = 'tria-3d-rl-model-'

t_ini= 70.0; h_ini= 40.0; a_ini= 10.0 

t_min =-40.0; t_max=110; h_min=0.0; h_max=100.0; a_min=0.0; a_max=5000.0

act_state = 2

stat_rand_min = -1.0; stat_rand_max = 1.0

equilibrium_cycles= 60

r1 = -0.25; r2 = -0.5; r3 = 2; nr3 = -2

const_weight_vec  = [1, 1, 1, 1]

d3 = {
     0 : [65.0, 80.0, 50.0, 85.0, 40.0, 90.0], 
     1 : [30.0, 50.0, 20.0, 60.0, 10.0, 70.0], 
     2 : [0.0, 19.0, 200.0, 599.0, 600.0, 2000.0]
    }

d1 = {0: [65.0, 80.0], 1: [30.0, 50.0], 2: [0.0, 20.0]}

ppo_model_timesteps= 200000; neural_model_timesteps=200000; a2c_model_timesteps=200000

ppo_model_name = env_name + 'ppo'; neural_model_name = env_name + 'ppo-neural'; a2c_model_name = env_name + 'a2c'

''' * * * gym tria environment class difination * * * '''
class TriaEnv(Env):
    
    def __init__(self):
        self.action_space = MultiDiscrete(np.array([act_state, act_state, act_state, act_state, act_state]))
        

        low = np.array([t_min, h_min, a_min]).astype(np.float32)
        high = np.array([t_max, h_max, a_max]).astype(np.float32)

        self.observation_space = Box(low, high, shape=(3,))
        
        self.state = [t_ini + random.uniform(stat_rand_min, stat_rand_max), h_ini + random.uniform(stat_rand_min, stat_rand_max), a_ini + random.uniform(stat_rand_min, stat_rand_max)]
        
        #print('^^^', self.state, self.action_space)
        
        self.equilibrium_cycles_len = equilibrium_cycles
        
    def step(self, action):
        
        ap_scaled = [1 if e == 1 else -1 for e in action]  # 0 (off) => -1 and 1 (on) => 1
        
        actionPrime = [a * b for a, b in zip(ap_scaled, const_weight_vec)] 
        
        actionAlgo = [actionPrime[a] - actionPrime[len(actionPrime) -a -1] for a in range(len(actionPrime) // 2)]
        
        actionAlgo.append(actionPrime[len(actionPrime) // 2])                                                              
        
        #print('***',actionAlgo, self.state)
        
        self.state = [a + b for a, b in zip(actionAlgo, self.state)]
        
        #print('&&&', actionAlgo, self.state)
        
        #reduce tria simulation length by 1 second
        self.equilibrium_cycles_len -= 1
        
        reward = [r3 if e >= d3[i][0] and e<= d3[i][1] else r2 if e >= d3[i][2] and e<= d3[i][3] else r1 if e >= d3[i][4] and e <= d3[i][5] else nr3 for i, e in enumerate(self.state)]
        #reward = [r3 if e >= d1[i][0] and e <= d1[i][1] else nr3  for i, e in enumerate(self.state)]

        reward = sum(reward)
        #print('$$$', reward)
            
        if self.equilibrium_cycles_len <= 0:
            terminated = True
        else:
            terminated = False
            
        info = {}
        #print('reward:{} state:{}'.format(reward, self.state))
        return self.state, reward, terminated,  info
    
    def render(self):
        pass
    
    def reset(self):
        
        self.state =[t_ini + random.uniform(stat_rand_min, stat_rand_max), h_ini + random.uniform(stat_rand_min, stat_rand_max), a_ini + random.uniform(stat_rand_min, stat_rand_max)]
        #print('@@@', self.state)
        self.equilibrium_cycles_len = equilibrium_cycles
        
        return self.state

''' * * * gym tria environment instance * * * '''
env= TriaEnv()

print("1. Sample observation space: {}".format(env.observation_space.sample()))
print("1. Sample observation space: {}".format(env.observation_space))
print("1. Sample observation space: {}".format(env.observation_space.dtype))
print("2. Sample action space     : {}".format(env.action_space.sample()))
print("3. Sample state            : {}".format(env.state))    

''' * * * gym tria environment instance validation before model * * * '''
episodes = 5
for episode in range(1, episodes+1):
    state = env.reset()
    #print(state)
    terminated = False
    score = 0 #[0,0,0] 
    
    while not terminated:
        #env.render()
        action = env.action_space.sample()
        #print(action, terminated , reward)
        #print(env.step(action))
        next_state, reward, terminated,  info = env.step(action) 
        score += reward #[a + b for a, b in zip(reward, score)]
    print('Episode: {} Score: {}'.format(episode, score))
env.close()

''' * * *  tria logging folders * * * '''
log_path = os.path.join('train','log')
print(log_path)

'''  *  Tria Custom PPO Network  * '''
print('* * * Tria PPO model for tria 3D environment * * *')

env = DummyVecEnv([lambda: env])
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

episodes=10
for episode in range(1, episodes+1):
    observation = env.reset()
    terminated = False
    score = 0
    while not terminated:
        #env.render()
        action, _ = ppo_model.predict(observation)
        observation, reward, terminated , info = env.step(action)
        score += reward
    print('Model Name: {} Episone:{} Score:{}'.format( ppo_model_name, episode, score))
env.close()  

'''  *  Tria Custom Neural Network  * '''
print('* * * Tria neural network model for tria 3D environment * * *')

net_arch = dict(pi=[128,128,128,128], vf=[128,128,128,128])

nn_model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path, policy_kwargs={'net_arch':net_arch})

nn_model.learn(total_timesteps=neural_model_timesteps)

tria_neural_model_path = os.path.join('train','save', neural_model_name)

nn_model.save(tria_neural_model_path)

del nn_model

nn_model = PPO.load(tria_neural_model_path, env=env)

evaluate_policy(nn_model, env, n_eval_episodes=20, render=False)

env.close()

print('* * * Tria neural network model for tria 3D environment predictions * * *')

episodes=10
for episode in range(1, episodes+1):
    observation = env.reset()
    terminated = False
    score = 0
    while not terminated:
        #env.render()
        action, _ = nn_model.predict(observation)
        observation, reward, terminated , info = env.step(action)
        score += reward
    print('Model Name: {} Episone:{} Score:{}'.format( neural_model_name, episode, score))

env.close()  


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

#tensorboard --logdir './train/log/' --bind_all  # training_log_path