import os
import sys
import numpy as np
import gym
from gym import Env
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

import tria_rl
#env = gym.make('tria_rl/TriaClimate-v0')
env_id = 'tria_rl/TriaClimate-v0'

# Parallel environments
#vec_env = make_vec_env(env_id, n_envs=4)
env = gym.make(env_id)

tria_a2c_model_path = os.path.join('test','model', 'tria_a2c_normalized')
print('loading tranined model at: ', tria_a2c_model_path)

""" model = A2C("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=25000)
model.save("a2c_tria")

del model # remove to demonstrate saving and loading 
"""

model = A2C.load(tria_a2c_model_path)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    #=vec_env.render("human")
    print('reward: {} obs: {}'.format( rewards, obs))