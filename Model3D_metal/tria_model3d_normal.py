import os
import numpy as np

import gym
from gym import Env

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

import tria_rl
#env = gym.make('tria_rl/TriaClimate-v0')
env_id = 'tria_rl/TriaClimate-v0'
env = gym.make(env_id)

print('---- observation space ----')
print('observation space size:   ',env.observation_space.shape[0])
print('observation space sample: ',env.observation_space.sample)

print('---- action space ----')
print('action space: ', env.action_space)
print('action space sample: ', env.action_space.sample())

env = make_vec_env(env_id, n_envs=4)

env = VecNormalize(env, norm_obs=True, norm_reward=False)

model = A2C(policy = "MlpPolicy",
            env = env,
            gae_lambda = 0.9,
            gamma = 0.99,
            learning_rate = 0.00096,
            max_grad_norm = 0.5,
            n_steps = 8,
            vf_coef = 0.4,
            ent_coef = 0.0,
            policy_kwargs=dict(
            log_std_init=-2, ortho_init=False),
            normalize_advantage=False,
            use_rms_prop= True,
            #use_sde= True,
            verbose=1)

model.learn(total_timesteps=200000)

tria_a2c_model_path = os.path.join('train','save', "tria_a2c_normalized")

model.save(tria_a2c_model_path)

del model

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
    print('Model Name: {} Episone:{} Score:{}'.format( "tria_a2c_normalized", episode, score))

env.close() 

print('------------------------------------------------------------------')