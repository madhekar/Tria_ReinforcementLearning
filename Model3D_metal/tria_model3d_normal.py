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
env_s = gym.make(env_id)

print('---- observation space ----')
print('observation space size:   ',env_s.observation_space.shape[0])
print('observation space sample: ',env_s.observation_space.sample)

print('---- action space ----')
print('action space: ', env_s.action_space)
print('action space sample: ', env_s.action_space.sample())

env = make_vec_env(env_id, n_envs=1)

env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=50, training=True, gamma=.99 )

model = A2C(policy = "MlpPolicy",
            env = env,
            gae_lambda = 0.117120962797502,
            gamma =0.0016248762308103,
            learning_rate = 1.7072936513375555e-04,
            max_grad_norm = 0.5,
            n_steps = 8,
            vf_coef = 0.00200901228628941,
            ent_coef = 0.0,
            policy_kwargs=dict(
            log_std_init=-2, ortho_init=False),
            normalize_advantage=False,
            use_rms_prop= True,
            #use_sde= True,
            verbose=1)

model.learn(total_timesteps=2000000)

tria_a2c_model_path = os.path.join('train','save', "tria_a2c_normalized")

model.save(tria_a2c_model_path)

del model

model = A2C.load(tria_a2c_model_path, env=env_s)

evaluate_policy(model, env_s, n_eval_episodes=20, render=False)

env_s.close()

print('* * * Tria A2C model for tria 3D environment predictions * * *')

episodes=10
for episode in range(1, episodes+1):
    observation = env_s.reset()
    terminated = False
    score = 0
    while not terminated:
        #env.render()
        action, _ = model.predict(observation)
        observation, reward, terminated , info = env_s.step(action)
        score += reward
    print('Model Name: {} Episone:{} Score:{}'.format( "tria_a2c_normalized", episode, score))

env_s.close() 

print('------------------------------------------------------------------')