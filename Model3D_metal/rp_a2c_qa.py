import os
import gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import tria_rl
env_id = 'tria_rl/TriaClimate-v0'
env_s = gym.make(env_id)

print('---- observation space attributes ----')
print('observation space size:   ',env_s.observation_space.shape[0])
print('observation space sample: ',env_s.observation_space.sample)

print('---- action space attributes ----')
print('action space: ', env_s.action_space)
print('action space sample: ', env_s.action_space.sample())

env_s = DummyVecEnv([lambda: env_s])

env_s = VecNormalize(env_s, training=True, norm_obs=True, norm_reward=True, epsilon=1e-08, gamma=0.99)

log_path = os.path.join('train', 'log')

tria_a2c_model_path = os.path.join('train','save', "tria_a2c_normalized")

model = A2C.load(tria_a2c_model_path, env=env_s)

env_s.close()

print('* * * Tria A2C model for tria 3D environment predictions * * *')

episodes=4
for episode in range(0, episodes):
    observation = env_s.reset()
    terminated = False
    itruncated = False 
    score = 0
    norm_score=0
    while not terminated:        
        ob = env_s.get_original_obs().ravel().tolist()

        action, _ = model.predict(observation, deterministic=True)

        observation, norm_reward, terminated , info = env_s.step(action)

        norm_score += norm_reward
        score +=env_s.get_original_reward()

        r = env_s.get_original_reward().tolist()[0]

        print('E: {} O: {} A: {} R: {}'.format( episode, [str(round(o,2)) for o in ob], action.tolist()[0], r))
    print('Model Name: {} Episone:{} Score:{}'.format( "tria_a2c_normalized", episode, score))

env_s.close() 

print('------------------------------------------------------------------')
