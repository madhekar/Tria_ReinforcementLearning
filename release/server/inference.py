import os
import gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import tria_rl

env_id = 'tria_rl/TriaClimate-v0'
env_s = gym.make(env_id)

print('* * * Tria A2C model for tria 3D environment predictions * * *')
print('---- observation and action space attributes ----')
print('observation space size:   ',env_s.observation_space.shape[0])
print('observation space sample: ',env_s.observation_space.sample)
print('action space:             ', env_s.action_space)
print('action space sample:      ', env_s.action_space.sample())

env_s = DummyVecEnv([lambda: env_s])

env_n = VecNormalize(env_s, training=True, norm_obs=True, norm_reward=True, epsilon=1e-08, gamma=0.99)

log_path = os.path.join('../', 'logs')

tria_a2c_model_path = os.path.join('../','model', "tria_a2c_normalized")

model = A2C.load(tria_a2c_model_path, env=env_n)

#
#observation_o = env_s.reset()
observation_ext = np.array( (4.,  5., 6.)) 
#ob =  env_s.get_original_obs().ravel().tolist()
observation_api = env_n.normalize_obs(observation_ext).ravel().tolist()
action_o = model.predict(observation_api, deterministic=True)
print('unnorm obs: {} apinorm obs: {} action: {}'.format(observation_ext, observation_api, action_o))
#
env_s.close() 

