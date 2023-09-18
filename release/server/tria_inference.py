import os
import gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import tria_rl
import logging as log

class tria_inference_engine():
    def __init__(self, 
                 name,
                 desc, 
                 envPath='tria_rl', 
                 envName='TriaClimate-v0', 
                 modelFileName='tria_a2c_normalized',
                 envFileName="tria_env_normalized",
                 rootPath='../', 
                 logPath='logs', 
                 envAndModelpath='env_model'):
        self.name =  name
        self.desc = desc
        self.env_id = os.path.join(envPath, envName)
        self.model_path = os.path.join(rootPath, envAndModelpath, modelFileName)
        self.norm_env_path = os.path.join(rootPath, envAndModelpath, envFileName)
        self.log_path = os.path.join(rootPath, logPath) 
        self.model = None
        self.norm_env = None

    def loadEnvironment(self):
        env_raw = gym.make(self.env_id)
        return env_raw

    def loadNormalizedEnv(self, env):    
        env_vec = DummyVecEnv([lambda: env])
        env_norm = VecNormalize(env_vec, training=True, norm_obs=True, norm_reward=True, epsilon=1e-08, gamma=0.99)
        return env_norm
    
    def showEnvionmentProperties(self, env):
        print('----------------------------------------------------------------')
        print('-- Tria Enviromnet for tria device controll predictions --------')
        print('-- observation space size:   ',env.observation_space.shape[0])
        print('-- action space:             ',env.action_space)
        print('----------------------------------------------------------------')

    def loadEnvAndModel(self, env):
        norm_env = VecNormalize.load(self.norm_env_path, env)
        self.model = A2C.load(self.model_path, env=norm_env)
        return norm_env

    def getActionPrediction(self, env, obs):
       retDict = {}
       obs = np.array([obs])
       obs_norm = env.normalize_obs(obs)  
       action_out = self.model.predict(obs_norm, deterministic=True)[0]
       retDict['action'] = action_out.item()
       print('-- obs rec: {} obs norm: {} action:{}'.format(obs, obs_norm, retDict['action']))
       return retDict


if __name__ == '__main__':
   log.basicConfig(filename='../logs/tria_inference.log', encoding='utf-8', level=log.INFO)
   tie = tria_inference_engine(name='Tria Inferance Engine',
                               desc='TIE serves prediction requests',
                                envPath='tria_rl',
                                envName='TriaClimate-v0',
                                envFileName='tria_a2c_normalized',
                                modelFileName='tria_env_normalized'
                                )
   env_r = tie.loadEnvironment()

   tie.showEnvionmentProperties(env_r)

   env_n = tie.loadNormalizedEnv(env_r)

   env_with_stats = tie.loadEnvAndModel(env_n)

   print('action: ', tie.getActionPrediction(env_with_stats, np.array([[59.78,57.89,20.976]], dtype=np.float32)) )