import os
import gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import tria_rl
import logging as log

class tria_inference_engine():
    def __init__(self, name, desc, envPath=None, envName=None, model_name=None, root_path='../', log_path='logs', model_path='model'):
        self.name =  name
        self.desc = desc
        self.env_id = os.path.join(envPath, envName)
        self.model_name = model_name
        self.model_path = os.path.join(root_path, model_path, model_name)
        self.log_path = os.path.join(root_path, log_path) 
        self.model = None

    def loadEnvironment(self):
        env_raw = gym.make(self.env_id)
        return env_raw

    def loadNormalizedEnv(self, env):    
        env_vec = DummyVecEnv([lambda: env])
        env_norm = VecNormalize(env_vec, training=True, norm_obs=True, norm_reward=True, epsilon=1e-08, gamma=0.99)
        return env_norm
    
    def showEnvionmentProperties(self, env):
        print('* * * Tria Enviromnet for tria device controll predictions * * *')
        print('>> observation space size:   ',env.observation_space.shape[0])
        print('>> observation space sample: ',env.observation_space.sample)
        print('>> action space:             ',env.action_space)
        print('>> action space sample:      ',env.action_space.sample())    

    def loadModel(self, env):
     self.model = A2C.load(self.model_path, env=env)

    def getActionPrediction(self, env, obs):
       retDict = {}
       obs = np.array([obs])
       print('>>orig obs:', obs)
       observation_api = env.normalize_obs(obs)  
       print('>>norm obs:', observation_api)
       action_o = self.model.predict(observation_api, deterministic=True)[0]
       retDict['action'] = action_o.item()
       print('>>action  :',retDict)
       return retDict


if __name__ == '__main__':
   log.basicConfig(filename='../logs/tria_inference.log', encoding='utf-8', level=log.INFO)
   tie = tria_inference_engine('Tria Inference Engine',
                                'TIE serves prediction requests',
                                'tria_rl',
                                'TriaClimate-v0',
                                'tria_a2c_normalized'
                                )
   env_r = tie.loadEnvironment()

   tie.showEnvionmentProperties(env_r)

   env_n = tie.loadNormalizedEnv(env_r)

   tie.loadModel(env_n)

   print('action: ', tie.getActionPrediction(env_n, np.array([[59.78,57.89,20.976]], dtype=np.float32)) )