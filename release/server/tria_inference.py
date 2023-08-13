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
        self.env = None
        self.model = None

    def loadEnvironment(self):
        self.env = gym.make(self.env_id)

    def loadNormalizedEnv(self):    
        self.env = DummyVecEnv([lambda: self.env])
        self.env = VecNormalize(self.env, training=True, norm_obs=True, norm_reward=True, epsilon=1e-08, gamma=0.99)

    def showEnvionmentProperties(self):
        print('* * * Tria Enviromnet for tria device controll predictions * * *')
        print('>> observation space size:   ',self.env.observation_space.shape[0])
        print('>> observation space sample: ',self.env.observation_space.sample)
        print('>> action space:             ', self.env.action_space)
        print('>> action space sample:      ', self.env.action_space.sample())    

    def loadModel(self):
     self.model = A2C.load(self.model_path, env=self.env)
     self.env.close()

    def getActionPrediction(self, obs):
       #obs= np.array(obs, dtype=np.float32)
       print('original observation: ', obs)
       #observation_api = self.env.normalize_obs(obs)  
       #print('normalized observation', obs)
       action_o = self.model.predict(obs, deterministic=True)[0]
       print(action_o)
       return action_o


if __name__ == '__main__':
   log.basicConfig(filename='../logs/tria_inference.log', encoding='utf-8', level=log.INFO)
   tie = tria_inference_engine('Tria Inference Engine',
                                'TIE serves prediction requests',
                                'tria_rl',
                                'TriaClimate-v0',
                                'tria_a2c_normalized'
                                )
   tie.loadEnvironment()

   tie.showEnvionmentProperties()

   tie.loadNormalizedEnv()

   tie.loadModel()

   

   print('action: ', tie.getActionPrediction(np.array([59.78,57.89,20.976], dtype=np.float32))[0] )