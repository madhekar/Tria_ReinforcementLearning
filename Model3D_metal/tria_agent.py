from collections import defaultdict

import matplotlib.pylot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import patches
from tdqm import tdqm

import gym

import tria_rl
env = gym.make('tria_rl/TriaClimate-v0')


class TriaAgent:
    def __init__(
            self,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            target_epsilon: float,
            descount_factor: float = 0.97):
        '''
        initialize agent with empty dictionary of state values (q_values),
        learning rate and epsilon
        '''
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = descount_factor
        self.eps = initial_epsilon
        self.eps_decay = epsilon_decay
        self.target_eps = target_epsilon

        self.train_err = []

    def get_action(self, obs: tuple[int, int, int]) -> int:
        '''
          returns best action with probability (1 - eps) 
          ow random action with prob epsilon to ensure exploration.
        '''    
        if np.random.random < self.eps:
            return env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))
        
    def update(
            self,
            obs: tuple[int,int,int],
            action: tuple[]
    )    