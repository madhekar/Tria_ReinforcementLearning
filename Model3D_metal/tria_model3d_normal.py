import os
import numpy as np

import gym
from gym import Env
import torch as th
import torch.nn as nn

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

from realtime_helper import plot

class HyperParameterCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
            "gae_lambda": self.model.gae_lambda,
            "ent_coef": self.model.ent_coef,
            "vf_coef": self.model.vf_coef,
            "max_grad_norm": self.model.max_grad_norm
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0.0,
        }
        self.logger.record(
            "hyper parameters",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True

import tria_rl
#env = gym.make('tria_rl/TriaClimate-v0')
env_id = 'tria_rl/TriaClimate-v0'
env_s = gym.make(env_id)


print('---- observation space attributes ----')
print('observation space size:   ',env_s.observation_space.shape[0])
print('observation space sample: ',env_s.observation_space.sample)

print('---- action space attributes ----')
print('action space: ', env_s.action_space)
print('action space sample: ', env_s.action_space.sample())

#env = make_vec_env(env_id, n_envs=4)

env_s = DummyVecEnv([lambda: env_s])

env_s = VecNormalize(env_s, training=True, norm_obs=True, norm_reward=True, epsilon=1e-08, gamma=0.99)

log_path = os.path.join('train', 'log')

model = A2C(policy = "MlpPolicy",
            env = env_s,
            gae_lambda = 0.95,#1.0,#0.99,#0.95,#0.89,#0.8979709455838538,#1.0, #0.117120962797502,
            gamma =  0.99,#0.995,#0.9657236425464014,#0.99,#0.80, #0.0016248762308103,
            learning_rate = 1.9e-05,#0.004988,#1.013e-05,#0.0007,#1.0767603107498563e-08,#0.0007,#1.7072936513375555e-01,
            max_grad_norm = 0.9,#2,#0.3,#0.88,#4.565654908777005,#0.5,
            n_steps = 8,#256,#1024,#16,#100,#8,
            vf_coef = 0.9,#0.93,#0.239,#0.38,#0.51,#0.0024435757218033904,#0.5, # 0.00200901228628941,
            ent_coef = 1.185e-08,#0.00294,#3.11e-06,#0.00037,#1.0976520036433521e-08,#0.04553259441269758,#0.0,
            policy_kwargs=dict(
            activation_fn = nn.Tanh,  
            net_arch = {'pi':[64, 64, 64], 'vf':[64, 64, 64]},  
            log_std_init=-2, 
            ortho_init=False
            ),
            normalize_advantage=False,
            rms_prop_eps=1e-07, 
            use_rms_prop= False,
            #use_sde= True,
            verbose=1,
            tensorboard_log=log_path)



model.learn(total_timesteps=90, callback=HyperParameterCallback())

tria_a2c_model_path = os.path.join('train','save', "tria_a2c_normalized")

model.save(tria_a2c_model_path)

del model

model = A2C.load(tria_a2c_model_path, env=env_s)

evaluate_policy(model, env_s, n_eval_episodes=20, render=False)

env_s.close()

print('* * * Tria A2C model for tria 3D environment predictions * * *')

episodes=100
plot_scores= []
plot_mean_scores=[]
actions=[]
score = 0
game=0
total_score=0
for episode in range(1, episodes):
    observation = env_s.reset()
    terminated = False
    truncated = False
    score = 0
    norm_score=0
    while not terminated:
        #env.render()
        action, _ = model.predict(observation, deterministic=True)
        observation, norm_reward, terminated , info = env_s.step(action)
        norm_score = norm_reward
        score =env_s.get_original_reward()
        game +=1

        plot_scores.append(score)
        actions.append(action)
        total_score += score
        mean_score = total_score / game
        plot_mean_scores.append(mean_score)
        plot(plot_scores, plot_mean_scores, actions)
        print('norm_obs: {} observation: {} action: {} norm reward: {} reward: {}'.format(observation, env_s.get_original_obs(), action, norm_reward, env_s.get_original_reward()));
    print('Model Name: {} Episone:{} Score:{}'.format( "tria_a2c_normalized", episode, score))

env_s.close() 

print('------------------------------------------------------------------')
