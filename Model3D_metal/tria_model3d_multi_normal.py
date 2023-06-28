import os
import numpy as np

import gym
from gym import Env
import torch as th

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.monitor import Monitor

import matplotlib.pyplot as plt

class HyperParameterCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._plot = None

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
        # get the monitor's data
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if self._plot is None: # make the plot
            plt.ion()
            fig = plt.figure(figsize=(6,3))
            ax = fig.add_subplot(111)
            line, = ax.plot(x, y)
            self._plot = (line, ax, fig)
            plt.show()
        else: # update and rescale the plot
            self._plot[0].set_data(x, y)
            self._plot[-2].relim()
            self._plot[-2].set_xlim([self.locals["total_timesteps"] * -0.02, 
                                    self.locals["total_timesteps"] * 1.02])
            self._plot[-2].autoscale_view(True,True,True)
            self._plot[-1].canvas.draw()
        return True
    
log_dir = os.path.join('train', 'log')

import tria_rl
#env = gym.make('tria_rl/TriaClimate-v0')
env_id = 'tria_rl/TriaClimate-v0'
env_s = gym.make(env_id)
env_s = Monitor(env_s, log_dir)
#env_s = make_vec_env(env_id, n_envs=1)
print('---- observation space attributes ----')
print('observation space size:   ',env_s.observation_space.shape[0])
print('observation space sample: ',env_s.observation_space.sample)

print('---- action space attributes ----')
print('action space: ', env_s.action_space)
print('action space sample: ', env_s.action_space.sample())

env_s = DummyVecEnv([lambda: env_s])

env_s = VecNormalize(env_s, norm_obs=True, norm_reward=True)

#log_dir = os.path.join('train', 'log')

model = A2C(policy = "MlpPolicy",
            env = env_s,
            gae_lambda = 0.042,#0.8979709455838538,#1.0, #0.117120962797502,
            gamma =  0.995,#0.9657236425464014,#0.99,#0.80, #0.0016248762308103,
            learning_rate = 0.051,#1.0767603107498563e-08,#0.0007,#1.7072936513375555e-01,
            max_grad_norm = 0.88,#4.565654908777005,#0.5,
            n_steps = 256,#8,
            vf_coef = 0.11,#0.0024435757218033904,#0.5, # 0.00200901228628941,
            ent_coef = 1.0976520036433521e-08,#0.04553259441269758,#0.0,
            policy_kwargs=dict(
            activation_fn=th.nn.Tanh, 
            log_std_init=-2, 
            ortho_init=False),
            normalize_advantage=False,
            rms_prop_eps=1e-07, 
            use_rms_prop= True,
            #use_sde= True,
            verbose=1,
            tensorboard_log=log_dir)

model.learn(total_timesteps=20000000, callback=HyperParameterCallback())

tria_a2c_model_path = os.path.join('train','save', "tria_a2c_normalized")

model.save(tria_a2c_model_path)

del model

model = A2C.load(tria_a2c_model_path, env=env_s)

evaluate_policy(model, env_s, n_eval_episodes=20, render=False)

env_s.close()

print('* * * Tria A2C model for tria 3D environment predictions * * *')

episodes=1000
for episode in range(1, episodes+1):
    observation = env_s.reset()
    terminated = False
    score = 0
    while not terminated:
        #env.render()
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated , info = env_s.step(action)
        score += reward
        print('observation: {} action: {} reward: {}'.format(observation, action, reward));
    print('Model Name: {} Episone:{} Score:{}'.format( "tria_a2c_normalized", episode, score))

env_s.close() 

print('------------------------------------------------------------------')
