import os
import numpy as np

import gym
from gym import Env

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

class HParamCallback(BaseCallback):
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
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True

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

#env = make_vec_env(env_id, n_envs=4)

#env = VecNormalize(env, norm_obs=True, norm_reward=False, training=True)

log_path = os.path.join('train', 'log')

model = A2C(policy = "MlpPolicy",
            env = env_s,
            gae_lambda = 1.0, #0.117120962797502,
            gamma = 0.99,#0.80, #0.0016248762308103,
            learning_rate = 0.00007,#1.7072936513375555e-01,
            max_grad_norm = 0.5,
            n_steps = 8,
            vf_coef =0.5, # 0.00200901228628941,
            ent_coef = 0.0,
            policy_kwargs=dict(
            log_std_init=-2, 
            ortho_init=False),
            normalize_advantage=False,
            rms_prop_eps=1e-07, 
            use_rms_prop= True,
            #use_sde= True,
            verbose=1,
            tensorboard_log=log_path)

model.learn(total_timesteps=20000000, callback=HParamCallback())

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