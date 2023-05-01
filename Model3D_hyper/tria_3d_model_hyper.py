import os
import numpy as np
import random
import pickle as pkl
from typing import Any, Dict

import gym
from gym import Env
from gym.spaces import Discrete, Box, MultiDiscrete

import torch
import torch.nn as nn

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
import tria_rl

env_name = 'tria_rl/TriaClimate-v0'

n_trials = 125

n_jobs = 1

n_startup_trials = 10

n_evaluations = 5

n_timesteps = int(2e4)

eval_freq = int(n_timesteps / n_evaluations)

n_eval_envs = 2

n_eval_episodes = 50

timeout = int(60 * 15) 

delafult_hyperparams = {
    'policy' :  'MlpPolicy',
    'env' : env_name,
    'seed' : 1234,
    #'use_rms_prop' : True
}

def tria_a2c_params(trial: optuna.Trial) -> Dict[str, Any]:

    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)

    vf_coef = trial.suggest_float("vf_coef", 0.001, 1, log=True)
    
    gae_lambda = 1.0 - trial.suggest_float("gae_lambda", 0.001, 0.2, log=True)
    
    n_steps = 2 ** trial.suggest_int("exponent_n_steps", 3, 10, log=True)
    
    learning_rate = trial.suggest_float("lr", 1e-8, 0.1, log=True)
    
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)

    rms_prop_eps = trial.suggest_float('rms_prop_eps',1e-05, 1e-02, log=True )
    
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    
    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small", "mid", "large"])
    
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu", "leakyRelu"])

    #learn_rate_schedule = trial.suggest_categorical('learn_rate_schedule',['constant','linear','double_linear_con','middle_drop','double_middle_drop'])

    # 
    trial.set_user_attr("gamma_", gamma)
    
    trial.set_user_attr("gae_lambda_", gae_lambda)
    
    trial.set_user_attr("n_steps", n_steps)
    
    #neural network selection choice
    if (net_arch == "tiny"): 
        net_arch = {"pi": [400], "vf": [300]} 
    elif net_arch == "small":
        net_arch = {"pi": [64, 64], "vf": [64, 64]}
    elif net_arch == "mid":
        net_arch = {"pi":[64, 64, 64], "vf":[64, 64, 64]}
    else:
        net_arch = {"pi":[128,128,128,128], "vf":[128,128,128,128]}   
     
    # activation / non-linearity selection 
    activation_fn = {
        "tanh": nn.Tanh, 
        "relu": nn.ReLU, 
        "leakyRelu": nn.LeakyReLU
        } [activation_fn]
    
    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        'vf_coef' : vf_coef,
        'rms_prop_eps': rms_prop_eps,
        "max_grad_norm": max_grad_norm,
        "policy_kwargs": {
            "net_arch": net_arch,
            "activation_fn": activation_fn,
            "ortho_init": ortho_init,
        },
    }
    
class TriaTrialEvalCallback(EvalCallback):
    def __init__( self, eval_env: VecEnv, trial: optuna.Trial, n_eval_episodes: int = 50, eval_freq: int = 10000, deterministic: bool = True, verbose: int = 0 ): 
        super().__init__(eval_env= eval_env, n_eval_episodes= n_eval_episodes, eval_freq= eval_freq, deterministic= deterministic, verbose= verbose)
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False
        
    def _on_step(self) -> bool:
        
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True    

def objective(trial: optuna.Trial) -> float:

    kwargs = delafult_hyperparams.copy()
    # Tria hyperparameters
    kwargs.update(tria_a2c_params(trial))
    # Create the RL model
    model = A2C(**kwargs)
    # Create env used for evaluation
    
    import tria_rl

    env = gym.make('tria_rl/TriaClimate-v0') #TriaEnv()

    env = DummyVecEnv([lambda: env])

    eval_envs = VecNormalize(env, norm_obs=True, norm_reward= True)

    #eval_envs = VecFrameStack(eval_envs, n_stack=4)
    
    #eval_envs = make_vec_env(env_name, n_eval_envs)
    # Create the callback that will periodically evaluate
    # and report the performance
    eval_callback = TriaTrialEvalCallback(
        eval_envs,
        trial,
        n_eval_episodes= n_eval_episodes, #N_EVAL_EPISODES,
        eval_freq= eval_freq, #EVAL_FREQ,
        deterministic=True
    )
    nan_encountered = False
    
    try:
        model.learn(n_timesteps, callback=eval_callback)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN
        print(e)
        nan_encountered = True
    finally:
        # Free memory
        model.env.close()
        eval_envs.close()

    # Tell the optimizer that the trial failed
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward    

print('startup trials: {} evaluations: {} trials: {}'.format(n_startup_trials, n_evaluations, n_trials))

torch.set_num_threads(1)

sampler = TPESampler(n_startup_trials=n_startup_trials)
    # Do not prune before 1/3 of the max budget is used

pruner = MedianPruner(
        n_startup_trials=n_startup_trials, n_warmup_steps=n_evaluations // 3
    )

study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")

try:
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, timeout=timeout)
except KeyboardInterrupt:
    pass

print("Number of finished trials: ", len(study.trials))

print("Best trial:")
trial = study.best_trial

print(f"  Value: {trial.value}")

print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

print("  User attrs:")
for key, value in trial.user_attrs.items():
    print(f"    {key}: {value}")

# Write report
study.trials_dataframe().to_csv("study_results_a2c_tria_3d.csv")

with open("study.pkl", "wb+") as f:
    pkl.dump(study, f)

fig1 = plot_optimization_history(study)
fig2 = plot_param_importances(study)

fig1.show()
fig2.show()
