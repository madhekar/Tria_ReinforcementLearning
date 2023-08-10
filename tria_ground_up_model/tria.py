#import gym
import gymnasium as gym
import numpy as np
from dueling_ddqn_torch import Agent
from tools import plotLearning
import tria_rl
from realtime_helper import plot


if __name__ == '__main__':
    #env = gym.make('LunarLander-v2')
    
    env = gym.make('tria_rl/TriaClimate-v0') #TriaEnv()
    num_games = 1000
    load_checkpoint = False

    agent = Agent(gamma=0.99, epsilon=1.0, lr=5e-3,
                  input_dims=[3], n_actions=6, mem_size=10000000, eps_min=0.01,
                  batch_size=64, eps_dec=1e-3, replace=100)

    if load_checkpoint:
        agent.load_models()

    filename = 'Tria-Dueling-DDQN-512-Adam-lr0005-replace100.png'
    scores = []
    #plot_scores =[]
    mean_scores = []
    #actions = []
    eps_history = []
    n_steps = 0
    #game = 0
    score =0
    total_score = 0
    for i in range(num_games):
        done = False
        truncate = False
        observation,_ = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done,truncate, info = env.step(action)
            #print('>>', observation, observation_)
            score += reward
            #game += 1
            agent.store_transition(observation, action, reward, observation_, int(done))
            agent.learn()

            observation = observation_

            #total_score += reward
            #mean_score = total_score / game
            #actions.append(action)
            #plot_scores.append(reward)
            #plot_mean_scores.append(mean_score)
            #plot(plot_scores, plot_mean_scores, actions)

        scores.append(score)
        avg_score = np.mean(scores[max(0, i-100):(i+1)])
        print('episode: ', i,'score %.1f ' % score,
             ' average score %.1f' % avg_score,
            'epsilon %.2f' % agent.epsilon)
        if i > 0 and i % 10 == 0:
            agent.save_models()

        eps_history.append(agent.epsilon)
        mean_scores.append(avg_score)
        plot(scores, mean_scores, eps_history )
    #print(scores, eps_history)
    x = [i+1 for i in range(num_games)]
    plotLearning(x, scores, eps_history, filename)
