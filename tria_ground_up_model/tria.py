import gym
import numpy as np
from dueling_ddqn_torch import Agent
from tools import plotLearning
import tria_rl

if __name__ == '__main__':
    #env = gym.make('LunarLander-v2')
    
    env = gym.make('tria_rl/TriaClimate-v0') #TriaEnv()
    num_games = 100000
    load_checkpoint = False

    agent = Agent(gamma=0.99, epsilon=1.0, lr=5e-4,
                  input_dims=[3], n_actions=14, mem_size=100000, eps_min=0.01,
                  batch_size=64, eps_dec=1e-3, replace=100)

    if load_checkpoint:
        agent.load_models()

    filename = 'Tria-Dueling-DDQN-512-Adam-lr0005-replace100.png'
    scores = []
    eps_history = []
    n_steps = 0

    for i in range(num_games):
        done = False
        observation = env.reset()
        #print('obs: ', observation)
        score = 0

        while not done:
            action = agent.choose_action(observation)
            print('action: ', action)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, int(done))
            agent.learn()

            observation = observation_

        scores.append(score)
        avg_score = np.mean(scores[max(0, i-100):(i+1)])
        print('episode: ', i,'score %.1f ' % score,
             ' average score %.1f' % avg_score,
            'epsilon %.2f' % agent.epsilon)
        if i > 0 and i % 10 == 0:
            agent.save_models()

        eps_history.append(agent.epsilon)
    #print(scores, eps_history)
    x = [i+1 for i in range(num_games)]
    plotLearning(x, scores, eps_history, filename)
