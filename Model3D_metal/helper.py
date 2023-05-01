import matplotlib.pyplot as plt
from IPython import display
import numpy as np

#plt.ion()

def plot(scores, mean_scores):
    #display.clear_output(wait=True)
    #display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) -1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) -1, mean_scores[-1], str(mean_scores[-1]))
    plt.show()

def plots(scores, mean_scores):

    print('Data:', scores[0], scores[1])
    freq = np.arange(0, len(scores[0]), dtype=int)
    fig, ax = plt.subplots(nrows=2,ncols=2, figsize=(8,8), layout='constrained')
    fig.tight_layout()

    ax[0,0].set_title('no model')
    ax[0,0].set_ylabel('score')
    #ax[0,0].set_xlabel('number episode')
    ax[0,0].plot( freq, scores[0])
    ax[0,0].plot(mean_scores[0])
    #ax[0,0].text(len(scores) -1, scores[-1], str(scores[-1]))
    #ax[0,0].text(len(mean_scores) -1, mean_scores[-1], str(mean_scores[-1]))

    ax[0,1].set_title('ppo model')
    ax[0,1].set_ylabel('score')
    #ax[0,1].set_xlabel('number episode')
    ax[0,1].plot(freq,scores[1])
    ax[0,1].plot(mean_scores[1])

    ax[1,0].set_title('neural model')
    ax[1,0].set_ylabel('score')
    #ax[1,0].set_xlabel('number episode')
    ax[1,0].plot(freq,scores[2])
    ax[1,0].plot(mean_scores[2])

    ax[1,1].set_title('a2c model')
    ax[1,1].set_ylabel('score')
    #x[1,1].set_xlabel('number episode')
    ax[1,1].plot(freq,scores[3])
    ax[1,1].plot(mean_scores[3])

    plt.show()