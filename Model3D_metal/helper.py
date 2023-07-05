import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython import display
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
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

    freq = np.arange(0, len(scores[0]), dtype=int)
    fig, ax = plt.subplots(nrows=2,ncols=2, figsize=(10,10))
    fig.tight_layout()

    ax[0,0].set_title('no model')
    ax[0,0].set_ylabel('score')
    #ax[0,0].set_xlabel('number episode')
    ax[0,0].plot( freq, scores[0])
    ax[0,0].plot(freq,mean_scores[0])
    #ax[0,0].text(len(scores) -1, scores[-1], str(scores[-1]))
    #ax[0,0].text(len(mean_scores) -1, mean_scores[-1], str(mean_scores[-1]))

    ax[0,1].set_title('ppo model')
    ax[0,1].set_ylabel('score')
    #ax[0,1].set_xlabel('number episode')
    ax[0,1].plot(freq,scores[1])
    ax[0,1].plot(freq,mean_scores[1])

    ax[1,0].set_title('neural model')
    ax[1,0].set_ylabel('score')
    #ax[1,0].set_xlabel('number episode')
    ax[1,0].plot(freq,scores[2])
    ax[1,0].plot(freq,mean_scores[2])

    ax[1,1].set_title('a2c model')
    ax[1,1].set_ylabel('score')
    #x[1,1].set_xlabel('number episode')
    ax[1,1].plot(freq,scores[3])
    ax[1,1].plot(freq,mean_scores[3])

    plt.show()

def plots_norm(scores, mean_scores):

    freq = np.arange(0, len(scores[0]), dtype=int)
    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(10,5))
    fig.tight_layout()

    ax[0].set_title('A2C model normalized obs and rewards')
    ax[0].set_ylabel(' original score')
    ax[0].set_xlabel('episode number')
    ax[0].plot( freq, scores[0])
    ax[0].plot(freq, mean_scores[0])

    ax[1].set_title('A2C model normalized obs and rewards')
    ax[1].set_ylabel(' normalized score')
    ax[1].set_xlabel(' episode number')
    ax[1].plot(freq,scores[1])
    ax[1].plot(freq,mean_scores[1])

    plt.show()    

def plots_3d(obs,acv,color, _rows, _cols):
    rows, cols = _rows, _cols
    fig, ax = plt.subplots(nrows=rows,ncols=cols, figsize=(20,10), subplot_kw=dict(projection='3d'))
    fig.suptitle('Tria A2C model predictions', fontsize=10)
    n=0
    for row in range(rows):
        for col in range(cols):
            ax[row, col].scatter(obs[n][:,0],obs[n][:,1],obs[n][:,2], c = color[n] ) 
            ax[row, col].set_title('empisode: '+ str(n) +' initial obs: ' + str(round(obs[n][0:,0][0],2))+ ':' + str(round(obs[n][0:,1][0],2))+ ':' +str(round(obs[n][0:,2][0],2))  + ' \n final obs: ' + str(round(obs[n][0:,0][-1],2))+ ':' + str(round(obs[n][0:,1][-1],2)) + ':' + str(round(obs[n][0:,2][-1],2)), fontsize=6)
            ax[row, col].set_xlabel("T")
            ax[row, col].set_ylabel("H")
            ax[row, col].set_zlabel("AQ")
            for t,h,a,ac in zip(obs[n][:,0],obs[n][:,1],obs[n][:,2],acv[n]):
               ax[row, col].text(t,h,a,ac, fontsize=2)
            n+=1   

    plt.subplots_adjust(left=0.13,
                    bottom=0.044,
                    right=0.91,
                    top=0.915,
                    wspace=0.32,
                    hspace=0.158)           
    plt.show()

def plotPredictions(rws, obs, acv, color, _rows, _cols):
    rows, cols = _rows, _cols

    fig, ax = plt.subplots(nrows=rows,ncols=cols, figsize=(20,10))
    fig.suptitle('Tria A2C model predictions learning trend', fontsize=10)
    x = np.arange(0, len(rws[0]), dtype=int)
    n=0
    for row in range(rows):
        for col in range(cols):
            ax[row, col].scatter(x, rws[n],marker='.', c = color[n]) 
            ax[row, col].set_title('Reward Trend Episode: ' + str(n), fontsize=8)
            #ax[row, col].set_xlabel("step")
            ax[row, col].set_ylabel("reward")
            n+=1

    plt.subplots_adjust(left=0.13,
                    bottom=0.044,
                    right=0.91,
                    top=0.915,
                    wspace=0.32,
                    hspace=0.158)           
    plt.show()


def plotSimulation(obs, color, rws, acv):
    rows, cols = 2, 2
    fig, ax = plt.subplots(nrows=rows,ncols=cols, figsize=(20,10), subplot_kw=dict(projection='3d'))
    fig.suptitle('Tria A2C model predictions', fontsize=10)
    n=0
    for row in range(rows):
        for col in range(cols):
            ax[row, col].scatter(obs[n][:,0],obs[n][:,1],obs[n][:,2], c = color[n] ) 
            ax[row, col].set_title('empisode: '+ str(n) +' initial obs: ' + str(round(obs[n][0:,0][0],2))+ ':' + str(round(obs[n][0:,1][0],2))+ ':' +str(round(obs[n][0:,2][0],2))  + ' \n final obs: ' + str(round(obs[n][0:,0][-1],2))+ ':' + str(round(obs[n][0:,1][-1],2)) + ':' + str(round(obs[n][0:,2][-1],2)), fontsize=6)
            ax[row, col].set_xlabel("T")
            ax[row, col].set_ylabel("H")
            ax[row, col].set_zlabel("AQ")
            for t,h,a,ac in zip(obs[n][:,0],obs[n][:,1],obs[n][:,2],acv[n]):
               ax[row, col].text(t,h,a,ac, fontsize=2)
            n+=1   

    plt.subplots_adjust(left=0.13,
                    bottom=0.044,
                    right=0.91,
                    top=0.915,
                    wspace=0.32,
                    hspace=0.158)           
    plt.show()

def update_lines(num, obs, lines):
    for line, ob in zip(lines, obs):
        print(ob)
        line.set_data(ob[:num, :2].T)
        line.set_3d_properties(ob[:num, 2])
    return lines

def observations(obs):
    start_observation = obs[0]
    walk = start_observation + np.cumsum(obs, axis=0)
    #print('walk: ', walk)
    return obs
    
def plotAnimation(obs):
    num_steps = len(obs)
    #print(num_steps)
    
    walks = np.array([observations(obs)])

    fig =plt.figure()
    ax = fig.add_subplot(projection='3d')

    lines = [ax.plot([],[],[])[0] for _ in walks]

    # Setting the axes properties
    ax.set(xlim3d=(-120, 120), xlabel='T')
    ax.set(ylim3d=(-100, 100), ylabel='H')
    ax.set(zlim3d=(-20000, 20000), zlabel='AQ')

# Creating the Animation object
    ani = animation.FuncAnimation(
    fig, update_lines, num_steps, fargs=(walks, lines), interval=100)

    plt.show()

