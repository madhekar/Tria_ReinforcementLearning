import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython import display
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
plt.ion()

def plot(scores, mean_scores, actions):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training Test...')
    plt.xlabel('Number of Episode/ steps')
    plt.ylabel('Reward')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.plot(actions)
    plt.ylim(ymin=-30)
    plt.text(len(scores) -1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) -1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)