import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

fig = plt.figure()


def f(x, y):
    return np.sin(x) + np.cos(y)

x = np.linspace(0, 2 * np.pi, 400)
y = np.linspace(0, 2 * np.pi, 400).reshape(-1, 1)

im = plt.imshow(f(x, y), animated=True)

count=0
t0=time.time()+1
def updatefig(*args):
    global x, y,count,t0
    x += np.pi / 15.
    y += np.pi / 20.
    im.set_array(f(x, y))
    if time.time()<t0:
        count+=1
    else:
        print (count)
        count=0
        t0=time.time()+1     
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
plt.show()