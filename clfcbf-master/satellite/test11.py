import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

x1 = np.arange(0, -0.2, -0.002)
y1 = np.arange(0, -0.2, -0.002)
x2 = np.arange(3.9, 3.7, -0.002)
y2 = np.arange(0, 1, 0.01)
x3 = np.arange(0, 1.8, 0.018)
y3 = np.array(x3**2)

fig,ax = plt.subplots()

def animate(i):
    ax.clear()
    ax.set_xlim(-4,4)
    ax.set_ylim(-4,4)
    line, = ax.plot(x1[0:i], y1[0:i], color = 'blue', lw=1)
    line2, = ax.plot(x2[0:i], y2[0:i], color = 'red', lw=1)
    line3, = ax.plot(x3[0:i], y3[0:i], color = 'purple', lw=1)
    point1, = ax.plot(x1[i], y1[i], marker='.', color='blue')
    point2, = ax.plot(x2[i], y2[i], marker='.', color='red')
    point3, = ax.plot(x3[i], y3[i], marker='.', color='purple')
#     print(i)
    return line, line2, line3, point1, point2, point3,
        
ani = FuncAnimation(fig, animate, interval=40, blit=True, repeat=True, frames=100)    
ani.save("TLI.gif", dpi=300, writer=PillowWriter(fps=25))