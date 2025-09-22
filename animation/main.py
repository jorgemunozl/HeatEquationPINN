import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

x = np.linspace(0, 2*np.pi, 100)
epochs = 50
data = [np.sin(x + i/5) for i in range(epochs)]

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)

ax.set_xlim(0, 2*np.pi)
ax.set_ylim(-1.2, 1.2)

def init():
    line.set_data([], [])
    return line,

def update(frame):
    y = data[frame]
    line.set_data(x, y)
    ax.set_title(f"Epoch {frame}")
    return line,

ani = animation.FuncAnimation(
    fig, update, frames=epochs, init_func=init,
    blit=True, interval=100, repeat=False
)

# To save as mp4 (needs ffmpeg installed):
ani.save("animation/training.mp4", writer="ffmpeg")
