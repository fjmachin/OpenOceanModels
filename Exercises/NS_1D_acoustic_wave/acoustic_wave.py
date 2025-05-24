import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Physical parameters
c = 340.0               # Speed of sound (m/s)
L = 1.0                 # Domain length (m)
nx = 200                # Number of spatial points
dx = L / (nx - 1)       # Spatial step
x = np.linspace(0, L, nx)

# Initial conditions: Gaussian pressure pulse
sigma = 0.05
p0 = np.exp(-((x - L/2)**2) / (2 * sigma**2))  # Initial pressure
p_prev = p0.copy()                             # Pressure at t - dt
p = p0.copy()                                  # Pressure at t
p_next = np.zeros_like(p)                      # Pressure at t + dt

# Time setup
cfl = 0.9                    # Courant number (must be < 1)
dt = cfl * dx / c            # Time step
t_max = 0.01                 # Total simulation time
nt = int(t_max / dt)         # Number of time steps

# Plot configuration
fig, ax = plt.subplots()
line, = ax.plot(x, p, lw=2)
ax.set_ylim(-1.1, 1.1)
ax.set_xlabel("x (m)")
ax.set_ylabel("Pressure (non-dimensional)")
ax.set_title("1D Acoustic Wave Propagation")

def animate(frame):
    global p, p_prev, p_next
    for i in range(1, nx - 1):
        p_next[i] = (2 * p[i] - p_prev[i] +
                     (c * dt / dx)**2 * (p[i+1] - 2 * p[i] + p[i-1]))
    # Rigid wall boundary conditions
    p_next[0] = 0
    p_next[-1] = 0

    # Rotate time layers
    p_prev, p, p_next = p, p_next, p_prev
    line.set_ydata(p)
    return line,

# Create the animation
frames = nt
anim = FuncAnimation(fig, animate, frames=frames, blit=False)

# Save the animation as an MP4 file using FFMpegWriter
writer = FFMpegWriter(fps=30)
anim.save(r'C:\Users\Usuario\Dropbox\Universidad\Investigacion\OpenOceanModels\Codigo\Ejercicio7_acoustic_wave\1D_acoustic_wave.mp4', writer=writer)

plt.ioff()
plt.show()
