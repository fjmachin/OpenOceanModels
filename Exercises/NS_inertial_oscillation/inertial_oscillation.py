import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

# Physical parameters
f      = 1e-4       # Coriolis parameter [s⁻¹], positive in NH
dt     = 250        # time step [s]
t_max  = 2 * 24*3600  # total simulation time: 2 days [s]
n_steps = int(t_max / dt)

# Initial conditions
u0, v0     = 1.0, 0.0       # eastward, northward velocity [m/s]
U0, V0     = 0.0, 0.0       # no translation
U1, V1     = -0.3, -0.3     # translation toward SW

# Time array
t = np.linspace(0, t_max, n_steps)

# Analytic solution for velocities
phi = f * t
u =  u0 * np.cos(phi) + v0 * np.sin(phi)
v =  v0 * np.cos(phi) - u0 * np.sin(phi)

# Analytic integration to get positions
x = (u0 * np.sin(phi) - v0 * (1 - np.cos(phi))) / f
y = (v0 * np.sin(phi) + u0 * (np.cos(phi) - 1)) / f

# Build two translated trajectories
x0 = x + U0 * t
y0 = y + V0 * t
x1 = x + U1 * t
y1 = y + V1 * t

# Video parameters for a 1-minute output at 24 fps
duration = 60
fps      = 24
n_frames = duration * fps
step     = max(1, n_steps // n_frames)
frames   = np.arange(0, n_steps, step)

# Set up the figure and subplots
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 8))
for ax in (ax0, ax1):
    ax.set_aspect('equal')
    ax.set_xlim(min(x0.min(), x1.min()) - 1000, max(x0.max(), x1.max()) + 1000)
    ax.set_ylim(min(y0.min(), y1.min()) - 1000, max(y0.max(), y1.max()) + 1000)
    ax.set_xlabel('x (m east)')
    ax.set_ylabel('y (m north)')

ax0.set_title('Inertial Oscillation (No Translation)')
ax1.set_title('Inertial Oscillation (SW Translation)')

# Initialize lines and points
line0, = ax0.plot([], [], lw=2)
point0, = ax0.plot([], [], 'ro')
line1, = ax1.plot([], [], lw=2)
point1, = ax1.plot([], [], 'ro')

def init():
    for ln, pt in [(line0, point0), (line1, point1)]:
        ln.set_data([], [])
        pt.set_data([], [])
    return line0, point0, line1, point1

def animate(i):
    idx = frames[i]
    # Left subplot (no translation)
    line0.set_data(x0[:idx+1], y0[:idx+1])
    point0.set_data([x0[idx]], [y0[idx]])
    # Right subplot (southwest translation)
    line1.set_data(x1[:idx+1], y1[:idx+1])
    point1.set_data([x1[idx]], [y1[idx]])
    return line0, point0, line1, point1

ani = animation.FuncAnimation(
    fig, animate,
    frames=len(frames),
    init_func=init,
    interval=1000/fps,  # milliseconds between frames
    blit=True
)

# Save to MP4 (uses default codec for maximum compatibility)
writer = FFMpegWriter(fps=fps, bitrate=2000)
output_path = r"C:\Users\Usuario\Dropbox\Universidad\Investigacion\OpenOceanModels\Codigo\Ejercicio6\dual_inertial_oscillations.mp4"
ani.save(output_path, writer=writer)

plt.tight_layout()
plt.show()
