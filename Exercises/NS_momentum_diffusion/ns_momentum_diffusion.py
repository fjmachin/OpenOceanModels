import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Physical parameters
nu = 0.01               # Kinematic viscosity (m^2/s)
L = 1.0                 # Length of the domain (m)
nx = 200                # Number of spatial points
dx = L / (nx - 1)       # Spatial step
x = np.linspace(0, L, nx)

# Initial condition: Gaussian velocity pulse
sigma = 0.05
u = np.exp(-((x - L/2)**2) / (2 * sigma**2))
u_prev = u.copy()
u_next = np.zeros_like(u)

# Time setup
cfl = 0.4
dt = cfl * dx**2 / nu   # Stability condition for explicit scheme
t_max = 1
nt = int(t_max / dt)

# Plot setup
fig, ax = plt.subplots()
line, = ax.plot(x, u, lw=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.set_ylim(-0.1, 1.1)
ax.set_xlabel("x (m)")
ax.set_ylabel("Velocity (u)")
ax.set_title("Momentum Diffusion (1D)")

def animate(frame):
    global u, u_next
    for i in range(1, nx - 1):
        u_next[i] = u[i] + nu * dt / dx**2 * (u[i+1] - 2*u[i] + u[i-1])
    # No-slip boundary conditions
    u_next[0] = 0
    u_next[-1] = 0
    u, u_next = u_next, u
    line.set_ydata(u)
    time_text.set_text(f"t = {frame * dt:.3f} s")
    return line, time_text

# Create animation
frames = nt
anim = FuncAnimation(fig, animate, frames=frames, blit=False)

# Save animation
writer = FFMpegWriter(fps=30)
anim.save(r'C:\Users\Usuario\Dropbox\Universidad\Investigacion\OpenOceanModels\Codigo\Ejercicio8_momentum_diffusion\momentum_diffusion_1D.mp4', writer=writer)

plt.ioff()
plt.show()
