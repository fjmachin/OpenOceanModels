import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Parameters
L = 1.0                   # Domain length (m)
nx = 400                  # Number of spatial points
dx = L / (nx - 1)
x = np.linspace(0, L, nx)

cfl = 0.8
t_max = 0.5

# Initial condition: step
u = np.zeros(nx)
u[x < 0.5 * L] = 1.0      # Step: u = 1 for x < 0.5, 0 elsewhere

dt = cfl * dx / np.max(u)
nt = int(t_max / dt)

u_new = u.copy()

# Plot setup
fig, ax = plt.subplots(figsize=(8, 4))
line, = ax.plot(x, u, lw=2)
time_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)
ax.set_ylim(-0.1, 1.1)
ax.set_xlim(0, L)
ax.set_xlabel("x (m)")
ax.set_ylabel("Velocity (u)")
ax.set_title("Nonlinear Advection")

def animate(frame):
    global u, u_new

    # Lax–Friedrichs scheme
    u_new[1:-1] = 0.5 * (u[2:] + u[:-2]) - \
                  dt / (2 * dx) * (0.5 * u[2:]**2 - 0.5 * u[:-2]**2)
    u_new[0] = 0
    u_new[-1] = 0
    u, u_new = u_new, u

    # Identify where the deforming front ends: first index where u[i]==1 and u[i+1]==1
    front_limit = nx
    for i in range(nx - 1):
        if u[i] == 1.0 and u[i+1] == 1.0:
            front_limit = i
            break

    # Only plot the deforming front: everything beyond front_limit is set to NaN
    u_masked = u.copy()
    if front_limit < nx:
        u_masked[front_limit+1:] = np.nan

    line.set_ydata(u_masked)
    time_text.set_text(f't = {frame * dt:.3f} s')
    return line, time_text

# Animate
frames = nt
anim = FuncAnimation(fig, animate, frames=frames, blit=False)

# Save
writer = FFMpegWriter(fps=10)
anim.save(r'C:\Users\Usuario\Dropbox\Universidad\Investigacion\OpenOceanModels\Codigo\Ejercicio9_advection\advection_step_front_clean.mp4', writer=writer)

plt.ioff()
plt.show()
