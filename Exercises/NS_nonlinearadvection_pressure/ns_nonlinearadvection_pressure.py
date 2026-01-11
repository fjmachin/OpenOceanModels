import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Physical and simulation parameters
rho = 1.0          # Density (kg/m³)
u0 = 1.0           # Initial velocity at x = 0 (m/s)
L = 1.0            # Domain length (m)
nx = 200           # Number of spatial points
x = np.linspace(0, L, nx)

# Animation parameters
n_frames = 900
a_max = 5.0        # Maximum pressure gradient in absolute value (Pa/m)
fps = 30

# Set up the figure and axes
fig, ax = plt.subplots(figsize=(8, 5))
color_u = 'tab:blue'
color_p = 'tab:red'

line_u, = ax.plot([], [], color=color_u, lw=2, label='Velocity $u(x)$')
ax.set_xlabel("x (m)")
ax.set_ylabel("Velocity $u(x)$", color=color_u)
ax.set_xlim(0, L)
ax.set_ylim(0, 4)
ax.tick_params(axis='y', labelcolor=color_u)

# Second axis for pressure
ax2 = ax.twinx()
line_p, = ax2.plot([], [], color=color_p, ls='--', lw=2, label='Pressure $p(x)$')
ax2.set_ylabel("Pressure $p(x)$", color=color_p)
ax2.set_ylim(-rho * a_max * L * 1.1, 0)
ax2.tick_params(axis='y', labelcolor=color_p)

# Title and time annotation
title = ax.set_title("")
text_slope = ax.text(0.02, 0.85, '', transform=ax.transAxes)

def animate(frame):
    a = (frame / (n_frames - 1)) * a_max  # From 0 to a_max
    p = -rho * a * x
    u_squared = 2 * a * x + u0**2
    u = np.sqrt(u_squared)

    line_u.set_data(x, u)
    line_p.set_data(x, p)
    title.set_text("Nonlinear Advection vs Pressure Gradient")
    text_slope.set_text(f'∂p/∂x = {-a:.2f} Pa/m')
    return line_u, line_p, title, text_slope

anim = FuncAnimation(fig, animate, frames=n_frames, blit=False)

# Save animation
writer = FFMpegWriter(fps=fps)
output_path = r"C:\Users\Usuario\Dropbox\Universidad\Investigacion\OpenOceanModels\Codigo\Ejercicio10\nonlinear_pressure_balance.mp4"
anim.save(output_path, writer=writer)

plt.close()
