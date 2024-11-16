# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 09:19:30 2024

@author: Francisco Machín (francisco.machin@ulpgc.es)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\...\ffmpeg-<version>\Library\bin\ffmpeg.exe'

# Wave parameters
len1 = 100.0      # Wavelength of wave 1
per1 = 60.0       # Period of wave 1
len2 = 95.0       # Wavelength of wave 2
per2 = 30.0       # Period of wave 2
amp = 1.0         # Amplitude of both waves
xrange = 10 * len1  # x-range shown in animation
x = np.linspace(0, xrange, 500)  # Discrete points in x-direction

# Time parameters
t = 0.0                   # Start time
trange = 10 * per1        # Simulate ten periods of wave 1
dt = trange / 200.0       # Time step
ntot = int(trange / dt)   # Total number of iteration steps

# Create figure and subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 8))

# Initialize plots
line1, = axs[0].plot([], [], 'b', linewidth=2)
line2, = axs[1].plot([], [], 'r', linewidth=2)
line3, = axs[2].plot([], [], 'g', linewidth=2)

# Set up each subplot
for ax in axs:
    ax.set_ylim(-2, 2)
    ax.set_xlim(0, 1000)
    ax.set_xticks([0, 200, 400, 600, 800, 1000])
    ax.set_yticks([-2, -1, 0, 1, 2])
axs[0].set_title("Wave 1", fontsize=10)
axs[1].set_title("Wave 2", fontsize=10)
axs[2].set_title("Wave 1 + Wave 2", fontsize=10)

# Animation function
def animate(n):
    global t  # Use global to modify t from the outer scope
    f1 = amp * np.sin(2 * np.pi * (x / len1 - t / per1))  # Wave 1 equation
    f2 = amp * np.sin(2 * np.pi * (x / len2 - t / per2))  # Wave 2 equation
    f3 = f1 + f2  # Superposition of waves
    
    line1.set_data(x, f1)
    line2.set_data(x, f2)
    line3.set_data(x, f3)
    
    t += dt  # Advance time
    return line1, line2, line3

# Create animation
ani = FuncAnimation(fig, animate, frames=ntot, blit=True, repeat=False)

# Show animation
plt.tight_layout()
plt.show()

ani.save(r'C:\path\omb_2.mp4', writer='ffmpeg', fps=15, dpi=300)


