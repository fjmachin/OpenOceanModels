import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import cumulative_trapezoid

# Model parameters
L_x = 5000e3  # Zonal extent (m)
L_y = 3000e3  # Meridional extent (m)
nx = 100      # Number of points in the x direction
ny = 100      # Number of points in the y direction
frames = 50   # Number of animation frames

# Maximum wind stress amplitude (N/m^2)
tau_0 = 0.1

# Gradient of the Coriolis parameter (1/m/s)
beta = 2e-11

# Meridional and zonal coordinates
y = np.linspace(0, L_y, ny)  # Meridional coordinate (m)
x = np.linspace(0, L_x, nx)  # Zonal coordinate (m)

# Idealized wind stress (sinusoidal)
tau_x_full = tau_0 * np.sin(2 * np.pi * (y / L_y - 0.5))

# Function to compute the streamfunction (psi) for a given wind stress
def compute_psi(tau_x):
    dy = y[1] - y[0]
    dtau_dy = np.gradient(tau_x, dy)
    v = dtau_dy / beta
    u = cumulative_trapezoid(v, y, initial=0)  # Discrete integral
    psi = np.zeros((ny, nx))
    for i in range(nx):
        psi[:, i] = cumulative_trapezoid(u, y, initial=0) * (1 - i / (nx - 1))
    return psi

# Figure setup
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ax1, ax2 = axes

# Wind stress plot
wind_line, = ax1.plot(tau_x_full * 0, y / 1e3, 'b', linewidth=1.5)
ax1.set_title("Wind Stress ($\\tau_x$)")
ax1.set_xlabel("$\\tau_x$ (N/m²)")
ax1.set_ylabel("Meridional Distance (km)")
ax1.set_xlim(-0.1, 0.1)
ax1.grid()

# Initial contour plot for streamfunction
psi_initial = compute_psi(tau_x_full * 0.01)
contour = ax2.contourf(x / 1e3, y / 1e3, psi_initial, levels=20, cmap="viridis")
cbar = plt.colorbar(contour, ax=ax2)
contour.set_clim(-4.6e15, 0)  # Set color limits
ax2.set_title("Subtropical Gyre ($\\psi$)")
ax2.set_xlabel("Zonal Distance (km)")
ax2.set_ylabel("Meridional Distance (km)")
ax2.grid()

# Update function for the animation
def update(frame):
    global contour
    # Interpolate wind stress
    tau_x = tau_x_full * (frame / frames)
    wind_line.set_xdata(tau_x)

    # Recalculate streamfunction
    psi_partial = compute_psi(tau_x)

    # Update contour plot
    for c in ax2.collections:
        c.remove()
    contour = ax2.contourf(x / 1e3, y / 1e3, psi_partial, levels=20, cmap="viridis")
    contour.set_clim(-4.6e15, 0)  # Adjust color limits
    cbar.update_normal(contour)
    ax2.set_title(f"Subtropical Gyre ($\\psi$) - Frame {frame + 1}/{frames}")

# Create the animation
ani = FuncAnimation(fig, update, frames=frames, interval=100)

# Save or show the animation
# Uncomment the next line to save the animation as an MP4 file
ani.save("wind_to_gyre.mp4", writer="ffmpeg", fps=2)

plt.show()
