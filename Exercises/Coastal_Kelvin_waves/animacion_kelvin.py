import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Registers 3D projection
from matplotlib import animation
from matplotlib.animation import FFMpegWriter

# -------------------------------
# Read input data
# -------------------------------
eta = np.loadtxt("eta.dat")    # Simulation data (stacked frames)
eta0 = np.loadtxt("eta0.dat")  # Initial sea surface elevation

n_rows, n_cols = eta.shape
frames = n_rows // 51         # Each frame occupies 51 rows
eta_frames = np.array([eta[i*51:(i+1)*51, :] for i in range(frames)])

# -------------------------------
# Create grid for plotting
# -------------------------------
x = np.linspace(1, 201, 201)
y = np.linspace(1, 51, 51)
X, Y = np.meshgrid(x, y)

# -------------------------------
# Set up the figure and axes
# -------------------------------
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 201)
ax.set_ylim(1, 51)
ax.set_zlim(-20, 20)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Global variable for the colorbar
cb = None

def init():
    """Initialize the surface plot and create the colorbar once."""
    global cb
    Z = 20 * (eta_frames[0] - eta0)
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', vmin=-20, vmax=20)
    ax.set_title("Frame 1")
    # Create the colorbar once and store it in the global variable
    cb = fig.colorbar(surf, ax=ax)
    return surf,

def animate(i):
    """Update the surface plot and update the existing colorbar."""
    global cb
    # Remove the previous surface objects
    for coll in list(ax.collections):
        coll.remove()
    Z = 20 * (eta_frames[i] - eta0)
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', vmin=-20, vmax=20)
    ax.set_title(f"Frame {i+1}")
    # Update the colorbar to refer to the new surface
    cb.update_normal(surf)
    return surf,

ani = animation.FuncAnimation(fig, animate, frames=frames, init_func=init,
                              interval=100, blit=False)

# -------------------------------
# Save the animation as an MP4 file using FFMpegWriter
# -------------------------------
# Ensure ffmpeg is installed and available in PATH
writer = FFMpegWriter(fps=2)
ani.save("kelvin_animation.mp4", writer=writer)

plt.show()





