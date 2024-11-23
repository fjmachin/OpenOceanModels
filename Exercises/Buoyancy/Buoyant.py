import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
rho_sea = 1025.0  # Sea water density (kg/m^3)
g = 9.81          # Gravity (m/s^2)
rho_obj = 1025.5   # Density of object (kg/m^3)
N2 = 1.0e-4       # Stability frequency squared
r = 0.00          # Friction parameter
dt = 1.0          # Time step (seconds)
z_init = -80.0    # Initial depth (m)
w_init = 0.0      # Initial vertical speed (m/s)
ntot = 3600       # Total number of time steps (for 1 hour simulation)

# Initialize variables
z = z_init        # Depth at time level n
w = w_init        # Vertical speed at time level n
time = 0.0        # Time counter

# Create figure for animation
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 30)  # Time in minutes
ax.set_ylim(-100, 0)  # Depth (m)
ax.set_xlabel("Time (minutes)")
ax.set_ylabel("Depth (m)")
line, = ax.plot([], [], lw=2)

# Initial equilibrium depth (position where the buoy is at rest)
z_equilibrium = -50.0  # Modify this as needed

# Add horizontal line at equilibrium position
ax.axhline(y=z_equilibrium, color='black', linestyle='--', label="Equilibrium Position")

line, = ax.plot([], [], lw=2)

# Output data for later analysis
output_data = []

# Function to calculate density at a given depth
def density(z, N2):
    return rho_sea * (1. - (N2 * z) / g)

# Main simulation loop
def update_wave(n):
    global z, w, time
    rhosea = density(z, N2)  # Calculate sea water density at current depth
    bf = -g * (rho_obj - rhosea) / rho_obj  # Calculate buoyancy force
    wn = (w + dt * bf) / (1. + r * dt)  # New vertical speed prediction
    zn = z + dt * wn  # New depth prediction
    zn = min(zn, 0.0)  # Constrain depth by sea surface
    zn = max(zn, -100.0)  # Constrain depth by seafloor

    # Update depth and speed for next time step
    w = wn
    z = zn
    time = n * dt / 60  # Convert time to minutes

    # Store data for later use (time, depth, speed, density)
    if n % 10 == 0:
        output_data.append([time, z, w, rho_obj - 1000])

    return time, z

# Animation function
def animate(n):
    time, z = update_wave(n)
    line.set_data(np.array([output_data[i][0] for i in range(len(output_data))]), 
                  np.array([output_data[i][1] for i in range(len(output_data))]))
    return line,

# Create animation with either 30 minutes or full simulation time
total_time_in_minutes = 30  # Set to 30 minutes or to ntot/60 for full simulation
frames = int(total_time_in_minutes * 60 / dt)  # Calculate frames based on time duration

ani = FuncAnimation(fig, animate, frames=frames, interval=50, blit=True)

# Save the animation as an .mp4 file
ani.save('buoy_simulation_without_friction.mp4', writer='ffmpeg', fps=30)

# Show the animation
plt.tight_layout()
plt.show()

# Optional: Save the output data to a file (for further analysis)
np.savetxt("output_data.txt", output_data, fmt='%12.4f', header="Time Depth Speed Relative_Density")
