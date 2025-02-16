import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# -------------------------------------------------
# Domain parameters
# -------------------------------------------------
Lx = 5000e3      # Zonal extent (m)
Ly = 3000e3      # Meridional extent (m)
nx = 100         # Number of grid points in x (model)
ny = 100         # Number of grid points in y (model)
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
# Create meshgrid for plotting; meshgrid returns arrays of shape (ny, nx)
Xgrid, Ygrid = np.meshgrid(x, y)

# -------------------------------------------------
# Physical parameters
# -------------------------------------------------
tau0 = 0.1       # Maximum wind stress amplitude (N/m^2)
beta = 2e-11     # Coriolis parameter gradient (1/m/s)
R_val = 1e-6     # Linear friction coefficient (1/s)
rho = 1025       # Water density (kg/m^3)
alpha = beta / R_val  # Intensification parameter

# -------------------------------------------------
# Extended grid for wind field (indices 0 to ny+1)
# -------------------------------------------------
ny_ext = ny + 2
y_ext = np.linspace(0, Ly, ny_ext)

# -------------------------------------------------
# Animation parameters
# -------------------------------------------------
frames = 50

# -------------------------------------------------
# Create figure and axes
# -------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# Left subplot: wind stress plot
ax1.set_xlabel(r'$\tau_x$ (N/m$^2$)')
ax1.set_ylabel('Meridional distance (km)')
ax1.set_title('Wind Stress Distribution')
ax1.grid(True)
ax1.set_xlim([-tau0 - 0.05, tau0 + 0.05])
ax1.set_ylim([y_ext[0] / 1e3, y_ext[-1] / 1e3])
# Right subplot: gyre plot
ax2.set_xlabel('Zonal distance (km)')
ax2.set_ylabel('Meridional distance (km)')
ax2.set_title('Stommel Gyre')
ax2.grid(True)

# Create initial plot objects as globals
global line1, im, Q
line1, = ax1.plot([], [], 'b', linewidth=2)  # Wind stress line
im = ax2.imshow(np.zeros((ny, nx)), extent=(0, Lx / 1e3, 0, Ly / 1e3),
                origin='lower', cmap='viridis', vmin=-2e7, vmax=0)
# Create a single colorbar for the imshow
cbar = fig.colorbar(im, ax=ax2)
Q = ax2.quiver(Xgrid / 1e3, Ygrid / 1e3, np.zeros_like(Xgrid), np.zeros_like(Ygrid),
               color='k', scale=1, scale_units='xy')

def animate(frame):
    global line1, im, Q  # Ensure global variables are used
    scale = frame / frames  # Scale factor from 0 to 1

    # -------------------------------------------------
    # Define wind stress on the extended grid:
    # For j = 0,...,ny+1:
    #   tau_x(j) = -tau0 * cos[ (j-1)/(ny-1)*pi ]
    # scaled by 'scale'.
    # -------------------------------------------------
    j_vals = np.arange(0, ny_ext)
    taux_ext = scale * (-tau0 * np.cos((j_vals - 1) / (ny - 1) * np.pi))
    
    # -------------------------------------------------
    # Compute d(tau_x)/dy on the extended grid using central differences.
    # -------------------------------------------------
    dy_ext = y_ext[1] - y_ext[0]
    dtau_dy_ext = np.zeros(ny_ext)
    dtau_dy_ext[1:-1] = (taux_ext[2:] - taux_ext[:-2]) / (2 * dy_ext)
    dtau_dy_ext[0] = (taux_ext[1] - taux_ext[0]) / dy_ext
    dtau_dy_ext[-1] = (taux_ext[-1] - taux_ext[-2]) / dy_ext
    
    # -------------------------------------------------
    # Interpolate the derivative onto the model grid (vector of length ny).
    # -------------------------------------------------
    forcing_deriv = np.interp(y, y_ext, dtau_dy_ext)
    
    # Forcing term in the Stommel equation: (1/(rho*R)) * d(tau_x)/dy
    RHS_vec = (1 / (rho * R_val)) * forcing_deriv  # vector of length ny

    # -------------------------------------------------
    # Assemble the finite-difference system using an upwind scheme for ψₓ:
    # For interior nodes (i = 1...nx-2, j = 1...ny-2):
    #   (ψ[j, i+1] - 2ψ[j, i] + ψ[j, i-1]) / dx² +
    #   (ψ[j+1, i] - 2ψ[j, i] + ψ[j-1, i]) / dy² +
    #   α*(ψ[j, i] - ψ[j, i-1]) / dx = RHS_vec[j]
    # with Dirichlet boundary conditions: ψ = 0.
    # Note: ψ is stored in an array of shape (ny, nx) (rows = y, columns = x).
    # -------------------------------------------------
    N = nx * ny
    A = lil_matrix((N, N))
    b = np.zeros(N)
    
    for j_idx in range(ny):
        for i_idx in range(nx):
            idx = i_idx + j_idx * nx
            if i_idx == 0 or i_idx == nx - 1 or j_idx == 0 or j_idx == ny - 1:
                A[idx, idx] = 1
                b[idx] = 0
            else:
                A[idx, idx - 1] = 1 / dx**2 - alpha / dx  # West neighbor (upwind)
                A[idx, idx + 1] = 1 / dx**2               # East neighbor
                A[idx, idx - nx] = 1 / dy**2              # South neighbor
                A[idx, idx + nx] = 1 / dy**2              # North neighbor
                A[idx, idx] = -2 / dx**2 - 2 / dy**2 + alpha / dx
                b[idx] = RHS_vec[j_idx]
    
    A = A.tocsr()
    psi_vec = spsolve(A, b)
    psi = psi_vec.reshape((ny, nx))
    
    # -------------------------------------------------
    # Compute velocity field: u = ∂ψ/∂y, v = -∂ψ/∂x.
    # -------------------------------------------------
    dpsi_dy, dpsi_dx = np.gradient(psi, dy, dx)
    u = dpsi_dy      # u = ∂ψ/∂y
    v = -dpsi_dx     # v = -∂ψ/∂x
    
    # -------------------------------------------------
    # Update the wind stress plot (ax1)
    # -------------------------------------------------
    line1.set_data(taux_ext, y_ext / 1e3)
    
    # -------------------------------------------------
    # Update the ψ image (ax2) and the quiver plot
    # -------------------------------------------------
    im.set_data(psi)
    Q.set_UVC(u, v)
    ax2.set_title(f'Stommel Gyre (Frame {frame}/{frames})')
    # Ensure the vertical limits of ax2 match those of ax1
    ax2.set_ylim(ax1.get_ylim())
    
    return line1, im, Q

# Create the animation using FuncAnimation
anim = FuncAnimation(fig, animate, frames=frames, blit=False)

# Save the animation as an MP4 file using FFMpegWriter
writer = FFMpegWriter(fps=2)
anim.save("stommel_gyre.mp4", writer=writer)

plt.ioff()
plt.show()
