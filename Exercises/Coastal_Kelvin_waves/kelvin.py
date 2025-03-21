import numpy as np

class KelvinModel:
    def __init__(self):
        # Grid parameters and constants
        self.nx = 201
        self.ny = 51
        self.PI = 3.1416  # keep original value (could also use np.pi)

        # All arrays are defined with size (ny+2, nx+2) to include ghost boundaries
        shape = (self.ny + 2, self.nx + 2)
        self.hzero = np.full(shape, 10.0)    # base bathymetry
        self.h = np.zeros(shape)             # total depth
        self.eta = np.zeros(shape)           # sea surface elevation
        self.etan = np.zeros(shape)          # predicted sea surface elevation
        self.u = np.zeros(shape)             # x-velocity
        self.un = np.zeros(shape)            # updated u-velocity
        self.v = np.zeros(shape)             # y-velocity
        self.vn = np.zeros(shape)            # updated v-velocity
        self.T = np.zeros(shape)             # tracer (e.g., temperature or salinity)
        self.TN = np.zeros(shape)            # updated tracer
        self.wet = np.ones(shape, dtype=int) # mask: 1 = wet, 0 = dry

        # Arrays for nonlinear terms and auxiliary variables for advection
        self.CuP = np.zeros(shape)
        self.CuN = np.zeros(shape)
        self.CvP = np.zeros(shape)
        self.CvN = np.zeros(shape)
        self.Cu = np.zeros(shape)
        self.Cv = np.zeros(shape)
        self.B = np.zeros(shape)
        self.BN = np.zeros(shape)
        self.advx = np.zeros(shape)
        self.advy = np.zeros(shape)

        # Physical and numerical parameters
        self.dt = 10.0         # time interval [s]
        self.dx = 2000.0       # spatial resolution in x [m]
        self.dy = 2000.0       # spatial resolution in y [m]
        self.g = 9.81          # gravitational acceleration [m/s²]
        self.f = 5.e-4         # Coriolis parameter [1/s]
        self.r = 0.0           # bottom friction (disabled)
        self.rho = 1028.0      # water density [kg/m³]
        beta_temp = 0.5 * self.f * self.dt
        self.beta = beta_temp * beta_temp  # coefficient for semi-implicit treatment of Coriolis
        self.eps = 0.05        # parameter for the Shapiro filter
        self.hmin = 0.10       # minimum depth to define a "wet" cell
        self.taux = 0.0        # wind forcing in x
        self.tauy = 0.0        # wind forcing in y

        # Advection scheme: 1: upstream, 2: Lax-Wendroff, 3: Superbee, 4: Super-C
        self.mode = 3

        # Set up bathymetry and land boundaries:
        # Set hzero = 10 everywhere, except:
        self.hzero[0:5, :] = -10.0      # first 5 rows (coast)
        self.hzero[self.ny + 1, :] = -10.0  # last row
        self.hzero[:, 0] = -10.0          # first column
        self.hzero[:, self.nx + 1] = -10.0  # last column

        # Initial condition for sea surface elevation: eta = -min(0, hzero)
        self.eta = -np.minimum(0, self.hzero)
        self.etan = self.eta.copy()
        # Total depth is defined as h = hzero + eta
        self.h = self.hzero + self.eta
        # Define wet cells (wet = 1) if h >= hmin, otherwise dry (0)
        self.wet[self.h < self.hmin] = 0

    def psi(self, r2, cfl):
        """
        Vectorized psi function according to the chosen advection scheme.
        Works on arrays (or scalars) and depends on self.mode.
        """
        if self.mode == 1:
            return np.zeros_like(r2)
        elif self.mode == 2:
            return np.ones_like(r2)
        elif self.mode == 3:
            term1 = np.minimum(2.0 * r2, 1.0)
            term2 = np.minimum(r2, 2.0)
            term3 = np.maximum(term1, term2)
            return np.maximum(term3, 0.0)
        elif self.mode == 4:
            psi_val = np.zeros_like(r2)
            mask = r2 > 0.0
            # cfl is a scalar
            psi_val[mask] = np.where(
                r2[mask] > 1.0,
                np.minimum(r2[mask], 2.0 / (1.0 - cfl)),
                np.minimum(2.0 * r2[mask] / cfl, 1.0)
            )
            return psi_val
        else:
            return np.zeros_like(r2)

    def advect(self):
        """
        Vectorized implementation of the ADVECT subroutine.
        Calculates the BN term from the variable B using horizontal and vertical differences.
        Adjusted slicing ensures that all arrays have matching shapes.
        """
        ny, nx = self.ny, self.nx

        # Horizontal differences for RxP:
        dB_h = self.B[1:ny+1, 2:nx+2] - self.B[1:ny+1, 1:nx+1]  # shape (ny, nx)
        num_h = self.B[1:ny+1, 1:nx+1] - self.B[1:ny+1, 0:nx]     # shape (ny, nx)
        RxP = np.where(np.abs(dB_h) > 0, num_h / dB_h, 0)

        # Horizontal differences for RxN:
        dB_hN = self.B[1:ny+1, 1:nx+1] - self.B[1:ny+1, 0:nx]     # shape (ny, nx)
        num_hN = self.B[1:ny+1, 2:nx+2] - self.B[1:ny+1, 1:nx+1]   # shape (ny, nx)
        RxN = np.where(np.abs(dB_hN) > 0, num_hN / dB_hN, 0)

        # Vertical differences for RyP:
        dB_v = self.B[2:ny+2, 1:nx+1] - self.B[1:ny+1, 1:nx+1]     # shape (ny, nx)
        num_v = self.B[1:ny+1, 1:nx+1] - self.B[0:ny, 1:nx+1]       # shape (ny, nx)
        RyP = np.where(np.abs(dB_v) > 0, num_v / dB_v, 0)

        # Vertical differences for RyN:
        dB_vN = self.B[1:ny+1, 1:nx+1] - self.B[0:ny, 1:nx+1]       # shape (ny, nx)
        num_vN = self.B[2:ny+2, 1:nx+1] - self.B[1:ny+1, 1:nx+1]     # shape (ny, nx)
        RyN = np.where(np.abs(dB_vN) > 0, num_vN / dB_vN, 0)

        # Compute horizontal flux terms
        BwP = ( self.B[1:ny+1, 0:nx]
               + 0.5 * self.psi(RxP, self.Cu[1:ny+1, 0:nx]) * (1 - self.CuP[1:ny+1, 0:nx])
               * (self.B[1:ny+1, 1:nx+1] - self.B[1:ny+1, 0:nx]) )
        BwN = ( self.B[1:ny+1, 1:nx+1]
               - 0.5 * self.psi(RxN, self.Cu[1:ny+1, 0:nx]) * (1 + self.CuN[1:ny+1, 0:nx])
               * (self.B[1:ny+1, 1:nx+1] - self.B[1:ny+1, 0:nx]) )
        BeP = ( self.B[1:ny+1, 1:nx+1]
               + 0.5 * self.psi(RxP, self.Cu[1:ny+1, 1:nx+1]) * (1 - self.CuP[1:ny+1, 1:nx+1])
               * (self.B[1:ny+1, 2:nx+2] - self.B[1:ny+1, 1:nx+1]) )
        BeN = ( self.B[1:ny+1, 2:nx+2]
               - 0.5 * self.psi(RxN, self.Cu[1:ny+1, 1:nx+1]) * (1 + self.CuN[1:ny+1, 1:nx+1])
               * (self.B[1:ny+1, 2:nx+2] - self.B[1:ny+1, 1:nx+1]) )

        # Compute vertical flux terms
        BsP = ( self.B[0:ny, 1:nx+1]
               + 0.5 * self.psi(RyP, self.Cv[0:ny, 1:nx+1]) * (1 - self.CvP[0:ny, 1:nx+1])
               * (self.B[1:ny+1, 1:nx+1] - self.B[0:ny, 1:nx+1]) )
        BsN = ( self.B[1:ny+1, 1:nx+1]
               - 0.5 * self.psi(RyN, self.Cv[0:ny, 1:nx+1]) * (1 + self.CvN[0:ny, 1:nx+1])
               * (self.B[1:ny+1, 1:nx+1] - self.B[0:ny, 1:nx+1]) )
        BnP = ( self.B[1:ny+1, 1:nx+1]
               + 0.5 * self.psi(RyP, self.Cv[1:ny+1, 1:nx+1]) * (1 - self.CvP[1:ny+1, 1:nx+1])
               * (self.B[2:ny+2, 1:nx+1] - self.B[1:ny+1, 1:nx+1]) )
        BnN = ( self.B[2:ny+2, 1:nx+1]
               - 0.5 * self.psi(RyN, self.Cv[1:ny+1, 1:nx+1]) * (1 + self.CvN[1:ny+1, 1:nx+1])
               * (self.B[2:ny+2, 1:nx+1] - self.B[1:ny+1, 1:nx+1]) )

        term1 = self.CuP[1:ny+1, 0:nx] * BwP + self.CuN[1:ny+1, 0:nx] * BwN
        term2 = self.CuP[1:ny+1, 1:nx+1] * BeP + self.CuN[1:ny+1, 1:nx+1] * BeN
        term3 = self.CvP[0:ny, 1:nx+1] * BsP + self.CvN[0:ny, 1:nx+1] * BsN
        term4 = self.CvP[1:ny+1, 1:nx+1] * BnP + self.CvN[1:ny+1, 1:nx+1] * BnN

        self.BN[1:ny+1, 1:nx+1] = term1 - term2 + term3 - term4

    def dyn(self):
        """
        Vectorized implementation of the dynamic evolution (DYN subroutine).
        Updates the velocity, elevation, tracer, etc. fields.
        Block operations (slicing) are used for the interior region.
        """
        ny, nx = self.ny, self.nx

        # --- Nonlinear terms for the u-momentum equation (horizontal differences) ---
        self.CuP[:, :nx] = 0.25 * (self.u[:, :nx] + self.u[:, 1:nx+1] +
                                   np.abs(self.u[:, :nx]) + np.abs(self.u[:, 1:nx+1])) * self.dt / self.dx
        self.CuN[:, :nx] = 0.25 * (self.u[:, :nx] + self.u[:, 1:nx+1] -
                                   np.abs(self.u[:, :nx]) - np.abs(self.u[:, 1:nx+1])) * self.dt / self.dx
        self.CvP[:, :nx] = 0.25 * (self.v[:, :nx] + self.v[:, 1:nx+1] +
                                   np.abs(self.v[:, :nx]) + np.abs(self.v[:, 1:nx+1])) * self.dt / self.dy
        self.CvN[:, :nx] = 0.25 * (self.v[:, :nx] + self.v[:, 1:nx+1] -
                                   np.abs(self.v[:, :nx]) - np.abs(self.v[:, 1:nx+1])) * self.dt / self.dy
        self.Cu[:, :nx] = 0.5 * np.abs(self.u[:, :nx] + self.u[:, 1:nx+1]) * self.dt / self.dx
        self.Cv[:, :nx] = 0.5 * np.abs(self.v[:, :nx] + self.v[:, 1:nx+1]) * self.dt / self.dy

        # Calculate advx using B = u in the interior domain
        self.B[1:ny+1, 1:nx+1] = self.u[1:ny+1, 1:nx+1]
        self.advect()
        div1 = 0.5 * (self.u[1:ny+1, 2:nx+2] - self.u[1:ny+1, 0:nx]) / self.dx
        div2 = 0.5 * (self.v[1:ny+1, 1:nx+1] + self.v[1:ny+1, 2:nx+2] -
                      self.v[0:ny, 1:nx+1] - self.v[0:ny, 2:nx+2]) / self.dy
        div = self.dt * self.B[1:ny+1, 1:nx+1] * (div1 + div2)
        self.advx[1:ny+1, 1:nx+1] = self.BN[1:ny+1, 1:nx+1] + div

        # --- Nonlinear terms for the v-momentum equation (vertical differences) ---
        self.CuP[:ny+1, :] = 0.25 * (self.u[:ny+1, :] + self.u[1:ny+2, :] +
                                    np.abs(self.u[:ny+1, :]) + np.abs(self.u[1:ny+2, :])) * self.dt / self.dx
        self.CuN[:ny+1, :] = 0.25 * (self.u[:ny+1, :] + self.u[1:ny+2, :] -
                                    np.abs(self.u[:ny+1, :]) - np.abs(self.u[1:ny+2, :])) * self.dt / self.dx
        self.CvP[:ny+1, :] = 0.25 * (self.v[:ny+1, :] + self.v[1:ny+2, :] +
                                    np.abs(self.v[:ny+1, :]) + np.abs(self.v[1:ny+2, :])) * self.dt / self.dy
        self.CvN[:ny+1, :] = 0.25 * (self.v[:ny+1, :] + self.v[1:ny+2, :] -
                                    np.abs(self.v[:ny+1, :]) - np.abs(self.v[1:ny+2, :])) * self.dt / self.dy
        self.Cu[:ny+1, :] = 0.5 * np.abs(self.u[:ny+1, :] + self.u[1:ny+2, :]) * self.dt / self.dx
        self.Cv[:ny+1, :] = 0.5 * np.abs(self.v[:ny+1, :] + self.v[1:ny+2, :]) * self.dt / self.dy

        self.B[1:ny+1, 1:nx+1] = self.v[1:ny+1, 1:nx+1]
        self.advect()
        div1 = 0.5 * (self.u[1:ny+1, 1:nx+1] + self.u[2:ny+2, 1:nx+1] -
                      self.u[1:ny+1, 0:nx]) / self.dx
        div2 = 0.5 * (self.v[2:ny+2, 1:nx+1] - self.v[0:ny, 1:nx+1]) / self.dy
        div = self.dt * self.B[1:ny+1, 1:nx+1] * (div1 + div2)
        self.advy[1:ny+1, 1:nx+1] = self.BN[1:ny+1, 1:nx+1] + div

        # --- Step 1: Predictor for u and v without Coriolis force ---
        ustar = np.copy(self.u[1:ny+1, 1:nx+1])
        vstar = np.copy(self.v[1:ny+1, 1:nx+1])
        pgrdx = -self.dt * self.g * (self.eta[1:ny+1, 2:nx+2] - self.eta[1:ny+1, 1:nx+1]) / self.dx
        hu = 0.5 * (self.h[1:ny+1, 1:nx+1] + self.h[1:ny+1, 2:nx+2])
        uu = self.u[1:ny+1, 1:nx+1]
        vu = 0.25 * (self.v[1:ny+1, 1:nx+1] + self.v[1:ny+1, 2:nx+2] +
                     self.v[0:ny, 1:nx+1] + self.v[0:ny, 2:nx+2])
        speed = np.sqrt(uu**2 + vu**2)
        tx = np.zeros_like(hu)
        Rx = np.ones_like(hu)
        mask = hu > 0.0
        tx[mask] = self.dt * self.taux / (self.rho * hu[mask])
        Rx[mask] = 1.0 + self.dt * self.r * speed[mask] / hu[mask]
        ustar = self.u[1:ny+1, 1:nx+1] + pgrdx + tx + self.advx[1:ny+1, 1:nx+1]

        pgrdy = -self.dt * self.g * (self.eta[2:ny+2, 1:nx+1] - self.eta[1:ny+1, 1:nx+1]) / self.dy
        hv = 0.5 * (self.h[2:ny+2, 1:nx+1] + self.h[1:ny+1, 1:nx+1])
        uv = 0.25 * (self.u[1:ny+1, 1:nx+1] + self.u[2:ny+2, 1:nx+1] +
                     self.u[1:ny+1, 0:nx] + self.u[2:ny+2, 0:nx])
        vv = self.v[1:ny+1, 1:nx+1]
        speed = np.sqrt(uv**2 + vv**2)
        ty = np.zeros_like(hv)
        Ry = np.ones_like(hv)
        mask = hv > 0.0
        ty[mask] = self.dt * self.tauy / (self.rho * hv[mask])
        Ry[mask] = 1.0 + self.dt * self.r * speed[mask] / hv[mask]
        vstar = self.v[1:ny+1, 1:nx+1] + pgrdy + ty + self.advy[1:ny+1, 1:nx+1]

        # --- Step 2: Semi-implicit treatment of the Coriolis force ---
        vu_avg = 0.25 * (self.v[1:ny+1, 1:nx+1] + self.v[1:ny+1, 2:nx+2] +
                         self.v[0:ny, 1:nx+1] + self.v[0:ny, 2:nx+2])
        corx = self.dt * self.f * vu_avg
        du = (ustar - self.beta * self.u[1:ny+1, 1:nx+1] + corx) / (1.0 + self.beta) - self.u[1:ny+1, 1:nx+1]

        uv_avg = 0.25 * (self.u[1:ny+1, 1:nx+1] + self.u[2:ny+2, 1:nx+1] +
                         self.u[1:ny+1, 0:nx] + self.u[2:ny+2, 0:nx])
        cory = -self.dt * self.f * uv_avg
        dv = (vstar - self.beta * self.v[1:ny+1, 1:nx+1] + cory) / (1.0 + self.beta) - self.v[1:ny+1, 1:nx+1]

        # --- Step 3: Final predictor with flooding algorithm ---
        un_new = self.u[1:ny+1, 1:nx+1].copy()
        vn_new = self.v[1:ny+1, 1:nx+1].copy()
        # For u:
        mask_wet = self.wet[1:ny+1, 1:nx+1] == 1
        cond = (self.wet[1:ny+1, 2:nx+2] == 1) | (du > 0)
        un_new[mask_wet & cond] = (self.u[1:ny+1, 1:nx+1][mask_wet & cond] + du[mask_wet & cond]) / Rx[mask_wet & cond]
        cond2 = (~mask_wet) & (self.wet[1:ny+1, 2:nx+2] == 1) & (du < 0)
        un_new[cond2] = (self.u[1:ny+1, 1:nx+1][cond2] + du[cond2]) / Rx[cond2]
        # For v:
        cond = (self.wet[2:ny+2, 1:nx+1] == 1) | (dv > 0)
        vn_new[mask_wet & cond] = (self.v[1:ny+1, 1:nx+1][mask_wet & cond] + dv[mask_wet & cond]) / Ry[mask_wet & cond]
        cond2 = (~mask_wet) & (self.wet[2:ny+2, 1:nx+1] == 1) & (dv < 0)
        vn_new[cond2] = (self.v[1:ny+1, 1:nx+1][cond2] + dv[cond2]) / Ry[cond2]
        self.un[1:ny+1, 1:nx+1] = un_new
        self.vn[1:ny+1, 1:nx+1] = vn_new

        # Boundary conditions (mimicking Fortran conditions)
        self.un[1:ny+1, 0] = self.un[1:ny+1, 1]
        self.un[1:ny+1, nx+1] = self.un[1:ny+1, nx]
        self.vn[1:ny+1, 0] = self.vn[1:ny+1, 1]
        self.vn[1:ny+1, nx+1] = self.vn[1:ny+1, nx]
        self.un[0, 1:nx+1] = self.un[1, 1:nx+1]
        self.un[ny+1, 1:nx+1] = self.un[ny, 1:nx+1]
        self.vn[0, 1:nx+1] = self.vn[1, 1:nx+1]
        self.vn[ny, 1:nx+1] = 0.0
        self.vn[ny+1, 1:nx+1] = self.vn[ny, 1:nx+1]
        self.un[0, :] = self.un[1, :]
        self.un[ny+1, :] = self.un[ny, :]
        self.vn[0, :] = self.vn[1, :]
        self.vn[ny+1, :] = self.vn[ny, :]

        # --- Eulerian tracer predictor ---
        self.CuP[:, :] = 0.5 * (self.u + np.abs(self.u)) * self.dt / self.dx
        self.CuN[:, :] = 0.5 * (self.u - np.abs(self.u)) * self.dt / self.dx
        self.CvP[:, :] = 0.5 * (self.v + np.abs(self.v)) * self.dt / self.dy
        self.CvN[:, :] = 0.5 * (self.v - np.abs(self.v)) * self.dt / self.dy
        self.Cu[:, :] = np.abs(self.u) * self.dt / self.dx
        self.Cv[:, :] = np.abs(self.v) * self.dt / self.dy

        self.B[:, :] = self.T[:, :]
        self.advect()
        div1 = (self.u[1:ny+1, 1:nx+1] - self.u[1:ny+1, 0:nx]) / self.dx
        div2 = (self.v[1:ny+1, 1:nx+1] - self.v[0:ny, 1:nx+1]) / self.dy
        div = self.dt * self.B[1:ny+1, 1:nx+1] * (div1 + div2)
        self.TN[1:ny+1, 1:nx+1] = self.T[1:ny+1, 1:nx+1] + self.BN[1:ny+1, 1:nx+1] + div

        # Boundary conditions for TN
        self.TN[:, 0] = self.TN[:, 1]
        self.TN[:, nx+1] = self.TN[:, nx]
        self.TN[0, :] = self.TN[1, :]
        self.TN[ny+1, :] = self.TN[ny, :]

        # --- Sea level predictor ---
        self.CuP[:, :] = 0.5 * (self.un + np.abs(self.un)) * self.dt / self.dx
        self.CuN[:, :] = 0.5 * (self.un - np.abs(self.un)) * self.dt / self.dx
        self.CvP[:, :] = 0.5 * (self.vn + np.abs(self.vn)) * self.dt / self.dy
        self.CvN[:, :] = 0.5 * (self.vn - np.abs(self.vn)) * self.dt / self.dy
        self.Cu[:, :] = np.abs(self.un) * self.dt / self.dx
        self.Cv[:, :] = np.abs(self.vn) * self.dt / self.dy

        self.B[:, :] = self.h[:, :]
        self.advect()
        self.etan[1:ny+1, 1:nx+1] = self.eta[1:ny+1, 1:nx+1] + self.BN[1:ny+1, 1:nx+1]

        # Boundary conditions for etan
        self.etan[:, 0] = self.etan[:, 1]
        self.etan[:, nx+1] = self.etan[:, nx]
        self.etan[0, :] = self.etan[1, :]
        self.etan[ny+1, :] = self.etan[ny, :]

        # First-order Shapiro filter (using loops for clarity)
        for j in range(1, ny+1):
            for k in range(1, nx+1):
                if self.wet[j, k] == 1:
                    term1 = (1.0 - 0.25 * self.eps *
                             (self.wet[j, k+1] + self.wet[j, k-1] +
                              self.wet[j+1, k] + self.wet[j-1, k])) * self.etan[j, k]
                    term2 = 0.25 * self.eps * (self.wet[j, k+1] * self.etan[j, k+1] +
                                               self.wet[j, k-1] * self.etan[j, k-1])
                    term3 = 0.25 * self.eps * (self.wet[j+1, k] * self.etan[j+1, k] +
                                               self.wet[j-1, k] * self.etan[j-1, k])
                    self.eta[j, k] = term1 + term2 + term3
                else:
                    self.eta[j, k] = self.etan[j, k]

        # Boundary conditions for eta
        self.eta[:, 0] = self.eta[:, 1]
        self.eta[:, nx+1] = self.eta[:, nx]
        self.eta[0, :] = self.eta[1, :]
        self.eta[ny+1, :] = self.eta[ny, :]

        # Final update of h, wet, u, v, and T
        self.h = self.hzero + self.eta
        self.wet[self.h < self.hmin] = 0
        self.u = self.un.copy()
        self.v = self.vn.copy()
        self.T = self.TN.copy()

    def run(self, total_time, output_interval):
        """
        Runs the simulation for a given total_time (in seconds),
        producing outputs every output_interval (in seconds).
        Intermediate solutions are appended to the output files, as in the original Fortran code.
        
        Output files:
         - "eta0.dat" and "h0.dat" are written at the start.
         - "eta.dat", "h.dat", "u.dat", "v.dat", and "T.dat" are appended at each output interval.
        """
        ntot = int(total_time / self.dt)
        nout = int(output_interval / self.dt)

        # Write initial output files (interior domain only)
        np.savetxt("eta0.dat", self.eta[1:self.ny+1, 1:self.nx+1], fmt="%12.6f")
        np.savetxt("h0.dat", self.hzero[1:self.ny+1, 1:self.nx+1], fmt="%12.6f")

        # Open output files in append mode
        eta_file = open("eta.dat", "a")
        h_file = open("h.dat", "a")
        u_file = open("u.dat", "a")
        v_file = open("v.dat", "a")
        T_file = open("T.dat", "a")

        for n in range(1, ntot + 1):
            current_time = n * self.dt
            print("time (days):", current_time / (24.0 * 3600.0))
            self.taux = 0.0
            self.tauy = 0.0

            # Sea level forcing example: set eta[6,2] = sin(time/(2*3600)*2*PI)
            ad = np.sin(current_time / (2 * 3600.0) * 2 * self.PI)
            self.eta[6, 2] = ad

            self.dyn()

            if n % nout == 0:
                print("Data output at time =", current_time / (24.0 * 3600.0))
                # Append the current interior domain to each file
                np.savetxt(eta_file, self.eta[1:self.ny+1, 1:self.nx+1], fmt="%12.6f")
                np.savetxt(h_file, self.h[1:self.ny+1, 1:self.nx+1], fmt="%12.6f")
                np.savetxt(u_file, self.u[1:self.ny+1, 1:self.nx+1], fmt="%12.6f")
                np.savetxt(v_file, self.v[1:self.ny+1, 1:self.nx+1], fmt="%12.6f")
                np.savetxt(T_file, self.T[1:self.ny+1, 1:self.nx+1], fmt="%12.6f")

        # Close output files
        eta_file.close()
        h_file.close()
        u_file.close()
        v_file.close()
        T_file.close()

        print("Simulation completed.")

if __name__ == "__main__":
    model = KelvinModel()
    # Example: simulate 12 hours (43,200 s) with outputs every 10 minutes (600 s)
    model.run(43200, 600)



