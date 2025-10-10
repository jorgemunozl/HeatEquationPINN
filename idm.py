#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
L = 1.0          # length of the rod
nx = 51          # number of spatial points
dx = L / (nx - 1)
alpha = 1e-3     # thermal diffusivity
dt = 0.4 * dx**2 / alpha   # time step (stability condition)
nt = 5000        # number of time steps

# --- Discretized grid ---
x = np.linspace(0, L, nx)

# --- Initial condition ---
# for instance, a Gaussian bump in the middle
u = np.exp(-((x - 0.5*L)**2) / 0.01)

# --- Time evolution ---
for n in range(nt):
    u_new = u.copy()

    # interior points
    u_new[1:-1] = u[1:-1] + alpha * dt / dx**2 * (u[2:] - 2*u[1:-1] + u[:-2])
    # Neumann boundary conditions (du/dx = 0)
    u_new[0]  = u_new[1]      # left boundary
    u_new[-1] = u_new[-2]     # right boundary
    u = u_new

# --- Plot result ---
plt.figure(figsize=(7,4))
plt.plot(x, u, label='t final', linewidth=2)
plt.title("1D Heat Equation (Neumann BC = 0)")
plt.xlabel("x")
plt.ylabel("u(x, t)")
plt.grid(True)
plt.legend()
plt.show()
