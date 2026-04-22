import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Newmark-beta Method: Fixed 3-DOF example from the PDF
# ============================================================

# -----------------------------
# Time stepping
# -----------------------------
dt = 0.01
N = 300
t = np.arange(N + 1) * dt

# -----------------------------
# Load definition from the PDF
# -----------------------------
Tload = 0.3
omega_load = 2.0 * np.pi / Tload
ft = np.sin(omega_load * t)

# p(t) = [1, 2, 3]^T * sin(omega_load * t)
p = np.vstack([
    1.0 * ft,
    2.0 * ft,
    3.0 * ft
])

# Load increments Δp_j = p_{j+1} - p_j
dp = p[:, 1:] - p[:, :-1]

# -----------------------------
# System properties from the PDF
# -----------------------------
ma = 0.259
ka = 168.0

M = ma * np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 0.5]
])

K = (ka / 9.0) * np.array([
    [16.0, -7.0,  0.0],
    [-7.0, 10.0, -3.0],
    [ 0.0, -3.0,  3.0]
])

alpha_d = 0.5505
beta_d_rayleigh = 0.001179
C = alpha_d * M + beta_d_rayleigh * K

# -----------------------------
# Newmark parameters from the PDF
# -----------------------------
# Change these two if you want different Newmark parameters.
gamma = 1.0 / 2.0
beta = 1.0 / 6.0

# Effective matrices used in the incremental formulation
Khat = K + M / (beta * dt**2) + gamma * C / (beta * dt)
A = M / (beta * dt) + gamma * C / beta
B = M / (2.0 * beta) + dt * C * (gamma / (2.0 * beta) - 1.0)

# -----------------------------
# Initial conditions from the PDF
# -----------------------------
n = 3
u = np.zeros((n, N + 1))
ud = np.zeros((n, N + 1))
udd = np.zeros((n, N + 1))

# Since u0 = 0, ud0 = 0, and p(:,0)=0, this gives udd(:,0)=0
udd[:, 0] = np.linalg.solve(M, p[:, 0] - C @ ud[:, 0] - K @ u[:, 0])

# -----------------------------
# Newmark step-by-step solution
# -----------------------------
for j in range(N):
    dpj = dp[:, j]
    udj = ud[:, j]
    uddj = udd[:, j]

    du = np.linalg.solve(Khat, dpj + A @ udj + B @ uddj)

    dud = (
        (gamma / (beta * dt)) * du
        - (gamma / beta) * udj
        - dt * (gamma / (2.0 * beta) - 1.0) * uddj
    )

    dudd = (
        (1.0 / (beta * dt**2)) * du
        - (1.0 / (beta * dt)) * udj
        - (1.0 / (2.0 * beta)) * uddj
    )

    u[:, j + 1] = u[:, j] + du
    ud[:, j + 1] = ud[:, j] + dud
    udd[:, j + 1] = udd[:, j] + dudd

# ============================================================
# Plot 1: Load histories p0, p1, p2
# ============================================================
plt.figure(figsize=(9, 5))
plt.plot(t, p[0, :], label="p0")
plt.plot(t, p[1, :], label="p1")
plt.plot(t, p[2, :], label="p2")
plt.xlabel("t")
plt.ylabel("Load")
plt.title("Load Histories")
plt.legend()
plt.grid(True)
plt.tight_layout()

# ============================================================
# Plot 2: Response histories u0 and u1
# ============================================================
plt.figure(figsize=(9, 5))
plt.plot(t, u[0, :], label="u0")
plt.plot(t, u[1, :], label="u1")
plt.xlabel("t")
plt.ylabel("Displacement")
plt.title("Displacement Response: u0 and u1")
plt.legend()
plt.grid(True)
plt.tight_layout()

# ============================================================
# Plot 3: Response history u2
# ============================================================
plt.figure(figsize=(9, 5))
plt.plot(t, u[2, :], label="u2")
plt.xlabel("t")
plt.ylabel("Displacement")
plt.title("Displacement Response: u2")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()