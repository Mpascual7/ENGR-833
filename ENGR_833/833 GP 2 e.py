import numpy as np
from scipy.linalg import solve

K = np.array([
    [38.233, -20.976,   5.267, -0.581],
    [-20.976, 24.759, -13.045,  2.320],
    [5.267,  -13.045,  13.669, -4.782],
    [-0.581,   2.320,  -4.782,  2.921]
], dtype=float)

# Correct condensed mass matrix
# Roof mass is m/2, so last DOF has mass coefficient 0.5
M = np.diag([1.0, 1.0, 1.0, 0.5])

def inverse_iteration(K, M, shift, tol=1e-10, max_iter=1000):
    n = K.shape[0]

    # Initial guess
    x = np.ones(n)

    # M-normalize initial vector
    x = x / np.sqrt(x.T @ M @ x)

    # Shifted matrix
    A = K - shift * M

    for i in range(max_iter):
        # Solve (K - shift M)y = Mx
        y = solve(A, M @ x)

        # M-normalize
        y = y / np.sqrt(y.T @ M @ y)

        # Rayleigh quotient for generalized eigenproblem
        lam = (y.T @ K @ y) / (y.T @ M @ y)

        # Check convergence
        if np.linalg.norm(y - x) < tol or np.linalg.norm(y + x) < tol:
            return lam, y, i + 1

        x = y

    raise RuntimeError("Inverse iteration did not converge")


# Shifts near expected eigenvalues from part (d)
# lambda = omega^2
shifts = [
    0.7717**2,
    2.2939**2,
    4.3978**2,
    7.5700**2
]

for mode, shift in enumerate(shifts, start=1):
    lam, phi, iters = inverse_iteration(K, M, shift)

    omega = np.sqrt(lam)
    freq = omega / (2 * np.pi)

    # Normalize mode shape so roof displacement = 1
    phi = phi / phi[-1]

    print(f"Mode {mode}")
    print(f"lambda = {lam:.6f}")
    print(f"omega = {omega:.4f} * sqrt(EI/(m h^3)) rad/s")
    print(f"f = {freq:.4f} * sqrt(EI/(m h^3)) Hz")
    print(f"mode shape = {phi}")
    print(f"iterations = {iters}")
    print()