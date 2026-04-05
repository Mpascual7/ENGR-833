import numpy as np
import matplotlib.pyplot as plt

pi = np.pi

# ------------------------------------------------------------
# Normalized response u/(ust)_0 for td/Tn = r
# ------------------------------------------------------------
def response_normalized(x, r):
    u = np.zeros_like(x)

    # Interval 1: 0 <= x <= r/2
    m1 = x <= r / 2
    u[m1] = 1 - np.cos(2 * pi * x[m1])

    # Interval 2: r/2 <= x <= r
    m2 = (x > r / 2) & (x <= r)
    xm = x[m2]
    u[m2] = (
        (2 - np.cos(pi * r)) * np.cos(2 * pi * (xm - r / 2))
        + np.sin(pi * r) * np.sin(2 * pi * (xm - r / 2))
        - 1
    )

    # Interval 3: x >= r
    m3 = x > r
    xm = x[m3]
    u[m3] = (
        (2 * np.cos(pi * r) - np.cos(2 * pi * r) - 1)
        * np.cos(2 * pi * (xm - r))
        + (np.sin(2 * pi * r) - 2 * np.sin(pi * r))
        * np.sin(2 * pi * (xm - r))
    )

    return u


# ------------------------------------------------------------
# Shock spectrum Rd
# ------------------------------------------------------------
def Rd(r):
    return 4 * np.sin(pi * r / 2) ** 2


# ------------------------------------------------------------
# Plot 1: Time response for td/Tn = 1
# ------------------------------------------------------------
r = 1.0
x = np.linspace(0, 4, 2000)

u = response_normalized(x, r)

plt.figure()
plt.plot(x, u)
plt.axhline(0)
plt.grid(True)
plt.xlabel(r'$t/T_n$')
plt.ylabel(r'$u(t)/(u_{st})_0$')
plt.title(r'Problem 4.25: Response for $t_d/T_n = 1$')
plt.show()


# ------------------------------------------------------------
# Plot 2: Shock spectrum
# ------------------------------------------------------------
r_vals = np.linspace(0, 4, 2000)
Rd_vals = Rd(r_vals)

plt.figure()
plt.plot(r_vals, Rd_vals)
plt.grid(True)
plt.xlabel(r'$t_d/T_n$')
plt.ylabel(r'$R_d$')
plt.title('Shock Spectrum')
plt.show()