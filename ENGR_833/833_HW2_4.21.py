import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Problem 4.21: triangular pulse response of undamped SDOF
#
# Normalized variables:
#   x = t / Tn
#   r = td / Tn
#
# Response:
#   u/(ust)_0
# ------------------------------------------------------------

pi = np.pi


def response_normalized(x, r):
    """
    Piecewise normalized response u/(ust)_0
    x = t/Tn
    r = td/Tn
    """
    x = np.asarray(x)
    u = np.zeros_like(x, dtype=float)

    # Forced vibration phase: 0 <= x <= r
    mask_forced = (x <= r)
    xf = x[mask_forced]
    u[mask_forced] = xf / r - (1 / (2 * pi)) * (1 / r) * np.sin(2 * pi * xf)

    # Free vibration phase: x >= r
    mask_free = (x > r)
    xf = x[mask_free]
    u[mask_free] = (
        np.cos(2 * pi * (xf - r))
        + (1 / (2 * pi)) * (1 / r) * np.sin(2 * pi * (xf - r))
        - (1 / (2 * pi)) * (1 / r) * np.sin(2 * pi * xf)
    )

    return u


def static_response_normalized(x, r):
    """
    Normalized static response for the triangular pulse:
      u_st/(ust)_0 = t/td  for 0 <= t <= td
                    = 0    for t >= td
    """
    x = np.asarray(x)
    us = np.zeros_like(x, dtype=float)
    mask = (x <= r)
    us[mask] = x[mask] / r
    return us


def forced_max(r):
    """
    Eq. (g): maximum during forced vibration phase
    """
    return 1 - (1 / (2 * pi * r)) * np.sin(2 * pi * r)


def free_max(r):
    """
    Eq. (i): maximum during free vibration phase
    """
    term1 = 1 - (1 / (2 * pi * r)) * np.sin(2 * pi * r)
    term2 = (1 / (pi**2 * r**2)) * (np.sin(pi * r) ** 4)
    return np.sqrt(term1**2 + term2)


def overall_max(r):
    """
    Overall maximum = larger of forced-phase and free-phase maxima
    """
    return np.maximum(forced_max(r), free_max(r))


# ------------------------------------------------------------
# Part (b): response plots for td/Tn = 1/2 and 2
# ------------------------------------------------------------
r_values = [0.5, 2.0]
x = np.linspace(0, 4, 2000)

fig1, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

for ax, r in zip(axes, r_values):
    u = response_normalized(x, r)
    us = static_response_normalized(x, r)

    ax.plot(x, u, linewidth=2, label=r'$u(t)/(u_{st})_0$')
    ax.plot(x, us, '--', linewidth=1.5, label='static response')

    ax.axhline(0, linewidth=1)
    ax.set_ylabel(r'$u(t)/(u_{st})_0$')
    ax.set_title(rf'Response for $t_d/T_n = {r}$')
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel(r'$t/T_n$')
fig1.suptitle('Problem 4.21: Normalized Response', fontsize=14)
fig1.tight_layout(rect=[0, 0, 1, 0.96])

# ------------------------------------------------------------
# Part (c): maximum response curves
# ------------------------------------------------------------
r = np.linspace(0.05, 4.0, 2000)

R_forced = forced_max(r)
R_free = free_max(r)
R_overall = overall_max(r)

# Forced response maximum only
fig2 = plt.figure(figsize=(8, 4.5))
plt.plot(r, R_forced, '--', linewidth=2, label='Forced response maximum')
plt.xlabel(r'$t_d/T_n$')
plt.ylabel(r'$R_d$')
plt.title('Maximum During Forced Vibration Phase')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Forced + free response maxima together
fig3 = plt.figure(figsize=(8, 4.5))
plt.plot(r, R_forced, '--', linewidth=2, label='Forced response')
plt.plot(r, R_free, linewidth=2, label='Free response')
plt.xlabel(r'$t_d/T_n$')
plt.ylabel(r'$R_d$')
plt.title('Forced-Phase and Free-Phase Maximum Response')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Overall maximum (shock spectrum)
fig4 = plt.figure(figsize=(8, 4.5))
plt.plot(r, R_overall, linewidth=2, label='Overall maximum')
plt.xlabel(r'$t_d/T_n$')
plt.ylabel(r'$R_d$')
plt.title('Problem 4.21 Shock Spectrum')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()