import numpy as np
import matplotlib.pyplot as plt

# Values of a/wn required in the problem
ratios = [0.01, 0.1, 1.0]

# Time axis in terms of t/Tn
tau = np.linspace(0, 4, 1000)   # tau = t/Tn

# Create separate figures like the book
fig1, axes1 = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
fig2, axes2 = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

for i, r in enumerate(ratios):
    # r = a/wn
    wn_t = 2 * np.pi * tau         # because wn * Tn = 2*pi
    a_t = r * wn_t                 # since a*t = (a/wn)*(wn*t)

    # Part (b) load plot: p(t)/p0 = exp(-a t)
    p_norm = np.exp(-a_t)

    # Response plot:
    # u/(ust)_0 = 1/(1+r^2) * [ r*sin(wn*t) - cos(wn*t) + exp(-a*t) ]
    u_norm = (1 / (1 + r**2)) * (
        r * np.sin(wn_t) - np.cos(wn_t) + np.exp(-a_t)
    )

    # ---- Load plots ----
    axes1[i].plot(tau, p_norm, linewidth=2)
    axes1[i].set_ylabel(r'$p(t)/p_0$')
    axes1[i].set_title(rf'$a/\omega_n = {r}$')
    axes1[i].grid(True)

    # ---- Response plots ----
    axes2[i].plot(tau, u_norm, linewidth=2)
    axes2[i].axhline(0, linewidth=1)
    axes2[i].set_ylabel(r'$u(t)/(u_{st})_0$')
    axes2[i].set_title(rf'$a/\omega_n = {r}$')
    axes2[i].grid(True)

# x-label only on bottom plots
axes1[-1].set_xlabel(r'$t/T_n$')
axes2[-1].set_xlabel(r'$t/T_n$')

fig1.suptitle('Normalized Load Decay: $p(t)/p_0 = e^{-at}$', fontsize=14)
fig2.suptitle('Normalized Displacement Response', fontsize=14)

fig1.tight_layout(rect=[0, 0, 1, 0.97])
fig2.tight_layout(rect=[0, 0, 1, 0.97])

plt.show()