import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# GIVEN / ASSUMED
# -----------------------------
zeta = 0.05
td = 0.1
alpha = 1/5

F0 = 1.0   # unit force (ok because spectrum is normalized)
m  = 1.0   # per your instruction

# Newmark constant average acceleration
beta = 1/4
gamma = 1/2

# -----------------------------
# Force pulse P(t)
# Linear from +F0 at t=0 to -alpha*F0 at t=td, then zero
# -----------------------------
def P(t):
    if 0.0 <= t <= td:
        return F0 * (1.0 - (1.0 + alpha) * t / td)
    return 0.0

# -----------------------------
# One SDOF simulation for a given Tn
# Returns R_MBA = Mmax/Mst (normalized)
# -----------------------------
def spectrum_value(Tn):
    wn = 2*np.pi / Tn
    k  = m * wn**2
    c  = 2*zeta*m*wn

    # time step + duration (capture free vibration peaks)
    dt = min(Tn/200, td/200)
    t_end = td + 10*Tn
    n = int(np.ceil(t_end/dt)) + 1
    t = np.linspace(0, (n-1)*dt, n)

    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)

    # initial acceleration from equilibrium
    a[0] = (P(0.0) - c*v[0] - k*u[0]) / m

    # Newmark constants
    a0 = 1.0/(beta*dt**2)
    a1 = gamma/(beta*dt)
    a2 = 1.0/(beta*dt)
    a3 = 1.0/(2*beta) - 1.0
    a4 = gamma/beta - 1.0
    a5 = dt*(gamma/(2*beta) - 1.0)

    k_eff = k + a0*m + a1*c

    for i in range(n-1):
        p_next = P(t[i+1])

        p_eff = (p_next
                 + m*(a0*u[i] + a2*v[i] + a3*a[i])
                 + c*(a1*u[i] + a4*v[i] + a5*a[i]))

        u[i+1] = p_eff / k_eff
        a[i+1] = a0*(u[i+1] - u[i]) - a2*v[i] - a3*a[i]
        v[i+1] = v[i] + dt*((1-gamma)*a[i] + gamma*a[i+1])

    umax = np.max(np.abs(u))

    # Normalized moment spectrum:
    # R_MBA = Mmax/Mst = xmax/xst = (k*umax/F0)
    R_MBA = (k * umax) / F0
    return R_MBA

# -----------------------------
# Build spectrum vs td/Tn (like Fig 4.7.3)
# -----------------------------
Tn_vals = np.logspace(np.log10(td/3), np.log10(3.0), 180)  # td/Tn ~ 3 down to ~0.033
xratio  = td / Tn_vals

R_MBA_vals = np.array([spectrum_value(Tn) for Tn in Tn_vals])

# -----------------------------
# Plot (required format)
# -----------------------------
plt.figure()
plt.plot(xratio, R_MBA_vals, linewidth=2)
plt.xlim(0, 3)
plt.ylim(0, max(2.1, 1.05*np.max(R_MBA_vals)))
plt.xlabel(r"$t_d/T_n$")
plt.ylabel(r"$R_{MBA} = M_{BA,\max}/M_{BA,st}$")
plt.grid(True)
plt.show()