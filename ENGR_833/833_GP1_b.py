import numpy as np
import matplotlib.pyplot as plt

# Given
zeta = 0.05
td = 0.1
alpha = 1/5
F0 = 1.0   # unit force
m = 1.0    # unit mass (OK for spectrum)

# Trial periods (covers td/Tn up to 3 and down near 0)
Tn = np.logspace(np.log10(td/3), np.log10(3.0), 160)

beta = 1/4
gamma = 1/2

Rd = np.zeros_like(Tn)
xratio = td / Tn

def pulse(t):
    if 0.0 <= t <= td:
        return F0 * (1.0 - (1.0 + alpha) * t / td)
    return 0.0

for i, T in enumerate(Tn):
    wn = 2*np.pi / T
    k = m * wn**2
    c = 2*zeta*m*wn

    # time step (small enough for accuracy)
    dt = min(T/200, td/200)
    t_end = td + 8*T
    n = int(np.ceil(t_end/dt)) + 1
    t = np.linspace(0, (n-1)*dt, n)

    # arrays
    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)

    # initial acceleration from equilibrium
    p0 = pulse(0.0)
    a[0] = (p0 - c*v[0] - k*u[0]) / m

    # Newmark constants
    a0 = 1.0/(beta*dt**2)
    a1 = gamma/(beta*dt)
    a2 = 1.0/(beta*dt)
    a3 = 1.0/(2*beta) - 1.0
    a4 = gamma/beta - 1.0
    a5 = dt*(gamma/(2*beta) - 1.0)

    k_eff = k + a0*m + a1*c

    # time stepping
    for j in range(n-1):
        pj1 = pulse(t[j+1])

        p_eff = (pj1
                 + m*(a0*u[j] + a2*v[j] + a3*a[j])
                 + c*(a1*u[j] + a4*v[j] + a5*a[j]))

        u[j+1] = p_eff / k_eff
        a[j+1] = a0*(u[j+1] - u[j]) - a2*v[j] - a3*a[j]
        v[j+1] = v[j] + dt*((1-gamma)*a[j] + gamma*a[j+1])

    umax = np.max(np.abs(u))
    Rd[i] = k * umax   # since ust = F0/k and F0=1 => Rd = k*umax

# Plot like textbook axis
plt.figure()
plt.plot(xratio, Rd, linewidth=2)
plt.xlim(0, 3)
plt.ylim(0, max(2.1, 1.05*np.max(Rd)))
plt.xlabel(r"$t_d/T_n$")
plt.ylabel(r"$R_d = u_{\max}/u_{st}$")
plt.grid(True)
plt.show()