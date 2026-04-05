import numpy as np
import matplotlib.pyplot as plt

# Time axis in normalized form: tau = t/Tn
tau = np.linspace(0, 4, 1000)

# Normalized displacement: u'/(vTn)
u_norm = tau - (1 / (2 * np.pi)) * np.sin(2 * np.pi * tau)

# Plot
plt.figure(figsize=(7, 4))

plt.plot(tau, u_norm, linewidth=2)
plt.xlabel(r'$t/T_n$')
plt.ylabel(r"$u'(t)/(vT_n)$")
plt.title('Problem 4.8 Response')
plt.grid(True)

plt.xlim(0, 4)
plt.ylim(0, 4)

plt.show()