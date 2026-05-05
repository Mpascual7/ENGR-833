import numpy as np
import matplotlib.pyplot as plt

# Updated mode shapes
phi1 = np.array([0.1583, 0.4433, 0.7443, 1.0000])
phi2 = np.array([-0.3607, -0.6473, -0.2096, 1.0000])
phi3 = np.array([1.0411, 0.5965, -1.2484, 1.0000])
phi4 = np.array([-6.0138, 4.8651, -2.2899, 1.0000])

modes = [phi1, phi2, phi3, phi4]

# Updated frequencies
freqs = [0.1229, 0.3651, 0.6999, 1.2048]

floors = np.array([0, 1, 2, 3, 4])  # include base

plt.figure(figsize=(8, 6))

for i, phi in enumerate(modes):
    plt.subplot(2, 2, i + 1)

    # Include base displacement = 0
    mode_with_base = np.insert(phi, 0, 0.0)

    plt.plot(mode_with_base, floors, marker='o', linewidth=2)
    plt.axvline(0, linewidth=0.8)

    plt.xlabel("Displacement")
    plt.ylabel("Floor")
    plt.title(f"Mode {i+1}, f={freqs[i]:.4f} Hz", fontweight="bold")
    plt.grid(True)
    plt.yticks(floors)

plt.tight_layout()
plt.show()