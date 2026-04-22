import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Problem 6.8
# Undamped SDOF response to a full-cycle sine pulse ground motion
# ============================================================


# -----------------------------
# Core response functions
# -----------------------------
def normalized_response(tau, alpha):
    tau = np.asarray(tau)
    y = np.zeros_like(tau, dtype=float)

    forced = tau <= alpha
    free = tau >= alpha

    if np.isclose(alpha, 1.0):
        # Eq. (e) and (g)
        y[forced] = -0.5 * (
            np.sin(2 * np.pi * tau[forced])
            - 2 * np.pi * tau[forced] * np.cos(2 * np.pi * tau[forced])
        )
        y[free] = np.pi * np.cos(2 * np.pi * tau[free])

    else:
        # Eq. (b) and (d)
        r = 1.0 / alpha

        y[forced] = -(1.0 / (1.0 - r**2)) * (
            np.sin(2 * np.pi * tau[forced] / alpha)
            - r * np.sin(2 * np.pi * tau[forced])
        )

        amp = 2.0 * alpha * np.sin(np.pi * alpha) / (alpha**2 - 1.0)
        y[free] = amp * np.cos(2 * np.pi * (tau[free] - alpha / 2.0))

    return y


def normalized_static_response(tau, alpha):
    tau = np.asarray(tau)
    yst = np.zeros_like(tau, dtype=float)
    mask = tau <= alpha
    yst[mask] = -np.sin(2 * np.pi * tau[mask] / alpha)
    return yst


# -----------------------------
# Spectral quantities
# -----------------------------
def forced_phase_max_min(alpha, npts=20000):
    tau = np.linspace(0.0, alpha, npts)
    y = normalized_response(tau, alpha)
    return np.max(y), -np.min(y)


def forced_response_spectrum(alpha):
    if np.isclose(alpha, 1.0):
        return np.pi

    lmax = int(np.ceil(alpha))
    vals = []

    for l in range(1, lmax + 1):
        tau_l = l * alpha / (1.0 + alpha)
        if tau_l < alpha - 1e-12:
            val = abs(
                (1.0 / (1.0 - (1.0 / alpha) ** 2))
                * (
                    np.sin(2 * np.pi * l / (1.0 + alpha))
                    - (1.0 / alpha)
                    * np.sin(2 * np.pi * l * alpha / (1.0 + alpha))
                )
            )
            vals.append(val)

    return max(vals) if vals else 0.0


def free_response_spectrum(alpha):
    if np.isclose(alpha, 1.0):
        return np.pi

    return abs(2.0 * alpha * np.sin(np.pi * alpha) / (1.0 - alpha**2))


def overall_max_spectrum(alpha):
    return max(forced_response_spectrum(alpha), free_response_spectrum(alpha))


# -----------------------------
# Plotting: Fig. 6.8a
# -----------------------------
def plot_fig_P68a():
    alphas = [1/8, 1/4, 1/2, 3/4, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5]

    fig, axes = plt.subplots(6, 2, figsize=(12, 18))
    axes = axes.flatten()

    for ax, alpha in zip(axes, alphas):
        tau_max = 2.0 * alpha
        tau = np.linspace(0.0, tau_max, 3000)

        y = normalized_response(tau, alpha)
        yst = normalized_static_response(tau, alpha)

        ax.plot(tau, yst, "--", lw=2.0, label="Static")
        ax.plot(tau, y, "-", lw=2.5, label="Dynamic")

        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_xlim(0.0, tau_max)
        ax.set_ylim(-3.2, 3.2)
        ax.grid(True, alpha=0.25)

        ax.set_title(rf"$t_d/T_n = {alpha:g}$", fontsize=12)
        ax.set_xlabel(r"$t/T_n$")
        ax.set_ylabel(r"$\omega_n^2 u(t)/\ddot{u}_{g0}$")

    fig.suptitle("Fig. 6.8 Response Histories", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])


# -----------------------------
# Plotting: Fig. 6.8b
# -----------------------------
def plot_fig_P68b():
    alpha_vals = np.linspace(0.02, 6.0, 1200)

    umax_vals = []
    minus_umin_vals = []

    for alpha in alpha_vals:
        umax, minus_umin = forced_phase_max_min(alpha)
        umax_vals.append(umax)
        minus_umin_vals.append(minus_umin)

    plt.figure(figsize=(9, 5.5))
    plt.plot(alpha_vals, umax_vals, "--", lw=2.5, label=r"$u_{\max}/(u_{st})_0$ for $t \leq t_d$")
    plt.plot(alpha_vals, minus_umin_vals, "-", lw=2.5, label=r"$-u_{\min}/(u_{st})_0$ for $t \leq t_d$")

    plt.xlim(0, 6)
    plt.ylim(0, 3.6)
    plt.grid(True, alpha=0.25)
    plt.xlabel(r"$t_d/T_n$")
    plt.ylabel(r"$A/\ddot{u}_{g0} = |u_0/u_{st0}|$")
    plt.title("Fig. 6.8 Max/Min During Forced Phase")
    plt.legend(frameon=False)
    plt.tight_layout()


# -----------------------------
# Plotting: Fig. 6.8c
# -----------------------------
def plot_fig_P68c():
    alpha_vals = np.linspace(0.02, 6.0, 2000)

    forced_vals = np.array([forced_response_spectrum(a) for a in alpha_vals])
    free_vals = np.array([free_response_spectrum(a) for a in alpha_vals])

    plt.figure(figsize=(9, 5.5))
    plt.plot(alpha_vals, forced_vals, "--", lw=2.5, label="Forced Response")
    plt.plot(alpha_vals, free_vals, "-.", lw=2.5, label="Free Response")

    plt.xlim(0, 6)
    plt.ylim(0, 3.6)
    plt.grid(True, alpha=0.25)
    plt.xlabel(r"$t_d/T_n$")
    plt.ylabel(r"$A/\ddot{u}_{g0}$")
    plt.title("Fig. 6.8 Forced and Free Response Spectra")
    plt.legend(frameon=False)
    plt.tight_layout()


# -----------------------------
# Plotting: Fig. 6.8d
# -----------------------------
def plot_fig_P68d():
    alpha_vals = np.linspace(0.02, 6.0, 2000)
    overall_vals = np.array([overall_max_spectrum(a) for a in alpha_vals])

    plt.figure(figsize=(9, 5.5))
    plt.plot(alpha_vals, overall_vals, "-", lw=2.5)

    plt.xlim(0, 6)
    plt.ylim(0, 3.6)
    plt.grid(True, alpha=0.25)
    plt.xlabel(r"$t_d/T_n$")
    plt.ylabel(r"$A/\ddot{u}_{g0} = \ddot{u}_0/\ddot{u}_{g0}$")
    plt.title("Fig. 6.8 Overall Maximum Response Spectrum")
    plt.tight_layout()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "lines.solid_capstyle": "round",
    })

    plot_fig_P68a()
    plot_fig_P68b()
    plot_fig_P68c()
    plot_fig_P68d()

    plt.show()