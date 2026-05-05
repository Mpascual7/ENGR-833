import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# -----------------------------
# User settings
# -----------------------------
AT2_FILE = "RSN6_IMPVALL.I_I-ELC180.AT2"

DAMPING_RATIOS = [0.02, 0.05, 0.10, 0.14]
PERIODS = np.linspace(0.3, 5.0, 120)

G = 981.0  # cm/s^2, because PEER AT2 files are usually in units of g


# -----------------------------
# Read PEER .AT2 file
# -----------------------------
def read_at2(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    dt = None
    accel_values = []

    for line in lines:
        if "DT=" in line.upper() or "DT =" in line.upper():
            match = re.search(r"DT\s*=\s*([0-9.Ee+-]+)", line, re.IGNORECASE)
            if match:
                dt = float(match.group(1))

        # Try reading numeric acceleration values
        parts = line.replace(",", " ").split()
        for p in parts:
            try:
                accel_values.append(float(p))
            except ValueError:
                pass

    if dt is None:
        raise ValueError("Could not find DT in AT2 file.")

    # Remove header numbers accidentally captured before acceleration block
    # For PEER files, acceleration data usually begins after the line containing DT.
    start_index = 0
    for i, line in enumerate(lines):
        if "DT" in line.upper():
            start_index = i + 1
            break

    accel_values = []
    for line in lines[start_index:]:
        parts = line.replace(",", " ").split()
        for p in parts:
            try:
                accel_values.append(float(p))
            except ValueError:
                pass

    accel_g = np.array(accel_values)
    time = np.arange(len(accel_g)) * dt

    return time, accel_g, dt


# -----------------------------
# Linear SDOF response solver
# Equation:
# x'' + 2*zeta*w*x' + w^2*x = -ag(t)
# -----------------------------
def compute_sdof_response(time, ag_cm_s2, period, zeta):
    omega = 2 * np.pi / period

    def ag_interp(t):
        return np.interp(t, time, ag_cm_s2)

    def equation(t, y):
        x, v = y
        a_ground = ag_interp(t)
        dxdt = v
        dvdt = -2 * zeta * omega * v - omega**2 * x - a_ground
        return [dxdt, dvdt]

    y0 = [0.0, 0.0]

    sol = solve_ivp(
        equation,
        (time[0], time[-1]),
        y0,
        t_eval=time,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )

    x = sol.y[0]
    sd = np.max(np.abs(x))

    psv = omega * sd
    psa = omega**2 * sd

    return sd, psv, psa


# -----------------------------
# Main workflow
# -----------------------------
def main():
    time, accel_g, dt = read_at2(AT2_FILE)

    # Convert ground acceleration from g to cm/s^2
    ag_cm_s2 = accel_g * G

    pga = np.max(np.abs(ag_cm_s2))

    print(f"Loaded ground motion: {AT2_FILE}")
    print(f"Number of points: {len(time)}")
    print(f"Time step DT: {dt} sec")
    print(f"PGA: {pga:.2f} cm/s^2 = {pga / G:.3f} g")

    results = []

    for zeta in DAMPING_RATIOS:
        print(f"Computing spectra for damping = {zeta*100:.0f}%")

        for T in PERIODS:
            sd, psv, psa = compute_sdof_response(time, ag_cm_s2, T, zeta)

            results.append(
                {
                    "Period_sec": T,
                    "Damping_ratio": zeta,
                    "Damping_percent": zeta * 100,
                    "Sd_cm": sd,
                    "PSV_cm_per_sec": psv,
                    "PSA_cm_per_sec2": psa,
                    "PSA_g": psa / G,
                    "PSV_over_PGA_sec": psv / pga,
                }
            )

    df = pd.DataFrame(results)
    df.to_csv("elcentro_response_spectra.csv", index=False)

    # -----------------------------
    # Plot Spectral Displacement
    # -----------------------------
    plt.figure(figsize=(8, 6))
    for zeta in DAMPING_RATIOS:
        subset = df[df["Damping_ratio"] == zeta]
        plt.plot(subset["Period_sec"], subset["Sd_cm"], label=f"{zeta*100:.0f}% damping")

    plt.xlabel("Period, T (sec)")
    plt.ylabel("Spectral Displacement, Sd (cm)")
    plt.title("El Centro Spectral Displacement")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig("spectral_displacement_Sd.png", dpi=300, bbox_inches="tight")
    plt.show()

    # -----------------------------
    # Plot Pseudo Velocity
    # -----------------------------
    plt.figure(figsize=(8, 6))
    for zeta in DAMPING_RATIOS:
        subset = df[df["Damping_ratio"] == zeta]
        plt.plot(subset["Period_sec"], subset["PSV_cm_per_sec"], label=f"{zeta*100:.0f}% damping")

    plt.xlabel("Period, T (sec)")
    plt.ylabel("Pseudo-Velocity, PSV (cm/sec)")
    plt.title("El Centro Pseudo-Velocity Response Spectrum")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig("pseudo_velocity_PSV.png", dpi=300, bbox_inches="tight")
    plt.show()

    # -----------------------------
    # Plot Pseudo Acceleration
    # -----------------------------
    plt.figure(figsize=(8, 6))
    for zeta in DAMPING_RATIOS:
        subset = df[df["Damping_ratio"] == zeta]
        plt.plot(subset["Period_sec"], subset["PSA_g"], label=f"{zeta*100:.0f}% damping")

    plt.xlabel("Period, T (sec)")
    plt.ylabel("Pseudo-Acceleration, PSA (g)")
    plt.title("El Centro Pseudo-Acceleration Response Spectrum")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig("pseudo_acceleration_PSA.png", dpi=300, bbox_inches="tight")
    plt.show()

    # -----------------------------
    # Paper-style normalized PSV plot
    # -----------------------------
    plt.figure(figsize=(8, 6))
    for zeta in DAMPING_RATIOS:
        subset = df[df["Damping_ratio"] == zeta]
        plt.plot(
            subset["Period_sec"],
            subset["PSV_over_PGA_sec"],
            label=f"{zeta*100:.0f}% damping",
        )

    plt.xlabel("Period, T (sec)")
    plt.ylabel("Normalized PSV / PGA (sec)")
    plt.title("Simplified Paper-Style Normalized PSV Spectrum")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig("normalized_PSV_over_PGA.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Done.")
    print("Files created:")
    print("- elcentro_response_spectra.csv")
    print("- spectral_displacement_Sd.png")
    print("- pseudo_velocity_PSV.png")
    print("- pseudo_acceleration_PSA.png")
    print("- normalized_PSV_over_PGA.png")


if __name__ == "__main__":
    main()