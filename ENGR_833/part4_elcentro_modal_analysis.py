from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Part 4 combined modal analysis for the alternate earthquake record
# -----------------------------------------------------------------------------
# This script redoes Part 1 and Part 2 using:
#   RSN6_IMPVALL.I_I-ELC270.AT2
#
# It runs two cases in one file:
#   1. Classical modal analysis with 5 percent viscous damping in all modes.
#   2. Rayleigh damping calibrated to modes 1 and 3.
#
# Internal units:
#   mass         = tonne = kN*s^2/m
#   stiffness    = kN/m
#   acceleration = m/s^2
#   displacement = m
#   base shear   = kN
#
# Extra output columns are also saved in inches and kips.

KIP_TO_KN = 4.4482216152605
IN_TO_M = 0.0254
DEFAULT_G = 9.81
DEFAULT_FLOOR_WEIGHT_KIP = 50.0
DEFAULT_STORY_STIFFNESS_KIP_PER_IN = 200.0
DEFAULT_RECORD = "RSN6_IMPVALL.I_I-ELC270.AT2"
DEFAULT_XI = 0.05
DEFAULT_TARGET_DAMPING = 0.05


def _unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def find_record_file(record_name: str) -> Path:
    """Find the requested PEER AT2 file.

    The function first tries the exact file name/path, then tries adding .AT2
    if the extension was omitted, and finally searches for Imperial Valley /
    ELC270-looking AT2 files in the current and script folders.
    """

    requested = Path(record_name).expanduser()
    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent

    search_dirs = _unique_paths([cwd, script_dir, cwd.parent, script_dir.parent])

    names_to_try = [requested.name]
    if requested.suffix == "":
        names_to_try.append(requested.name + ".AT2")
    names_to_try_lower = {name.lower() for name in names_to_try}

    # Absolute or explicitly relative path.
    if requested.is_absolute() and requested.is_file():
        return requested
    if requested.is_file():
        return requested
    if requested.suffix == "" and requested.with_suffix(".AT2").is_file():
        return requested.with_suffix(".AT2")

    # Exact name search in likely folders.
    exact_matches: list[Path] = []
    for folder in search_dirs:
        if not folder.exists():
            continue
        for file in folder.iterdir():
            if file.is_file() and file.name.lower() in names_to_try_lower:
                exact_matches.append(file)

    exact_matches = _unique_paths(exact_matches)
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        print("\nMultiple exact matches were found:")
        for file in exact_matches:
            print(f"    {file}")
        raise FileNotFoundError("Use --record with the full path to the desired AT2 file.")

    # Recursive fallback search for the alternate earthquake record.
    fallback_matches: list[Path] = []
    for folder in search_dirs:
        if not folder.exists():
            continue
        for file in folder.rglob("*.AT2"):
            name = file.name.lower()
            if (
                "rsn6" in name
                or "elc270" in name
                or ("impvall" in name and "270" in name)
            ):
                fallback_matches.append(file)

    fallback_matches = _unique_paths(fallback_matches)
    if len(fallback_matches) == 1:
        print("Using earthquake record found automatically:")
        print(f"    {fallback_matches[0]}")
        return fallback_matches[0]
    if len(fallback_matches) > 1:
        print("\nMultiple Imperial Valley / ELC270-looking AT2 files were found:")
        for file in fallback_matches:
            print(f"    {file}")
        raise FileNotFoundError("Use --record with the exact filename or full path.")

    print("\nPython searched these folders:")
    for folder in search_dirs:
        print(f"    {folder}")

    if script_dir.exists():
        print("\nFiles Python can see in the script folder:")
        for file in script_dir.iterdir():
            print(f"    {repr(file.name)}")

    raise FileNotFoundError(
        f"Could not find record file: {record_name}\n"
        f"Current working directory: {cwd}\n"
        f"Script directory: {script_dir}"
    )


def read_peer_at2(path: str | Path) -> tuple[np.ndarray, float, dict[str, float]]:
    """Read a PEER NGA .AT2 file whose acceleration values are in g."""

    path = Path(path)
    lines = path.read_text(errors="ignore").splitlines()

    npts = None
    dt = None
    data_start = None
    number_pattern = r"[+-]?(?:\d*\.\d+|\d+\.?\d*)(?:[Ee][+-]?\d+)?"

    for i, line in enumerate(lines):
        if "NPTS" in line.upper() and "DT" in line.upper():
            npts_match = re.search(r"NPTS\s*=\s*(\d+)", line, flags=re.IGNORECASE)
            dt_match = re.search(
                r"DT\s*=\s*(" + number_pattern + r")",
                line,
                flags=re.IGNORECASE,
            )
            if npts_match is None or dt_match is None:
                raise ValueError(f"Could not parse NPTS and DT from line:\n{line}")
            npts = int(npts_match.group(1))
            dt = float(dt_match.group(1))
            data_start = i + 1
            break

    if npts is None or dt is None or data_start is None:
        raise ValueError(f"Could not find NPTS/DT header in file: {path}")

    accel_g = np.fromstring(" ".join(lines[data_start:]), sep=" ")
    if accel_g.size < npts:
        raise ValueError(f"Expected {npts} points, but found {accel_g.size}.")

    return accel_g[:npts], dt, {"NPTS": float(npts), "DT": float(dt)}


def build_shear_building_matrices(
    n: int,
    m_floor_tonne: float,
    k_story_kn_per_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    m_mat = m_floor_tonne * np.eye(n)
    k_mat = np.zeros((n, n), dtype=float)

    # Story 1 connects ground to floor 1.
    k_mat[0, 0] += k_story_kn_per_m

    # Stories 2 through n connect adjacent floors.
    for story in range(1, n):
        lower = story - 1
        upper = story
        k_mat[lower, lower] += k_story_kn_per_m
        k_mat[upper, upper] += k_story_kn_per_m
        k_mat[lower, upper] -= k_story_kn_per_m
        k_mat[upper, lower] -= k_story_kn_per_m

    return m_mat, k_mat


def modal_properties(m_mat: np.ndarray, k_mat: np.ndarray) -> dict[str, np.ndarray]:
    n = m_mat.shape[0]
    r = np.ones(n)

    m_diag = np.diag(m_mat)
    if np.any(m_diag <= 0.0):
        raise ValueError("All floor masses must be positive.")

    m_inv_sqrt = np.diag(1.0 / np.sqrt(m_diag))

    # Symmetric standard eigenproblem equivalent to K phi = omega^2 M phi.
    a_mat = m_inv_sqrt @ k_mat @ m_inv_sqrt
    omega2, y = np.linalg.eigh(a_mat)

    order = np.argsort(omega2)
    omega2 = omega2[order]
    y = y[:, order]

    omega = np.sqrt(omega2)
    periods = 2.0 * np.pi / omega

    # Mass-normalized modes first, then roof-normalized modes for reporting.
    phi_mass = m_inv_sqrt @ y

    for mode in range(n):
        if phi_mass[-1, mode] < 0.0:
            phi_mass[:, mode] *= -1.0

    phi_roof = phi_mass / phi_mass[-1, :]

    modal_mass = np.zeros(n)
    modal_load = np.zeros(n)
    gamma = np.zeros(n)
    effective_modal_mass = np.zeros(n)

    for mode in range(n):
        phij = phi_roof[:, mode]
        modal_mass[mode] = phij.T @ m_mat @ phij
        modal_load[mode] = phij.T @ m_mat @ r
        gamma[mode] = modal_load[mode] / modal_mass[mode]
        effective_modal_mass[mode] = modal_load[mode] ** 2 / modal_mass[mode]

    total_mass = r.T @ m_mat @ r
    effective_modal_mass_ratio = effective_modal_mass / total_mass

    return {
        "omega2": omega2,
        "omega": omega,
        "periods": periods,
        "phi_roof": phi_roof,
        "modal_mass": modal_mass,
        "modal_load": modal_load,
        "gamma": gamma,
        "effective_modal_mass": effective_modal_mass,
        "effective_modal_mass_ratio": effective_modal_mass_ratio,
        "r": r,
    }


def rayleigh_coefficients(
    omega: np.ndarray,
    target_damping: float,
    mode_i: int = 1,
    mode_j: int = 3,
) -> tuple[float, float, np.ndarray]:
    """Compute Rayleigh alpha and beta using two target modes."""

    i = mode_i - 1
    j = mode_j - 1

    if i < 0 or j < 0 or i >= omega.size or j >= omega.size:
        raise ValueError("Rayleigh target modes must be valid mode numbers.")
    if i == j:
        raise ValueError("Rayleigh target modes must be different.")

    w_i = omega[i]
    w_j = omega[j]

    # xi_n = 0.5*(alpha/omega_n + beta*omega_n)
    a = np.array(
        [
            [1.0 / (2.0 * w_i), w_i / 2.0],
            [1.0 / (2.0 * w_j), w_j / 2.0],
        ]
    )
    b = np.array([target_damping, target_damping])

    alpha, beta = np.linalg.solve(a, b)
    xi_modes = 0.5 * (alpha / omega + beta * omega)

    return alpha, beta, xi_modes


def newmark_average_acceleration(
    force: np.ndarray,
    dt: float,
    mass: float,
    damping: float,
    stiffness: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve m*u_ddot + c*u_dot + k*u = force(t)."""

    beta = 1.0 / 4.0
    gamma_nm = 1.0 / 2.0

    n = force.size
    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)

    a[0] = (force[0] - damping * v[0] - stiffness * u[0]) / mass

    a0 = 1.0 / (beta * dt**2)
    a1 = gamma_nm / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma_nm / beta - 1.0
    a5 = dt * (gamma_nm / (2.0 * beta) - 1.0)
    a6 = dt * (1.0 - gamma_nm)
    a7 = gamma_nm * dt

    k_eff = stiffness + a0 * mass + a1 * damping

    for step in range(n - 1):
        f_eff = (
            force[step + 1]
            + mass * (a0 * u[step] + a2 * v[step] + a3 * a[step])
            + damping * (a1 * u[step] + a4 * v[step] + a5 * a[step])
        )

        u[step + 1] = f_eff / k_eff
        a[step + 1] = a0 * (u[step + 1] - u[step]) - a2 * v[step] - a3 * a[step]
        v[step + 1] = v[step] + a6 * a[step] + a7 * a[step + 1]

    return u, v, a


def compute_modal_time_histories(
    ag_m_per_s2: np.ndarray,
    dt: float,
    props: dict[str, np.ndarray],
    xi_modes: np.ndarray,
    k_mat: np.ndarray,
) -> dict[str, np.ndarray]:
    omega = props["omega"]
    phi = props["phi_roof"]
    gamma = props["gamma"]
    r = props["r"]

    n_steps = ag_m_per_s2.size
    n_modes = omega.size
    n_dof = phi.shape[0]

    if xi_modes.size != n_modes:
        raise ValueError("xi_modes must have one damping ratio for each mode.")

    modal_d = np.zeros((n_steps, n_modes))
    modal_v = np.zeros((n_steps, n_modes))
    modal_a = np.zeros((n_steps, n_modes))

    # Modal SDOF equation:
    # Dddot + 2*xi_n*omega_n*Ddot + omega_n^2*D = -ag(t)
    force = -ag_m_per_s2

    for mode in range(n_modes):
        c_mode = 2.0 * xi_modes[mode] * omega[mode]
        k_mode = omega[mode] ** 2

        modal_d[:, mode], modal_v[:, mode], modal_a[:, mode] = (
            newmark_average_acceleration(
                force=force,
                dt=dt,
                mass=1.0,
                damping=c_mode,
                stiffness=k_mode,
            )
        )

    u_modal = np.zeros((n_steps, n_dof, n_modes))
    for mode in range(n_modes):
        u_modal[:, :, mode] = gamma[mode] * modal_d[:, mode, None] * phi[:, mode]

    u_total = np.sum(u_modal, axis=2)
    roof_modal = u_modal[:, -1, :]
    roof_total = u_total[:, -1]

    base_modal = np.zeros((n_steps, n_modes))
    for mode in range(n_modes):
        base_modal[:, mode] = gamma[mode] * (r.T @ k_mat @ phi[:, mode]) * modal_d[:, mode]

    base_total = np.sum(base_modal, axis=1)

    return {
        "modal_D": modal_d,
        "modal_D_dot": modal_v,
        "modal_D_ddot": modal_a,
        "u_modal": u_modal,
        "u_total": u_total,
        "roof_modal": roof_modal,
        "roof_total": roof_total,
        "base_modal": base_modal,
        "base_total": base_total,
    }


def static_modal_contribution_for_force(
    props: dict[str, np.ndarray],
    m_mat: np.ndarray,
    k_mat: np.ndarray,
    force_distribution: np.ndarray,
    distribution_name: str,
) -> pd.DataFrame:
    """Chopra Table 12.11.1-style modal and cumulative factors."""

    phi = props["phi_roof"]
    omega2 = props["omega2"]
    r = props["r"]

    s = np.asarray(force_distribution, dtype=float).reshape(-1)
    n_modes = phi.shape[1]

    u_static_total = np.linalg.solve(k_mat, s)
    roof_static_total = u_static_total[-1]
    base_static_total = r.T @ s

    roof_modal = np.zeros(n_modes)
    base_modal = np.zeros(n_modes)

    for mode in range(n_modes):
        phij = phi[:, mode]
        modal_mass_j = phij.T @ m_mat @ phij
        modal_force_j = phij.T @ s

        u_static_j = phij * modal_force_j / (omega2[mode] * modal_mass_j)
        roof_modal[mode] = u_static_j[-1]
        base_modal[mode] = r.T @ k_mat @ u_static_j

    roof_factor = roof_modal / roof_static_total
    base_factor = base_modal / base_static_total

    return pd.DataFrame(
        {
            "mode": np.arange(1, n_modes + 1),
            "force_distribution": distribution_name,
            "roof_modal_static_displacement_m": roof_modal,
            "roof_modal_static_displacement_in": roof_modal / IN_TO_M,
            "roof_modal_factor": roof_factor,
            "roof_cumulative_factor": np.cumsum(roof_factor),
            "base_modal_static_shear_kN": base_modal,
            "base_modal_static_shear_kip": base_modal / KIP_TO_KN,
            "base_modal_factor": base_factor,
            "base_cumulative_factor": np.cumsum(base_factor),
        }
    )


def make_contribution_tables(
    props: dict[str, np.ndarray],
    m_mat: np.ndarray,
    k_mat: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    r = props["r"]

    # Earthquake inertia-force distribution for ground acceleration.
    s_eq = m_mat @ r

    # Chopra Table 12.11.1 force distributions for this same 5-story structure.
    s_a = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    s_b = np.array([0.0, 0.0, 0.0, -1.0, 2.0])

    earthquake_df = static_modal_contribution_for_force(
        props,
        m_mat,
        k_mat,
        s_eq,
        "earthquake_s_equals_Mr",
    )

    sa_df = static_modal_contribution_for_force(
        props,
        m_mat,
        k_mat,
        s_a,
        "s_a_Chopra_Table_12_11_1",
    )

    sb_df = static_modal_contribution_for_force(
        props,
        m_mat,
        k_mat,
        s_b,
        "s_b_Chopra_Table_12_11_1",
    )

    chopra_long_df = pd.concat([sa_df, sb_df], ignore_index=True)
    chopra_wide_df = pd.DataFrame(
        {
            "mode": sa_df["mode"],
            "sa_roof_modal_factor": sa_df["roof_modal_factor"],
            "sa_roof_cumulative_factor": sa_df["roof_cumulative_factor"],
            "sa_base_modal_factor": sa_df["base_modal_factor"],
            "sa_base_cumulative_factor": sa_df["base_cumulative_factor"],
            "sb_roof_modal_factor": sb_df["roof_modal_factor"],
            "sb_roof_cumulative_factor": sb_df["roof_cumulative_factor"],
            "sb_base_modal_factor": sb_df["base_modal_factor"],
            "sb_base_cumulative_factor": sb_df["base_cumulative_factor"],
        }
    )

    return earthquake_df, chopra_long_df, chopra_wide_df


def dynamic_peak_contribution_table(resp: dict[str, np.ndarray]) -> pd.DataFrame:
    roof_modal = resp["roof_modal"]
    base_modal = resp["base_modal"]
    roof_total = resp["roof_total"]
    base_total = resp["base_total"]

    n_modes = roof_modal.shape[1]

    roof_peak_modal = np.max(np.abs(roof_modal), axis=0)
    base_peak_modal = np.max(np.abs(base_modal), axis=0)
    roof_peak_total = np.max(np.abs(roof_total))
    base_peak_total = np.max(np.abs(base_total))
    roof_srss = np.sqrt(np.sum(roof_peak_modal**2))
    base_srss = np.sqrt(np.sum(base_peak_modal**2))

    return pd.DataFrame(
        {
            "mode": np.arange(1, n_modes + 1),
            "peak_abs_roof_modal_m": roof_peak_modal,
            "peak_abs_roof_modal_in": roof_peak_modal / IN_TO_M,
            "roof_fraction_of_srss_modal_peaks": roof_peak_modal / roof_srss,
            "roof_cumulative_srss_fraction": np.sqrt(np.cumsum(roof_peak_modal**2)) / roof_srss,
            "roof_peak_modal_over_exact_total_peak": roof_peak_modal / roof_peak_total,
            "peak_abs_base_modal_kN": base_peak_modal,
            "peak_abs_base_modal_kip": base_peak_modal / KIP_TO_KN,
            "base_fraction_of_srss_modal_peaks": base_peak_modal / base_srss,
            "base_cumulative_srss_fraction": np.sqrt(np.cumsum(base_peak_modal**2)) / base_srss,
            "base_peak_modal_over_exact_total_peak": base_peak_modal / base_peak_total,
        }
    )


def contribution_at_peak_total(
    resp: dict[str, np.ndarray],
    t: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def make_table(
        modal: np.ndarray,
        total: np.ndarray,
        quantity: str,
        unit: str,
    ) -> pd.DataFrame:
        idx = int(np.argmax(np.abs(total)))
        total_at_peak = total[idx]
        modal_at_peak = modal[idx, :]

        return pd.DataFrame(
            {
                "mode": np.arange(1, modal.shape[1] + 1),
                "quantity": quantity,
                "time_of_peak_total_s": t[idx],
                f"total_at_peak_{unit}": total_at_peak,
                f"modal_value_at_peak_{unit}": modal_at_peak,
                "modal_fraction_at_peak_total": modal_at_peak / total_at_peak,
                "cumulative_fraction_at_peak_total": np.cumsum(modal_at_peak) / total_at_peak,
            }
        )

    roof_df = make_table(resp["roof_modal"], resp["roof_total"], "roof displacement", "m")
    base_df = make_table(resp["base_modal"], resp["base_total"], "base shear", "kN")

    return roof_df, base_df


def save_line_plot(
    x: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.asarray(y)

    if y.ndim == 1:
        ax.plot(x, y, label=labels[0] if labels else None)
    else:
        for col in range(y.shape[1]):
            ax.plot(x, y[:, col], label=labels[col])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)

    if labels:
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_plots(
    outdir: Path,
    t: np.ndarray,
    accel_g: np.ndarray,
    resp: dict[str, np.ndarray],
    plot_title_record: str,
    damping_title: str,
    file_suffix: str,
) -> None:
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    n_modes = resp["modal_D"].shape[1]
    labels = [f"Mode {mode}" for mode in range(1, n_modes + 1)]

    save_line_plot(
        t,
        accel_g,
        ["Ground acceleration"],
        "Time (s)",
        "Ground acceleration (g)",
        f"{plot_title_record} ground acceleration input",
        figdir / "ground_acceleration.png",
    )

    save_line_plot(
        t,
        resp["modal_D"] / IN_TO_M,
        labels,
        "Time (s)",
        "SDF modal deformation Dn (in)",
        f"Modal SDF deformation responses, {damping_title}",
        figdir / f"modal_sdf_deformations{file_suffix}.png",
    )

    save_line_plot(
        t,
        resp["roof_modal"] / IN_TO_M,
        labels,
        "Time (s)",
        "Modal roof displacement contribution (in)",
        f"Modal roof displacement contributions, {damping_title}",
        figdir / f"modal_roof_displacement_contributions{file_suffix}.png",
    )

    save_line_plot(
        t,
        resp["roof_total"] / IN_TO_M,
        ["Total roof displacement"],
        "Time (s)",
        "Roof displacement (in)",
        f"Total roof displacement from all 5 modes, {damping_title}",
        figdir / f"total_roof_displacement{file_suffix}.png",
    )

    save_line_plot(
        t,
        resp["base_modal"] / KIP_TO_KN,
        labels,
        "Time (s)",
        "Modal base shear contribution (kip)",
        f"Modal base shear contributions, {damping_title}",
        figdir / f"modal_base_shear_contributions{file_suffix}.png",
    )

    save_line_plot(
        t,
        resp["base_total"] / KIP_TO_KN,
        ["Total base shear"],
        "Time (s)",
        "Base shear (kip)",
        f"Total base shear from all 5 modes, {damping_title}",
        figdir / f"total_base_shear{file_suffix}.png",
    )


def write_modal_properties(
    outdir: Path,
    props: dict[str, np.ndarray],
    xi_modes: np.ndarray,
    file_name: str,
    damping_column_name: str,
) -> pd.DataFrame:
    n_stories = len(props["omega"])

    modal_df = pd.DataFrame(
        {
            "mode": np.arange(1, n_stories + 1),
            "omega_rad_per_s": props["omega"],
            "period_s": props["periods"],
            damping_column_name: xi_modes,
            "damping_percent": 100.0 * xi_modes,
            "modal_mass_tonne": props["modal_mass"],
            "modal_load_tonne": props["modal_load"],
            "Gamma_roof_normalized": props["gamma"],
            "effective_modal_mass_tonne": props["effective_modal_mass"],
            "effective_modal_mass_ratio": props["effective_modal_mass_ratio"],
            "cumulative_effective_mass_ratio": np.cumsum(props["effective_modal_mass_ratio"]),
        }
    )

    for floor in range(n_stories):
        modal_df[f"phi_floor_{floor + 1}_roof_norm"] = props["phi_roof"][floor, :]

    modal_df.to_csv(outdir / file_name, index=False)
    return modal_df


def save_common_static_outputs(
    outdir: Path,
    props: dict[str, np.ndarray],
    m_mat: np.ndarray,
    k_mat: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    earthquake_df, chopra_long_df, chopra_wide_df = make_contribution_tables(props, m_mat, k_mat)

    earthquake_df.to_csv(outdir / "earthquake_Mr_static_contribution_factors.csv", index=False)
    chopra_long_df.to_csv(outdir / "chopra_table_12_11_1_sa_sb_long.csv", index=False)
    chopra_wide_df.to_csv(outdir / "chopra_table_12_11_1_sa_sb_wide.csv", index=False)
    chopra_wide_df.to_csv(outdir / "chopra_static_contribution_factors.csv", index=False)

    return earthquake_df, chopra_long_df, chopra_wide_df


def run_case(
    case_name: str,
    outdir: Path,
    record_label: str,
    accel_g_unscaled: np.ndarray,
    accel_g: np.ndarray,
    ag_m_per_s2: np.ndarray,
    dt: float,
    t: np.ndarray,
    meta: dict[str, float],
    m_mat: np.ndarray,
    k_mat: np.ndarray,
    props: dict[str, np.ndarray],
    xi_modes: np.ndarray,
    floor_weight_kip: float,
    story_stiffness_kip_per_in: float,
    m_floor_tonne: float,
    k_story_kn_per_m: float,
    scale: float,
    suffix: str,
    modal_properties_filename: str,
    response_summary_filename: str,
    dynamic_contribution_filename: str,
    roof_peak_filename: str,
    base_peak_filename: str,
    response_histories_filename: str,
    damping_title: str,
    damping_column_name: str,
    rayleigh_alpha: float | None = None,
    rayleigh_beta: float | None = None,
    rayleigh_modes: tuple[int, int] | None = None,
    target_xi: float | None = None,
) -> dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)

    np.savetxt(outdir / "M_matrix_tonne.csv", m_mat, delimiter=",")
    np.savetxt(outdir / "K_matrix_kN_per_m.csv", k_mat, delimiter=",")

    if rayleigh_alpha is not None and rayleigh_beta is not None:
        c_mat = rayleigh_alpha * m_mat + rayleigh_beta * k_mat
        np.savetxt(outdir / "C_rayleigh_matrix_tonne_per_s.csv", c_mat, delimiter=",")

    pd.DataFrame(
        {
            "time_s": t,
            "ag_g_unscaled": accel_g_unscaled,
            "ag_g_after_scale": accel_g,
            "ag_m_per_s2_after_scale": ag_m_per_s2,
        }
    ).to_csv(outdir / "ground_motion_used.csv", index=False)

    modal_df = write_modal_properties(
        outdir=outdir,
        props=props,
        xi_modes=xi_modes,
        file_name=modal_properties_filename,
        damping_column_name=damping_column_name,
    )

    if rayleigh_alpha is not None and rayleigh_beta is not None and rayleigh_modes is not None:
        rayleigh_summary_df = pd.DataFrame(
            {
                "quantity": [
                    "target_damping_ratio",
                    "target_damping_percent",
                    "rayleigh_mode_i",
                    "rayleigh_mode_j",
                    "alpha_mass_coefficient_1_per_s",
                    "beta_stiffness_coefficient_s",
                    "xi_mode_1",
                    "xi_mode_2",
                    "xi_mode_3",
                    "xi_mode_4",
                    "xi_mode_5",
                ],
                "value": [
                    target_xi,
                    100.0 * target_xi if target_xi is not None else np.nan,
                    rayleigh_modes[0],
                    rayleigh_modes[1],
                    rayleigh_alpha,
                    rayleigh_beta,
                    *xi_modes,
                ],
            }
        )
        rayleigh_summary_df.to_csv(outdir / "rayleigh_damping_summary.csv", index=False)

    resp = compute_modal_time_histories(
        ag_m_per_s2=ag_m_per_s2,
        dt=dt,
        props=props,
        xi_modes=xi_modes,
        k_mat=k_mat,
    )

    summary_quantities = [
        "case_name",
        "record_label",
        "NPTS",
        "DT_s",
        "floor_weight_kip",
        "story_stiffness_kip_per_in",
        "mass_per_floor_tonne_used",
        "story_stiffness_kN_per_m_used",
        "ground_motion_scale_factor",
        "peak_abs_roof_displacement_m",
        "peak_abs_roof_displacement_in",
        "time_of_peak_abs_roof_displacement_s",
        "peak_abs_base_shear_kN",
        "peak_abs_base_shear_kip",
        "time_of_peak_abs_base_shear_s",
        "PGA_g_after_scale",
        "PGA_m_per_s2_after_scale",
    ]
    summary_values: list[object] = [
        case_name,
        record_label,
        int(meta["NPTS"]),
        meta["DT"],
        floor_weight_kip,
        story_stiffness_kip_per_in,
        m_floor_tonne,
        k_story_kn_per_m,
        scale,
        np.max(np.abs(resp["roof_total"])),
        np.max(np.abs(resp["roof_total"])) / IN_TO_M,
        t[np.argmax(np.abs(resp["roof_total"]))],
        np.max(np.abs(resp["base_total"])),
        np.max(np.abs(resp["base_total"])) / KIP_TO_KN,
        t[np.argmax(np.abs(resp["base_total"]))],
        np.max(np.abs(accel_g)),
        np.max(np.abs(ag_m_per_s2)),
    ]

    if rayleigh_alpha is not None and rayleigh_beta is not None and rayleigh_modes is not None:
        insert_after = summary_quantities.index("ground_motion_scale_factor") + 1
        extra_quantities = [
            "target_damping_ratio_modes_1_and_3",
            "rayleigh_mode_i",
            "rayleigh_mode_j",
            "rayleigh_alpha_1_per_s",
            "rayleigh_beta_s",
        ]
        extra_values = [
            target_xi,
            rayleigh_modes[0],
            rayleigh_modes[1],
            rayleigh_alpha,
            rayleigh_beta,
        ]
        summary_quantities[insert_after:insert_after] = extra_quantities
        summary_values[insert_after:insert_after] = extra_values
    else:
        insert_after = summary_quantities.index("ground_motion_scale_factor") + 1
        summary_quantities.insert(insert_after, "damping_ratio_all_modes")
        summary_values.insert(insert_after, xi_modes[0])

    summary_df = pd.DataFrame({"quantity": summary_quantities, "value": summary_values})
    summary_df.to_csv(outdir / response_summary_filename, index=False)

    earthquake_df, chopra_long_df, chopra_wide_df = save_common_static_outputs(outdir, props, m_mat, k_mat)

    dynamic_df = dynamic_peak_contribution_table(resp)
    dynamic_df.to_csv(outdir / dynamic_contribution_filename, index=False)

    roof_peak_df, base_peak_df = contribution_at_peak_total(resp, t)
    roof_peak_df.to_csv(outdir / roof_peak_filename, index=False)
    base_peak_df.to_csv(outdir / base_peak_filename, index=False)

    response_df = pd.DataFrame(
        {
            "time_s": t,
            "ag_g": accel_g,
            "ag_m_per_s2": ag_m_per_s2,
            "roof_total_m": resp["roof_total"],
            "roof_total_in": resp["roof_total"] / IN_TO_M,
            "base_total_kN": resp["base_total"],
            "base_total_kip": resp["base_total"] / KIP_TO_KN,
        }
    )

    n_modes = props["omega"].size
    for mode in range(n_modes):
        response_df[f"D_mode_{mode + 1}_m"] = resp["modal_D"][:, mode]
        response_df[f"D_mode_{mode + 1}_in"] = resp["modal_D"][:, mode] / IN_TO_M
        response_df[f"Ddot_mode_{mode + 1}_m_per_s"] = resp["modal_D_dot"][:, mode]
        response_df[f"Dddot_mode_{mode + 1}_m_per_s2"] = resp["modal_D_ddot"][:, mode]
        response_df[f"roof_mode_{mode + 1}_m"] = resp["roof_modal"][:, mode]
        response_df[f"roof_mode_{mode + 1}_in"] = resp["roof_modal"][:, mode] / IN_TO_M
        response_df[f"base_mode_{mode + 1}_kN"] = resp["base_modal"][:, mode]
        response_df[f"base_mode_{mode + 1}_kip"] = resp["base_modal"][:, mode] / KIP_TO_KN

    response_df.to_csv(outdir / response_histories_filename, index=False)

    save_plots(
        outdir=outdir,
        t=t,
        accel_g=accel_g,
        resp=resp,
        plot_title_record=record_label,
        damping_title=damping_title,
        file_suffix=suffix,
    )

    return {
        "response": resp,
        "summary_df": summary_df,
        "modal_df": modal_df,
        "earthquake_df": earthquake_df,
        "chopra_long_df": chopra_long_df,
        "chopra_wide_df": chopra_wide_df,
        "dynamic_df": dynamic_df,
        "roof_peak_df": roof_peak_df,
        "base_peak_df": base_peak_df,
        "response_df": response_df,
    }


def _summary_value(summary_df: pd.DataFrame, quantity: str) -> float:
    value = summary_df.loc[summary_df["quantity"] == quantity, "value"].iloc[0]
    return float(value)


def save_part4_comparison(
    outdir: Path,
    t: np.ndarray,
    five_percent: dict[str, object],
    rayleigh: dict[str, object],
    xi_5: np.ndarray,
    xi_rayleigh: np.ndarray,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    summary_5 = five_percent["summary_df"]
    summary_r = rayleigh["summary_df"]
    if not isinstance(summary_5, pd.DataFrame) or not isinstance(summary_r, pd.DataFrame):
        raise TypeError("Expected summary data frames in results dictionaries.")

    quantities = [
        "PGA_g_after_scale",
        "peak_abs_roof_displacement_in",
        "time_of_peak_abs_roof_displacement_s",
        "peak_abs_base_shear_kip",
        "time_of_peak_abs_base_shear_s",
    ]

    rows = []
    for quantity in quantities:
        value_5 = _summary_value(summary_5, quantity)
        value_r = _summary_value(summary_r, quantity)
        rows.append(
            {
                "quantity": quantity,
                "five_percent_all_modes": value_5,
                "rayleigh_modes_1_and_3": value_r,
                "rayleigh_minus_five_percent": value_r - value_5,
                "rayleigh_over_five_percent": value_r / value_5 if value_5 != 0.0 else np.nan,
                "percent_difference_from_five_percent": 100.0 * (value_r - value_5) / value_5 if value_5 != 0.0 else np.nan,
            }
        )

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(outdir / "part4_5percent_vs_rayleigh_summary.csv", index=False)

    damping_df = pd.DataFrame(
        {
            "mode": np.arange(1, xi_5.size + 1),
            "five_percent_damping_ratio": xi_5,
            "five_percent_damping_percent": 100.0 * xi_5,
            "rayleigh_damping_ratio": xi_rayleigh,
            "rayleigh_damping_percent": 100.0 * xi_rayleigh,
        }
    )
    damping_df.to_csv(outdir / "part4_damping_ratios_comparison.csv", index=False)

    resp_5 = five_percent["response"]
    resp_r = rayleigh["response"]
    if not isinstance(resp_5, dict) or not isinstance(resp_r, dict):
        raise TypeError("Expected response dictionaries in results dictionaries.")

    save_line_plot(
        t,
        np.column_stack([resp_5["roof_total"] / IN_TO_M, resp_r["roof_total"] / IN_TO_M]),
        ["5% all modes", "Rayleigh modes 1 and 3"],
        "Time (s)",
        "Roof displacement (in)",
        "Part 4 total roof displacement comparison",
        outdir / "part4_total_roof_displacement_comparison.png",
    )

    save_line_plot(
        t,
        np.column_stack([resp_5["base_total"] / KIP_TO_KN, resp_r["base_total"] / KIP_TO_KN]),
        ["5% all modes", "Rayleigh modes 1 and 3"],
        "Time (s)",
        "Base shear (kip)",
        "Part 4 total base shear comparison",
        outdir / "part4_total_base_shear_comparison.png",
    )

    notes = (
        "Part 4 comparison notes\n\n"
        "1. Natural periods, mode shapes, participation factors, and static modal contribution factors depend only on M and K. "
        "They should match Parts 1 and 2 if the mass and stiffness values are unchanged.\n"
        "2. Dynamic response quantities such as peak roof displacement, peak base shear, and modal time histories change because the Imperial Valley ELC270 record has different amplitude, duration, and frequency content than the Northridge record.\n"
        "3. For the same Imperial Valley record, the 5 percent case and Rayleigh case differ only because the modal damping ratios differ. Rayleigh damping is exactly the target value at modes 1 and 3, but the other modes use the Rayleigh curve.\n"
        "4. Base shear is often more sensitive than roof displacement to high-mode response, so Rayleigh damping differences in higher modes may show up more strongly in base-shear plots and contribution tables.\n"
    )
    (outdir / "part4_comment_guidance.txt").write_text(notes)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Part 4 combined modal analysis for the Imperial Valley ELC270 record."
    )

    parser.add_argument(
        "--record",
        default=DEFAULT_RECORD,
        help="Path to the PEER .AT2 acceleration record. Default is RSN6_IMPVALL.I_I-ELC270.AT2.",
    )
    parser.add_argument(
        "--floor-weight-kip",
        type=float,
        default=DEFAULT_FLOOR_WEIGHT_KIP,
        help="Floor weight in kip. Default is 50 kip.",
    )
    parser.add_argument(
        "--story-stiffness-kip-per-in",
        type=float,
        default=DEFAULT_STORY_STIFFNESS_KIP_PER_IN,
        help="Story stiffness in kip/in. Default is 200 kip/in.",
    )
    parser.add_argument(
        "--m-tonne",
        type=float,
        default=None,
        help="Optional direct mass input in tonnes. Overrides --floor-weight-kip.",
    )
    parser.add_argument(
        "--k-kn-per-m",
        type=float,
        default=None,
        help="Optional direct stiffness input in kN/m. Overrides --story-stiffness-kip-per-in.",
    )
    parser.add_argument(
        "--xi",
        type=float,
        default=DEFAULT_XI,
        help="Damping ratio for the all-modes case. Default is 0.05.",
    )
    parser.add_argument(
        "--target-xi",
        type=float,
        default=DEFAULT_TARGET_DAMPING,
        help="Target damping ratio for Rayleigh calibration modes. Default is 0.05.",
    )
    parser.add_argument(
        "--rayleigh-mode-i",
        type=int,
        default=1,
        help="First target mode number for Rayleigh damping. Default is 1.",
    )
    parser.add_argument(
        "--rayleigh-mode-j",
        type=int,
        default=3,
        help="Second target mode number for Rayleigh damping. Default is 3.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Extra scale factor on ground motion. Default is 1.0.",
    )
    parser.add_argument(
        "--g",
        type=float,
        default=DEFAULT_G,
        help="m/s^2 per g. Default is 9.81.",
    )
    parser.add_argument(
        "--outdir-5percent",
        default="part4_results_elcentro_5percent",
        help="Output directory for the 5 percent damping case.",
    )
    parser.add_argument(
        "--outdir-rayleigh",
        default="part4_results_elcentro_rayleigh",
        help="Output directory for the Rayleigh damping case.",
    )
    parser.add_argument(
        "--comparison-outdir",
        default="part4_results_elcentro_comparison",
        help="Output directory for the Part 4 comparison files.",
    )

    args = parser.parse_args()

    record_path = find_record_file(args.record)
    record_label = "Imperial Valley ELC270"

    if args.m_tonne is None:
        m_floor_tonne = args.floor_weight_kip * KIP_TO_KN / args.g
    else:
        m_floor_tonne = args.m_tonne

    if args.k_kn_per_m is None:
        k_story_kn_per_m = args.story_stiffness_kip_per_in * KIP_TO_KN / IN_TO_M
    else:
        k_story_kn_per_m = args.k_kn_per_m

    accel_g_unscaled, dt, meta = read_peer_at2(record_path)
    accel_g = accel_g_unscaled * args.scale
    ag_m_per_s2 = accel_g * args.g
    t = np.arange(accel_g.size) * dt

    n_stories = 5
    m_mat, k_mat = build_shear_building_matrices(
        n=n_stories,
        m_floor_tonne=m_floor_tonne,
        k_story_kn_per_m=k_story_kn_per_m,
    )
    props = modal_properties(m_mat, k_mat)

    xi_5 = np.full(n_stories, args.xi)

    alpha_rayleigh, beta_rayleigh, xi_rayleigh = rayleigh_coefficients(
        omega=props["omega"],
        target_damping=args.target_xi,
        mode_i=args.rayleigh_mode_i,
        mode_j=args.rayleigh_mode_j,
    )

    five_percent_results = run_case(
        case_name="part4_5percent_all_modes",
        outdir=Path(args.outdir_5percent),
        record_label=record_label,
        accel_g_unscaled=accel_g_unscaled,
        accel_g=accel_g,
        ag_m_per_s2=ag_m_per_s2,
        dt=dt,
        t=t,
        meta=meta,
        m_mat=m_mat,
        k_mat=k_mat,
        props=props,
        xi_modes=xi_5,
        floor_weight_kip=args.floor_weight_kip,
        story_stiffness_kip_per_in=args.story_stiffness_kip_per_in,
        m_floor_tonne=m_floor_tonne,
        k_story_kn_per_m=k_story_kn_per_m,
        scale=args.scale,
        suffix="",
        modal_properties_filename="modal_properties.csv",
        response_summary_filename="response_summary.csv",
        dynamic_contribution_filename="time_history_peak_contribution_factors.csv",
        roof_peak_filename="roof_contribution_at_peak_total.csv",
        base_peak_filename="base_shear_contribution_at_peak_total.csv",
        response_histories_filename="response_time_histories.csv",
        damping_title=f"damping = {100.0 * args.xi:g}% in all modes",
        damping_column_name="damping_ratio_all_modes",
    )

    rayleigh_results = run_case(
        case_name="part4_rayleigh_modes_1_and_3",
        outdir=Path(args.outdir_rayleigh),
        record_label=record_label,
        accel_g_unscaled=accel_g_unscaled,
        accel_g=accel_g,
        ag_m_per_s2=ag_m_per_s2,
        dt=dt,
        t=t,
        meta=meta,
        m_mat=m_mat,
        k_mat=k_mat,
        props=props,
        xi_modes=xi_rayleigh,
        floor_weight_kip=args.floor_weight_kip,
        story_stiffness_kip_per_in=args.story_stiffness_kip_per_in,
        m_floor_tonne=m_floor_tonne,
        k_story_kn_per_m=k_story_kn_per_m,
        scale=args.scale,
        suffix="_rayleigh",
        modal_properties_filename="modal_properties_rayleigh.csv",
        response_summary_filename="response_summary_rayleigh.csv",
        dynamic_contribution_filename="time_history_peak_contribution_factors_rayleigh.csv",
        roof_peak_filename="roof_contribution_at_peak_total_rayleigh.csv",
        base_peak_filename="base_shear_contribution_at_peak_total_rayleigh.csv",
        response_histories_filename="response_time_histories_rayleigh.csv",
        damping_title="Rayleigh damping calibrated to modes 1 and 3",
        damping_column_name="rayleigh_damping_ratio",
        rayleigh_alpha=alpha_rayleigh,
        rayleigh_beta=beta_rayleigh,
        rayleigh_modes=(args.rayleigh_mode_i, args.rayleigh_mode_j),
        target_xi=args.target_xi,
    )

    save_part4_comparison(
        outdir=Path(args.comparison_outdir),
        t=t,
        five_percent=five_percent_results,
        rayleigh=rayleigh_results,
        xi_5=xi_5,
        xi_rayleigh=xi_rayleigh,
    )

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.precision", 6)

    print("\nPart 4 combined analysis complete.")
    print(f"Record used: {record_path}")
    print(f"NPTS = {int(meta['NPTS'])}")
    print(f"DT = {meta['DT']:.5f} s")
    print(f"Floor weight used = {args.floor_weight_kip:g} kip")
    print(f"Mass per floor used = {m_floor_tonne:.6g} tonne")
    print(f"Story stiffness used = {args.story_stiffness_kip_per_in:g} kip/in")
    print(f"Story stiffness used = {k_story_kn_per_m:.6g} kN/m")
    print(f"5 percent case damping ratio = {args.xi:g}")
    print(f"Target Rayleigh damping = {100.0 * args.target_xi:g}%")
    print(f"Rayleigh target modes = {args.rayleigh_mode_i} and {args.rayleigh_mode_j}")
    print(f"Rayleigh alpha = {alpha_rayleigh:.8g} 1/s")
    print(f"Rayleigh beta  = {beta_rayleigh:.8g} s")
    print(f"Ground motion scale factor = {args.scale:g}")
    print(f"5 percent outputs written to: {Path(args.outdir_5percent).resolve()}")
    print(f"Rayleigh outputs written to: {Path(args.outdir_rayleigh).resolve()}")
    print(f"Comparison outputs written to: {Path(args.comparison_outdir).resolve()}")

    modal_print_df = five_percent_results["modal_df"]
    if isinstance(modal_print_df, pd.DataFrame):
        print("\nNatural periods, participation factors, and effective mass ratios:")
        print(
            modal_print_df[
                [
                    "mode",
                    "omega_rad_per_s",
                    "period_s",
                    "Gamma_roof_normalized",
                    "effective_modal_mass_ratio",
                    "cumulative_effective_mass_ratio",
                ]
            ].to_string(index=False)
        )

    rayleigh_modal_print_df = rayleigh_results["modal_df"]
    if isinstance(rayleigh_modal_print_df, pd.DataFrame):
        print("\nRayleigh modal damping ratios:")
        print(
            rayleigh_modal_print_df[
                ["mode", "rayleigh_damping_ratio", "damping_percent"]
            ].to_string(index=False)
        )

    print("\nPart 4 response comparison:")
    comparison_csv = Path(args.comparison_outdir) / "part4_5percent_vs_rayleigh_summary.csv"
    print(pd.read_csv(comparison_csv).to_string(index=False))

    print("\nKey files to use for Part 4 write-up:")
    for file_path in [
        Path(args.outdir_5percent) / "response_summary.csv",
        Path(args.outdir_5percent) / "response_time_histories.csv",
        Path(args.outdir_5percent) / "figures",
        Path(args.outdir_rayleigh) / "response_summary_rayleigh.csv",
        Path(args.outdir_rayleigh) / "response_time_histories_rayleigh.csv",
        Path(args.outdir_rayleigh) / "figures",
        Path(args.comparison_outdir) / "part4_5percent_vs_rayleigh_summary.csv",
        Path(args.comparison_outdir) / "part4_damping_ratios_comparison.csv",
        Path(args.comparison_outdir) / "part4_comment_guidance.txt",
    ]:
        print(f"    {file_path}")


if __name__ == "__main__":
    main()
