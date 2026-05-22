from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Default structural properties requested by user
# -----------------------------------------------------------------------------
# Interpreting "k" as kip:
#   floor weight W = 50 kip at each floor, so mass m = W/g
#   story stiffness k = 200 kip/in
# Internal units used by the analysis:
#   mass       = tonne = kN*s^2/m
#   stiffness  = kN/m
#   acceleration = m/s^2
#   displacement = m
#   base shear   = kN
# Extra output columns are also written in inches and kips.

KIP_TO_KN = 4.4482216152605
IN_TO_M = 0.0254
DEFAULT_G = 9.81
DEFAULT_FLOOR_WEIGHT_KIP = 50.0
DEFAULT_STORY_STIFFNESS_KIP_PER_IN = 200.0


def find_record_file(record_name: str) -> Path:
    requested = Path(record_name).expanduser()
    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent

    search_dirs = []
    for folder in [cwd, script_dir, cwd.parent, script_dir.parent]:
        if folder not in search_dirs:
            search_dirs.append(folder)

    if requested.is_absolute() and requested.is_file():
        return requested

    for folder in search_dirs:
        candidate = folder / requested
        if candidate.is_file():
            return candidate

    matches = []
    for folder in search_dirs:
        if not folder.exists():
            continue
        for file in folder.rglob("*"):
            if file.is_file():
                name = file.name.lower()
                if "north" in name and ".at2" in name:
                    matches.append(file)

    matches = list(dict.fromkeys(matches))

    if len(matches) == 1:
        print("Using earthquake record found automatically:")
        print(f"    {matches[0]}")
        return matches[0]

    if len(matches) > 1:
        print("\nMultiple Northridge-looking AT2 files were found:")
        for file in matches:
            print(f"    {file}")
        raise FileNotFoundError("Use --record with the exact filename or full path.")

    print("\nPython searched these folders:")
    for folder in search_dirs:
        print(f"    {folder}")

    print("\nFiles Python can see in the script folder:")
    for file in script_dir.iterdir():
        print(f"    {repr(file.name)}")

    raise FileNotFoundError(
        f"Could not find record file: {record_name}\n"
        f"Current working directory: {cwd}\n"
        f"Script directory: {script_dir}"
    )


def read_peer_at2(path: str | Path) -> tuple[np.ndarray, float, dict[str, float]]:
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
):
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


def modal_properties(
    m_mat: np.ndarray,
    k_mat: np.ndarray,
) -> dict[str, np.ndarray]:
    n = m_mat.shape[0]
    r = np.ones(n)

    m_diag = np.diag(m_mat)
    m_inv_sqrt = np.diag(1.0 / np.sqrt(m_diag))

    # Symmetric standard eigenproblem equivalent to:
    # K phi = omega^2 M phi
    a_mat = m_inv_sqrt @ k_mat @ m_inv_sqrt

    omega2, y = np.linalg.eigh(a_mat)

    order = np.argsort(omega2)
    omega2 = omega2[order]
    y = y[:, order]

    omega = np.sqrt(omega2)
    periods = 2.0 * np.pi / omega

    # Mass-normalized mode shapes first, then roof-normalize them.
    phi_mass = m_inv_sqrt @ y

    for j in range(n):
        if phi_mass[-1, j] < 0.0:
            phi_mass[:, j] *= -1.0

    phi_roof = phi_mass / phi_mass[-1, :]

    modal_mass = np.zeros(n)
    modal_load = np.zeros(n)
    gamma = np.zeros(n)
    effective_modal_mass = np.zeros(n)

    for j in range(n):
        phij = phi_roof[:, j]

        modal_mass[j] = phij.T @ m_mat @ phij
        modal_load[j] = phij.T @ m_mat @ r
        gamma[j] = modal_load[j] / modal_mass[j]
        effective_modal_mass[j] = modal_load[j] ** 2 / modal_mass[j]

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


def newmark_average_acceleration(
    force: np.ndarray,
    dt: float,
    mass: float,
    damping: float,
    stiffness: float,
):
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

    for i in range(n - 1):
        f_eff = (
            force[i + 1]
            + mass * (a0 * u[i] + a2 * v[i] + a3 * a[i])
            + damping * (a1 * u[i] + a4 * v[i] + a5 * a[i])
        )

        u[i + 1] = f_eff / k_eff
        a[i + 1] = a0 * (u[i + 1] - u[i]) - a2 * v[i] - a3 * a[i]
        v[i + 1] = v[i] + a6 * a[i] + a7 * a[i + 1]

    return u, v, a


def compute_modal_time_histories(
    ag_m_per_s2: np.ndarray,
    dt: float,
    props: dict[str, np.ndarray],
    damping_ratio: float,
    k_mat: np.ndarray,
) -> dict[str, np.ndarray]:
    omega = props["omega"]
    phi = props["phi_roof"]
    gamma = props["gamma"]
    r = props["r"]

    n_steps = ag_m_per_s2.size
    n_modes = omega.size
    n_dof = phi.shape[0]

    modal_d = np.zeros((n_steps, n_modes))
    modal_v = np.zeros((n_steps, n_modes))
    modal_a = np.zeros((n_steps, n_modes))

    # Modal SDOF equation:
    # Dddot + 2 xi omega Ddot + omega^2 D = -ag(t)
    force = -ag_m_per_s2

    for j in range(n_modes):
        c_j = 2.0 * damping_ratio * omega[j]
        k_j = omega[j] ** 2

        modal_d[:, j], modal_v[:, j], modal_a[:, j] = (
            newmark_average_acceleration(
                force=force,
                dt=dt,
                mass=1.0,
                damping=c_j,
                stiffness=k_j,
            )
        )

    u_modal = np.zeros((n_steps, n_dof, n_modes))

    for j in range(n_modes):
        u_modal[:, :, j] = gamma[j] * modal_d[:, j, None] * phi[:, j]

    u_total = np.sum(u_modal, axis=2)

    roof_modal = u_modal[:, -1, :]
    roof_total = u_total[:, -1]

    base_modal = np.zeros((n_steps, n_modes))

    for j in range(n_modes):
        base_modal[:, j] = (
            gamma[j]
            * (r.T @ k_mat @ phi[:, j])
            * modal_d[:, j]
        )

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

    for j in range(n_modes):
        phij = phi[:, j]

        modal_mass_j = phij.T @ m_mat @ phij
        modal_force_j = phij.T @ s

        u_static_j = phij * modal_force_j / (omega2[j] * modal_mass_j)

        roof_modal[j] = u_static_j[-1]
        base_modal[j] = r.T @ k_mat @ u_static_j

    roof_factor = roof_modal / roof_static_total
    base_factor = base_modal / base_static_total

    return pd.DataFrame(
        {
            "mode": np.arange(1, n_modes + 1),
            "force_distribution": distribution_name,
            "roof_modal_static_displacement_m": roof_modal,
            "roof_modal_factor": roof_factor,
            "roof_cumulative_factor": np.cumsum(roof_factor),
            "base_modal_static_shear": base_modal,
            "base_modal_factor": base_factor,
            "base_cumulative_factor": np.cumsum(base_factor),
        }
    )


def make_contribution_tables(
    props: dict[str, np.ndarray],
    m_mat: np.ndarray,
    k_mat: np.ndarray,
):
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


def dynamic_peak_contribution_table(
    resp: dict[str, np.ndarray],
) -> pd.DataFrame:
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
            "roof_cumulative_srss_fraction": (
                np.sqrt(np.cumsum(roof_peak_modal**2)) / roof_srss
            ),
            "roof_peak_modal_over_exact_total_peak": (
                roof_peak_modal / roof_peak_total
            ),
            "peak_abs_base_modal_kN": base_peak_modal,
            "peak_abs_base_modal_kip": base_peak_modal / KIP_TO_KN,
            "base_fraction_of_srss_modal_peaks": base_peak_modal / base_srss,
            "base_cumulative_srss_fraction": (
                np.sqrt(np.cumsum(base_peak_modal**2)) / base_srss
            ),
            "base_peak_modal_over_exact_total_peak": (
                base_peak_modal / base_peak_total
            ),
        }
    )


def contribution_at_peak_total(
    resp: dict[str, np.ndarray],
    t: np.ndarray,
):
    def make_table(
        modal: np.ndarray,
        total: np.ndarray,
        quantity: str,
        unit: str,
    ):
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
                "cumulative_fraction_at_peak_total": (
                    np.cumsum(modal_at_peak) / total_at_peak
                ),
            }
        )

    roof_df = make_table(
        resp["roof_modal"],
        resp["roof_total"],
        "roof displacement",
        "m",
    )

    base_df = make_table(
        resp["base_modal"],
        resp["base_total"],
        "base shear",
        "kN",
    )

    return roof_df, base_df


def save_line_plot(
    x: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    y = np.asarray(y)

    if y.ndim == 1:
        ax.plot(x, y, label=labels[0] if labels else None)
    else:
        for j in range(y.shape[1]):
            ax.plot(x, y[:, j], label=labels[j])

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
    damping_ratio: float,
):
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    n_modes = resp["modal_D"].shape[1]
    labels = [f"Mode {j}" for j in range(1, n_modes + 1)]

    save_line_plot(
        t,
        accel_g,
        ["Ground acceleration"],
        "Time (s)",
        "Ground acceleration (g)",
        "Northridge ground acceleration input",
        figdir / "ground_acceleration.png",
    )

    save_line_plot(
        t,
        resp["modal_D"] / IN_TO_M,
        labels,
        "Time (s)",
        "SDF modal deformation Dn (in)",
        f"Modal SDF deformation responses, damping = {100 * damping_ratio:g}%",
        figdir / "modal_sdf_deformations.png",
    )

    save_line_plot(
        t,
        resp["roof_modal"] / IN_TO_M,
        labels,
        "Time (s)",
        "Modal roof displacement contribution (in)",
        f"Modal roof displacement contributions, damping = {100 * damping_ratio:g}%",
        figdir / "modal_roof_displacement_contributions.png",
    )

    save_line_plot(
        t,
        resp["roof_total"] / IN_TO_M,
        ["Total roof displacement"],
        "Time (s)",
        "Roof displacement (in)",
        "Total roof displacement from all 5 modes",
        figdir / "total_roof_displacement.png",
    )

    save_line_plot(
        t,
        resp["base_modal"] / KIP_TO_KN,
        labels,
        "Time (s)",
        "Modal base shear contribution (kip)",
        f"Modal base shear contributions, damping = {100 * damping_ratio:g}%",
        figdir / "modal_base_shear_contributions.png",
    )

    save_line_plot(
        t,
        resp["base_total"] / KIP_TO_KN,
        ["Total base shear"],
        "Time (s)",
        "Base shear (kip)",
        "Total base shear from all 5 modes",
        figdir / "total_base_shear.png",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Part 1 modal analysis for a 5-story shear building."
    )

    parser.add_argument(
        "--record",
        default="RSN942_NORTHR_ALH090.AT2",
        help="Path to the PEER .AT2 acceleration record.",
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
        default=0.05,
        help="Damping ratio for all modes.",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Extra scale factor on ground motion.",
    )

    parser.add_argument(
        "--g",
        type=float,
        default=DEFAULT_G,
        help="m/s^2 per g.",
    )

    parser.add_argument(
        "--outdir",
        default="part1_results",
        help="Output directory.",
    )

    args = parser.parse_args()

    record_path = find_record_file(args.record)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Convert requested structural values to internal units.
    if args.m_tonne is None:
        m_floor_tonne = args.floor_weight_kip * KIP_TO_KN / args.g
    else:
        m_floor_tonne = args.m_tonne

    if args.k_kn_per_m is None:
        k_story_kn_per_m = (
            args.story_stiffness_kip_per_in
            * KIP_TO_KN
            / IN_TO_M
        )
    else:
        k_story_kn_per_m = args.k_kn_per_m

    accel_g_unscaled, dt, meta = read_peer_at2(record_path)

    accel_g = accel_g_unscaled * args.scale
    ag_m_per_s2 = accel_g * args.g

    t = np.arange(accel_g.size) * dt

    n_stories = 5

    m_mat, k_mat = build_shear_building_matrices(
        n_stories,
        m_floor_tonne,
        k_story_kn_per_m,
    )

    props = modal_properties(m_mat, k_mat)

    resp = compute_modal_time_histories(
        ag_m_per_s2,
        dt,
        props,
        args.xi,
        k_mat,
    )

    # -------------------------------------------------------------------------
    # Save matrices and ground motion
    # -------------------------------------------------------------------------

    np.savetxt(
        outdir / "M_matrix_tonne.csv",
        m_mat,
        delimiter=",",
    )

    np.savetxt(
        outdir / "K_matrix_kN_per_m.csv",
        k_mat,
        delimiter=",",
    )

    pd.DataFrame(
        {
            "time_s": t,
            "ag_g_unscaled": accel_g_unscaled,
            "ag_g_after_scale": accel_g,
            "ag_m_per_s2_after_scale": ag_m_per_s2,
        }
    ).to_csv(
        outdir / "ground_motion_used.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save modal properties
    # -------------------------------------------------------------------------

    modal_df = pd.DataFrame(
        {
            "mode": np.arange(1, n_stories + 1),
            "omega_rad_per_s": props["omega"],
            "period_s": props["periods"],
            "modal_mass_tonne": props["modal_mass"],
            "modal_load_tonne": props["modal_load"],
            "Gamma_roof_normalized": props["gamma"],
            "effective_modal_mass_tonne": props["effective_modal_mass"],
            "effective_modal_mass_ratio": props["effective_modal_mass_ratio"],
            "cumulative_effective_mass_ratio": np.cumsum(
                props["effective_modal_mass_ratio"]
            ),
        }
    )

    for floor in range(n_stories):
        modal_df[f"phi_floor_{floor + 1}_roof_norm"] = (
            props["phi_roof"][floor, :]
        )

    modal_df.to_csv(
        outdir / "modal_properties.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save response summary
    # -------------------------------------------------------------------------

    summary_df = pd.DataFrame(
        {
            "quantity": [
                "floor_weight_kip",
                "story_stiffness_kip_per_in",
                "mass_per_floor_tonne_used",
                "story_stiffness_kN_per_m_used",
                "peak_abs_roof_displacement_m",
                "peak_abs_roof_displacement_in",
                "time_of_peak_abs_roof_displacement_s",
                "peak_abs_base_shear_kN",
                "peak_abs_base_shear_kip",
                "time_of_peak_abs_base_shear_s",
                "PGA_g_after_scale",
                "PGA_m_per_s2_after_scale",
            ],
            "value": [
                args.floor_weight_kip,
                args.story_stiffness_kip_per_in,
                m_floor_tonne,
                k_story_kn_per_m,
                np.max(np.abs(resp["roof_total"])),
                np.max(np.abs(resp["roof_total"])) / IN_TO_M,
                t[np.argmax(np.abs(resp["roof_total"]))],
                np.max(np.abs(resp["base_total"])),
                np.max(np.abs(resp["base_total"])) / KIP_TO_KN,
                t[np.argmax(np.abs(resp["base_total"]))],
                np.max(np.abs(accel_g)),
                np.max(np.abs(ag_m_per_s2)),
            ],
        }
    )

    summary_df.to_csv(
        outdir / "response_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save modal contribution tables
    # -------------------------------------------------------------------------

    earthquake_df, chopra_long_df, chopra_wide_df = make_contribution_tables(
        props,
        m_mat,
        k_mat,
    )

    earthquake_df.to_csv(
        outdir / "earthquake_Mr_static_contribution_factors.csv",
        index=False,
    )

    chopra_long_df.to_csv(
        outdir / "chopra_table_12_11_1_sa_sb_long.csv",
        index=False,
    )

    chopra_wide_df.to_csv(
        outdir / "chopra_table_12_11_1_sa_sb_wide.csv",
        index=False,
    )

    chopra_wide_df.to_csv(
        outdir / "chopra_static_contribution_factors.csv",
        index=False,
    )

    dynamic_df = dynamic_peak_contribution_table(resp)

    dynamic_df.to_csv(
        outdir / "time_history_peak_contribution_factors.csv",
        index=False,
    )

    roof_peak_df, base_peak_df = contribution_at_peak_total(resp, t)

    roof_peak_df.to_csv(
        outdir / "roof_contribution_at_peak_total.csv",
        index=False,
    )

    base_peak_df.to_csv(
        outdir / "base_shear_contribution_at_peak_total.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save time histories
    # -------------------------------------------------------------------------

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

    for j in range(n_stories):
        response_df[f"D_mode_{j + 1}_m"] = resp["modal_D"][:, j]
        response_df[f"D_mode_{j + 1}_in"] = (
            resp["modal_D"][:, j] / IN_TO_M
        )

        response_df[f"roof_mode_{j + 1}_m"] = resp["roof_modal"][:, j]
        response_df[f"roof_mode_{j + 1}_in"] = (
            resp["roof_modal"][:, j] / IN_TO_M
        )

        response_df[f"base_mode_{j + 1}_kN"] = resp["base_modal"][:, j]
        response_df[f"base_mode_{j + 1}_kip"] = (
            resp["base_modal"][:, j] / KIP_TO_KN
        )

    response_df.to_csv(
        outdir / "response_time_histories.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save plots
    # -------------------------------------------------------------------------

    save_plots(
        outdir,
        t,
        accel_g,
        resp,
        args.xi,
    )

    # -------------------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------------------

    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.precision", 6)

    print("\nAnalysis complete.")
    print(f"Record used: {record_path}")
    print(f"NPTS = {int(meta['NPTS'])}")
    print(f"DT = {meta['DT']:.5f} s")
    print(f"Floor weight used = {args.floor_weight_kip:g} kip")
    print(f"Mass per floor used = {m_floor_tonne:.6g} tonne")
    print(f"Story stiffness used = {args.story_stiffness_kip_per_in:g} kip/in")
    print(f"Story stiffness used = {k_story_kn_per_m:.6g} kN/m")
    print(f"Damping ratio in all modes = {args.xi:g}")
    print(f"Ground motion scale factor = {args.scale:g}")
    print(f"Outputs written to: {outdir.resolve()}")

    print("\nNatural periods and participation factors:")

    print(
        modal_df[
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

    print("\nResponse summary:")
    print(summary_df.to_string(index=False))

    print("\nEarthquake static contribution factors for s = M r:")

    print(
        earthquake_df[
            [
                "mode",
                "roof_modal_factor",
                "roof_cumulative_factor",
                "base_modal_factor",
                "base_cumulative_factor",
            ]
        ].to_string(index=False)
    )

    print("\nChopra Table 12.11.1-style contribution factors for s_a and s_b:")
    print(chopra_wide_df.to_string(index=False))

    print("\nGenerated files:")

    for file_name in [
        "modal_properties.csv",
        "earthquake_Mr_static_contribution_factors.csv",
        "chopra_table_12_11_1_sa_sb_wide.csv",
        "chopra_table_12_11_1_sa_sb_long.csv",
        "time_history_peak_contribution_factors.csv",
        "roof_contribution_at_peak_total.csv",
        "base_shear_contribution_at_peak_total.csv",
        "response_summary.csv",
        "response_time_histories.csv",
        "figures",
    ]:
        print(f"    {outdir / file_name}")


if __name__ == "__main__":
    main()