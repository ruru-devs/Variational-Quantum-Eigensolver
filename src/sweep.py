"""
sweep.py: Geometry sweep and potential energy curve scanner with warm-starting,
automatic checkpointing, CSV writing, and publication-ready plotting.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.molecule import get_molecule, MoleculeDataWrapper
from src.hamiltonian import MolecularHamiltonianSector
from src.ansatz import UCCSDAnsatz
from src.vqe import VQESolver


def run_single_geometry(
    molecule_name: str,
    r: Optional[float] = None,
    angle_deg: Optional[float] = None,
    use_jax: bool = False,
    use_adjoint: bool = False,
    results_dir: str = "results",
    drive_dir: Optional[str] = "/content/drive/MyDrive/mol-simul-VQE",
) -> Dict[str, Any]:
    """Run VQE for a single geometry, save CSV and plot, and return summary dict."""
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "checkpoints"), exist_ok=True)

    t0 = time.time()
    mol = get_molecule(molecule_name, r=r, angle_deg=angle_deg)
    ham_sec = MolecularHamiltonianSector(mol)
    ansatz = UCCSDAnsatz(ham_sec.n_qubits, ham_sec.n_electrons)
    solver = VQESolver(ham_sec, ansatz=ansatz, use_jax=use_jax, use_adjoint=use_adjoint)

    vqe_res = solver.optimize(theta_init=np.zeros(ansatz.n_params))
    wall_s = time.time() - t0

    e_vqe = vqe_res["energy"]
    e_fci = mol.fci_energy
    e_hf = mol.hf_energy
    err_mha = abs(e_vqe - e_fci) * 1000.0

    geom_str = f"R={r:.4f}" if r is not None else ("ang=" + str(angle_deg) if angle_deg else "ref")
    csv_path = os.path.join(results_dir, f"{molecule_name.lower()}_single_{geom_str}.csv")
    df = pd.DataFrame([
        {
            "molecule": molecule_name,
            "geometry": geom_str,
            "R": r if r is not None else np.nan,
            "angle_deg": angle_deg if angle_deg is not None else np.nan,
            "E_vqe": e_vqe,
            "E_fci": e_fci,
            "E_hf": e_hf,
            "abs_error_mHa": err_mha,
            "n_qubits": ham_sec.n_qubits,
            "sector_dim": ham_sec.sector_dim,
            "n_pauli_terms": ham_sec.n_pauli_terms,
            "n_params": ansatz.n_params,
            "iterations": vqe_res["nit"],
            "wall_s": wall_s,
        }
    ])
    df.to_csv(csv_path, index=False)

    # Plot optimization convergence
    fig, ax = plt.subplots(figsize=(6, 4))
    energies = vqe_res["iteration_energies"]
    ax.plot(range(len(energies)), energies, "o-", label="VQE Iteration", color="#1f77b4")
    ax.axhline(e_fci, color="green", linestyle="--", label="Exact FCI")
    ax.axhline(e_hf, color="red", linestyle=":", label="Hartree-Fock")
    ax.set_xlabel("Iteration / Evaluation")
    ax.set_ylabel("Energy (Hartree)")
    ax.set_title(f"{molecule_name} Ground State ({geom_str})\nError: {err_mha:.3f} mHa")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"{molecule_name.lower()}_single_{geom_str}.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    # Sync to drive if available
    if drive_dir and os.path.exists(drive_dir):
        import shutil
        drive_res = os.path.join(drive_dir, "results")
        os.makedirs(drive_res, exist_ok=True)
        shutil.copy(csv_path, drive_res)
        shutil.copy(plot_path, drive_res)

    return {
        "molecule": molecule_name,
        "geometry": geom_str,
        "n_qubits": ham_sec.n_qubits,
        "sector_dim": ham_sec.sector_dim,
        "n_pauli_terms": ham_sec.n_pauli_terms,
        "n_params": ansatz.n_params,
        "E_vqe": e_vqe,
        "E_fci": e_fci,
        "E_hf": e_hf,
        "abs_error_mHa": err_mha,
        "iterations": vqe_res["nit"],
        "wall_s": wall_s,
    }


def run_sweep(
    molecule_name: str,
    r_list: List[float],
    angle_deg: Optional[float] = None,
    use_jax: bool = False,
    use_adjoint: bool = False,
    results_dir: str = "results",
    drive_dir: Optional[str] = "/content/drive/MyDrive/mol-simul-VQE",
) -> pd.DataFrame:
    """Run potential energy curve sweep with warm-start and checkpointing."""
    os.makedirs(results_dir, exist_ok=True)
    ckpt_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_file = os.path.join(ckpt_dir, f"sweep_{molecule_name.lower()}_checkpoint.json")
    results_records: List[Dict[str, Any]] = []
    theta_warm: Optional[np.ndarray] = None

    # Resume from checkpoint if present
    if os.path.exists(ckpt_file):
        try:
            with open(ckpt_file, "r") as f:
                ckpt_data = json.load(f)
                results_records = ckpt_data.get("records", [])
                if ckpt_data.get("last_theta"):
                    theta_warm = np.array(ckpt_data["last_theta"], dtype=float)
                done_r = {round(rec["R"], 4) for rec in results_records}
                r_list = [r for r in r_list if round(r, 4) not in done_r]
                print(f"Resuming {molecule_name} sweep: {len(results_records)} already computed, {len(r_list)} remaining.")
        except Exception as e:
            print(f"Failed to read checkpoint: {e}. Starting fresh.")

    # Re-use ansatz basis matrices across geometries if active space matches
    ansatz: Optional[UCCSDAnsatz] = None

    for idx, r in enumerate(r_list):
        t0 = time.time()
        mol = get_molecule(molecule_name, r=r, angle_deg=angle_deg)
        ham_sec = MolecularHamiltonianSector(mol)

        if ansatz is None:
            ansatz = UCCSDAnsatz(ham_sec.n_qubits, ham_sec.n_electrons)

        solver = VQESolver(ham_sec, ansatz=ansatz, use_jax=use_jax, use_adjoint=use_adjoint)
        vqe_res = solver.optimize(theta_init=theta_warm)
        wall_s = time.time() - t0

        theta_warm = vqe_res["theta"]
        e_vqe = vqe_res["energy"]
        e_fci = mol.fci_energy
        e_hf = mol.hf_energy
        err_mha = abs(e_vqe - e_fci) * 1000.0

        record = {
            "molecule": molecule_name,
            "R": float(r),
            "E_vqe": e_vqe,
            "E_fci": e_fci,
            "E_hf": e_hf,
            "abs_error_mHa": err_mha,
            "n_qubits": ham_sec.n_qubits,
            "sector_dim": ham_sec.sector_dim,
            "n_pauli_terms": ham_sec.n_pauli_terms,
            "n_params": ansatz.n_params,
            "iterations": vqe_res["nit"],
            "wall_s": wall_s,
        }
        results_records.append(record)

        # Write checkpoint after every geometry
        with open(ckpt_file, "w") as f:
            json.dump({"records": results_records, "last_theta": theta_warm.tolist()}, f)

        if drive_dir and os.path.exists(drive_dir):
            import shutil
            drive_ckpt = os.path.join(drive_dir, "checkpoints")
            os.makedirs(drive_ckpt, exist_ok=True)
            shutil.copy(ckpt_file, drive_ckpt)

    # Save final CSV
    df = pd.DataFrame(results_records).sort_values("R").reset_index(drop=True)
    csv_path = os.path.join(results_dir, f"{molecule_name.lower()}_sweep.csv")
    df.to_csv(csv_path, index=False)

    # Generate Publication-Grade Potential Energy Curve & Error Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    # Top: Potential Energy Surface
    ax1.plot(df["R"], df["E_hf"], "r--", label="Hartree-Fock (HF)")
    ax1.plot(df["R"], df["E_fci"], "g-", linewidth=2.0, label="Exact FCI")
    ax1.plot(df["R"], df["E_vqe"], "bo", markersize=4, label="VQE (UCCSD)")
    ax1.set_ylabel("Energy (Hartree)")
    ax1.set_title(f"{molecule_name} Potential Energy Curve (STO-3G)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom: Energy Error with 1.6 mHa chemical accuracy shaded region
    ax2.plot(df["R"], df["abs_error_mHa"], "b.-", label="|E_VQE - E_FCI|")
    ax2.axhspan(0, 1.6, color="green", alpha=0.2, label="Chemical Accuracy (1.6 mHa)")
    ax2.set_xlabel("Bond Distance R (Å)")
    ax2.set_ylabel("Error (mHa)")
    ax2.set_yscale("log")
    ax2.set_ylim(bottom=1e-6, top=max(10.0, df["abs_error_mHa"].max() * 2))
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"{molecule_name.lower()}_sweep.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    # Sync final CSV and plot to drive
    if drive_dir and os.path.exists(drive_dir):
        import shutil
        drive_res = os.path.join(drive_dir, "results")
        os.makedirs(drive_res, exist_ok=True)
        shutil.copy(csv_path, drive_res)
        shutil.copy(plot_path, drive_res)

    return df


def run_2d_sweep(
    molecule_name: str,
    r_list: List[float],
    angle_list: List[float],
    use_jax: bool = False,
    results_dir: str = "results",
    drive_dir: Optional[str] = "/content/drive/MyDrive/mol-simul-VQE",
    max_points: Optional[int] = None,
) -> pd.DataFrame:
    """Run a 2D (bond length, bond angle) PES scan with snake-ordered warm-starting
    and per-geometry checkpointing for resumable execution.
    """
    os.makedirs(results_dir, exist_ok=True)
    ckpt_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_file = os.path.join(ckpt_dir, f"sweep_{molecule_name.lower()}_2d_checkpoint.json")
    results_records: List[Dict[str, Any]] = []
    theta_warm: Optional[np.ndarray] = None

    # Build snake-ordered grid so consecutive points are adjacent
    grid_points = []
    for i, ang in enumerate(angle_list):
        r_seq = r_list if i % 2 == 0 else list(reversed(r_list))
        for r in r_seq:
            grid_points.append((round(float(r), 4), round(float(ang), 2)))

    if os.path.exists(ckpt_file):
        try:
            with open(ckpt_file, "r") as f:
                ckpt_data = json.load(f)
                results_records = ckpt_data.get("records", [])
                if ckpt_data.get("last_theta"):
                    theta_warm = np.array(ckpt_data["last_theta"], dtype=float)
                done_pts = {(round(rec["R"], 4), round(rec["angle_deg"], 2)) for rec in results_records}
                grid_points = [pt for pt in grid_points if pt not in done_pts]
                print(
                    f"Resuming {molecule_name} 2D sweep: "
                    f"{len(results_records)} done, {len(grid_points)} remaining."
                )
        except Exception as e:
            print(f"Failed to read 2D checkpoint: {e}. Starting fresh.")

    if max_points is not None:
        grid_points = grid_points[:max_points]

    ansatz: Optional[UCCSDAnsatz] = None

    for r, ang in grid_points:
        t0 = time.time()
        mol = get_molecule(molecule_name, r=r, angle_deg=ang)
        ham_sec = MolecularHamiltonianSector(mol)

        if ansatz is None:
            ansatz = UCCSDAnsatz(ham_sec.n_qubits, ham_sec.n_electrons)

        solver = VQESolver(ham_sec, ansatz=ansatz, use_jax=use_jax, use_adjoint=use_adjoint)
        vqe_res = solver.optimize(theta_init=theta_warm)
        wall_s = time.time() - t0

        theta_warm = vqe_res["theta"]
        e_vqe = vqe_res["energy"]
        e_fci = mol.fci_energy
        e_hf = mol.hf_energy
        err_mha = abs(e_vqe - e_fci) * 1000.0

        record = {
            "molecule": molecule_name,
            "R": float(r),
            "angle_deg": float(ang),
            "E_vqe": e_vqe,
            "E_fci": e_fci,
            "E_hf": e_hf,
            "abs_error_mHa": err_mha,
            "n_qubits": ham_sec.n_qubits,
            "sector_dim": ham_sec.sector_dim,
            "n_pauli_terms": ham_sec.n_pauli_terms,
            "n_params": ansatz.n_params,
            "iterations": vqe_res["nit"],
            "wall_s": wall_s,
        }
        results_records.append(record)

        with open(ckpt_file, "w") as f:
            json.dump({"records": results_records, "last_theta": theta_warm.tolist()}, f)

        if drive_dir and os.path.exists(drive_dir):
            import shutil
            drive_ckpt = os.path.join(drive_dir, "checkpoints")
            os.makedirs(drive_ckpt, exist_ok=True)
            shutil.copy(ckpt_file, drive_ckpt)

    df = (
        pd.DataFrame(results_records)
        .sort_values(["angle_deg", "R"])
        .reset_index(drop=True)
    )
    csv_path = os.path.join(results_dir, f"{molecule_name.lower()}_2d_sweep.csv")
    df.to_csv(csv_path, index=False)

    # Plot 1: 2D PES contour + angle slices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    piv_vqe = df.pivot(index="angle_deg", columns="R", values="E_vqe")
    if not piv_vqe.isnull().values.any():
        X, Y = np.meshgrid(piv_vqe.columns.values, piv_vqe.index.values)
        cs = ax1.contourf(X, Y, piv_vqe.values, levels=20, cmap="viridis")
        fig.colorbar(cs, ax=ax1, label="VQE Energy (Hartree)")
        min_idx = df["E_vqe"].idxmin()
        ax1.plot(
            df.loc[min_idx, "R"],
            df.loc[min_idx, "angle_deg"],
            "r*",
            markersize=12,
            label=f"Min ({df.loc[min_idx, 'R']:.2f} Å, {df.loc[min_idx, 'angle_deg']:.0f}°)",
        )
        ax1.legend()
    ax1.set_xlabel("O-H Bond Distance R (Å)")
    ax1.set_ylabel("H-O-H Angle (deg)")
    ax1.set_title(f"{molecule_name} 2D VQE Potential Energy Surface")

    # Angle slices (VQE, FCI, HF)
    for ang, grp in df.groupby("angle_deg"):
        ax2.plot(grp["R"], grp["E_fci"], "-", alpha=0.7, label=f"FCI {ang:.0f}°")
        ax2.plot(grp["R"], grp["E_vqe"], "o", markersize=4)
        ax2.plot(grp["R"], grp["E_hf"], "--", alpha=0.4)
    ax2.set_xlabel("O-H Bond Distance R (Å)")
    ax2.set_ylabel("Energy (Hartree)")
    ax2.set_title(f"{molecule_name} PES Slices (Solid: FCI, Dots: VQE, Dashed: HF)")
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"{molecule_name.lower()}_2d_sweep.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    # Plot 2: Error plot with shaded 1.6 mHa chemical accuracy band
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ang, grp in df.groupby("angle_deg"):
        ax.plot(grp["R"], grp["abs_error_mHa"], "o-", markersize=4, label=f"{ang:.0f}°")
    ax.axhspan(0, 1.6, color="green", alpha=0.2, label="Chemical Accuracy (1.6 mHa)")
    ax.set_xlabel("O-H Bond Distance R (Å)")
    ax.set_ylabel("|E_VQE - E_FCI| (mHa)")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-6, top=max(10.0, df["abs_error_mHa"].max() * 2))
    ax.set_title(f"{molecule_name} 2D Grid VQE Error vs Exact FCI")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    err_plot_path = os.path.join(results_dir, f"{molecule_name.lower()}_2d_error.png")
    plt.savefig(err_plot_path, dpi=200)
    plt.close()

    if drive_dir and os.path.exists(drive_dir):
        import shutil
        drive_res = os.path.join(drive_dir, "results")
        os.makedirs(drive_res, exist_ok=True)
        shutil.copy(csv_path, drive_res)
        shutil.copy(plot_path, drive_res)
        shutil.copy(err_plot_path, drive_res)

    return df
