# Variational Quantum Eigensolver (VQE) for Molecular Chemistry

A portfolio-grade quantum chemistry repository simulating ground-state energies and potential energy curves for **H2**, **LiH**, **H2O**, and **N2** using the **Variational Quantum Eigensolver (VQE)** with a **spin-conserving UCCSD ansatz**, validated directly against exact Full Configuration Interaction (**FCI** via CASCI).

All computations are performed using classical CPU simulation with sparse matrices restricted to the particle-number and spin-conserving symmetry sectors.

---

## Computational Methodology

The core logic is modularized under [`src/`](file:///home/Rurushu/Projects/Molecular_Simulation/src):
1. **[`molecule.py`](file:///home/Rurushu/Projects/Molecular_Simulation/src/molecule.py)**: PySCF interface (STO-3G basis). Computes Hartree-Fock (HF) energy and exact active-space FCI reference using CASCI. Enforces frozen cores:
   - $\text{H}_2$: 0 frozen core orbitals (2 active electrons, 2 active spatial orbitals $\to$ 4 spin-orbitals/qubits).
   - $\text{LiH}$: 1 frozen core orbital (Li 1s) (2 active electrons, 5 active spatial orbitals $\to$ 10 qubits).
   - $\text{H}_2\text{O}$: 1 frozen core orbital (O 1s) (8 active electrons, 6 active spatial orbitals $\to$ 12 qubits).
   - $\text{N}_2$: 2 frozen core orbitals (N 1s) (10 active electrons, 8 active spatial orbitals $\to$ 16 qubits).
2. **[`hamiltonian.py`](file:///home/Rurushu/Projects/Molecular_Simulation/src/hamiltonian.py)**: OpenFermion molecular Hamiltonian and Jordan-Wigner transformation mapped directly into the fixed particle number and spin projection ($S_z = 0$) sector. Sector dimension for $\text{N}_2$ is reduced from $2^{16} = 65,536$ down to $3,136$.
3. **[`ansatz.py`](file:///home/Rurushu/Projects/Molecular_Simulation/src/ansatz.py)**: Spin-conserving UCCSD single and double excitation generators ($T - T^\dagger$) precomputed in the symmetry sector basis. Exponential state preparation via `scipy.sparse.linalg.expm_multiply`.
4. **[`vqe.py`](file:///home/Rurushu/Projects/Molecular_Simulation/src/vqe.py)**: Energy evaluation $\langle\psi(\theta)|H|\psi(\theta)\rangle$ via sparse matrix-vector products optimized with SciPy `L-BFGS-B`, supporting both finite-difference and JAX adjoint gradients.
5. **[`sweep.py`](file:///home/Rurushu/Projects/Molecular_Simulation/src/sweep.py)**: Geometry potential energy scans with warm-starting from previous optimal parameter vectors, automatic checkpointing after every geometry, CSV generation, and publication-ready dual-panel plots.

---

## Phase 1 Results Summary

### Reference Geometries (Single Geometry Runs)

| Molecule | Geometry | Qubits | Sector Dim | Pauli Terms | Parameters | Iterations | Wall Time (s) | FCI Energy (Ha) | VQE Energy (Ha) | Error (mHa) | Chemical Accuracy (< 1.6 mHa) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$\text{H}_2$** | $R = 0.735\text{ \AA}$ | 4 | 4 | 15 | 2 | 3 | 0.15s | -1.13730604 | -1.13730604 | $8.88 \times 10^{-10}$ | :white_check_mark: |
| **$\text{LiH}$** | $R = 1.600\text{ \AA}$ | 10 | 25 | 276 | 14 | 10 | 0.85s | -7.88209660 | -7.88209660 | $5.91 \times 10^{-6}$ | :white_check_mark: |
| **$\text{H}_2\text{O}$** | $R = 0.960\text{ \AA}, 104.5^\circ$ | 12 | 225 | 551 | 44 | 11 | 5.95s | -75.01307689 | -75.01297675 | 0.1001 | :white_check_mark: |

### Potential Energy Sweeps

1. **$\text{H}_2$ Dissociation Curve ($R = 0.30\text{ \AA}$ to $2.50\text{ \AA}$, step $0.05\text{ \AA}$, 45 points)**:
   - **Mean Error**: $0.0000\text{ mHa}$
   - **Max Error**: $0.0000\text{ mHa}$ (exact match with FCI across all points)
   - **Total Wall Time**: $5.38\text{ s}$
   - Results: [`results/h2_sweep.csv`](file:///home/Rurushu/Projects/Molecular_Simulation/results/h2_sweep.csv), [`results/h2_sweep.png`](file:///home/Rurushu/Projects/Molecular_Simulation/results/h2_sweep.png)

2. **$\text{LiH}$ Dissociation Curve ($R = 0.80\text{ \AA}$ to $3.50\text{ \AA}$, step $0.10\text{ \AA}$, 28 points)**:
   - **Mean Error**: $0.0000\text{ mHa}$
   - **Max Error**: $0.0000\text{ mHa}$ (all well below $10^{-5}\text{ mHa}$)
   - **Total Wall Time**: $28.42\text{ s}$
   - Results: [`results/lih_sweep.csv`](file:///home/Rurushu/Projects/Molecular_Simulation/results/lih_sweep.csv), [`results/lih_sweep.png`](file:///home/Rurushu/Projects/Molecular_Simulation/results/lih_sweep.png)

All sweeps feature checkpointing in [`results/checkpoints/`](file:///home/Rurushu/Projects/Molecular_Simulation/results/checkpoints) and are synced to `/content/drive/MyDrive/chem-vqe/`.

---

## Running the Code

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Tests
```bash
pytest tests/ -v
```

All 6 test suites verify:
- $\text{H}_2$ Hamiltonian has exactly 15 Pauli terms.
- Lowest eigenvalue of the symmetry sector sparse Hamiltonian equals active-space FCI.
- HF and UCCSD state normalization and particle number preservation.
- Strict variational lower bound compliance ($E_{\text{VQE}} \ge E_{\text{FCI}}$).
- $\text{H}_2$ VQE within 1.6 mHa chemical accuracy.
