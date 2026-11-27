# Simulation Notes & Observations

## $\text{N}_2$ (16 qubits, 2 frozen core orbitals, sector dim = 3,136, 135 UCCSD parameters)

### 1. Accuracy vs. Chemical Accuracy (1.6 mHa)
- **3A Single Geometry ($R = 1.098\text{ \AA}$)**:
  - VQE converged in **12 iterations** (`34.19 s`), reaching $E_{\text{VQE}} = -107.65056190\text{ Ha}$ vs. $E_{\text{FCI}} = -107.65270379\text{ Ha}$.
  - The absolute error is **`2.1419 mHa`**, which is slightly above the `1.6 mHa` chemical accuracy threshold.
- **3B Full Dissociation Sweep ($R = 0.90\text{ to }2.20\text{ \AA}$, step $0.10\text{ \AA}$, 14 points)**:
  - Mean error: **`4.5045 mHa`**; Max error: **`11.8024 mHa`** (at $R = 2.20\text{ \AA}$).
  - Only $R = 0.90\text{ \AA}$ (`1.2265 mHa`) falls under `1.6 mHa`; $R = 1.00\text{ \AA}$ is `1.6348 mHa`, and error grows as the $\text{N}\equiv\text{N}$ triple bond stretches.
- **Physical Explanation**: Breaking a triple bond introduces strong **static (multi-reference) correlation** involving higher-order (triple/quadruple/hextuple) excitations out of the closed-shell RHF determinant. A single-reference UCCSD ansatz (truncated at singles and doubles) cannot capture this multi-reference character at stretched geometries, which is a well-known limitation of UCCSD for $\text{N}_2$.

### 2. Memory & Runtime Feasibility on Colab CPU
- **Dense JAX Autodiff Limitation**: Storing 135 dense $3136 \times 3136$ complex128 generator matrices for JAX autodiff would require ~21.2 GB RAM, exceeding Colab's ~12.7 GB CPU RAM limit.
- **Sparse Adjoint Solution**: We implemented a sparse Duhamel/adjoint gradient (`use_adjoint=True` in `src/vqe.py`) using 8-point Gauss-Legendre quadrature and `scipy.sparse.linalg.expm_multiply`. This kept memory usage to a few MB and completed 3A in `34.19 s` and all 14 points of 3B in `444.70 s` (~7.4 min) via resumable checkpointed batches (`max_points=4`).
