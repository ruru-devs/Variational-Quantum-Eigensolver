"""
vqe.py: Variational Quantum Eigensolver optimization engine.
Evaluates energy E(theta) = <psi|H|psi> via sparse matrix-vector products.
Supports:
- Finite-difference gradient (default for cheap systems like H2, LiH)
- JAX automatic differentiation / dense adjoint gradient for systems where scaling is critical
- Tracks energy and parameter history per iteration
"""

import time
from typing import Dict, Any, Optional, List
import numpy as np
from scipy.optimize import minimize
import scipy.sparse as sp

from src.ansatz import UCCSDAnsatz
from src.hamiltonian import MolecularHamiltonianSector


class VQESolver:
    """Solves the ground state energy using VQE with L-BFGS-B."""

    def __init__(
        self,
        hamiltonian_sector: MolecularHamiltonianSector,
        ansatz: Optional[UCCSDAnsatz] = None,
        use_jax: bool = False,
    ):
        self.ham_sec = hamiltonian_sector
        self.n_qubits = self.ham_sec.n_qubits
        self.n_electrons = self.ham_sec.n_electrons

        if ansatz is None:
            self.ansatz = UCCSDAnsatz(self.n_qubits, self.n_electrons)
        else:
            self.ansatz = ansatz

        self.H_sparse = self.ham_sec.sparse_H
        self.n_params = self.ansatz.n_params
        self.use_jax = use_jax

        if self.use_jax:
            self._setup_jax()

    def _setup_jax(self):
        """Prepare JAX autodiff graph for exact gradient evaluation.
        Caches the JIT-compiled function on self.ansatz so geometry sweeps
        reuse the compiled XLA kernel without recompiling per geometry.
        """
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        if not hasattr(self.ansatz, "_jax_val_and_grad"):
            basis_jax = jnp.array(
                [m.toarray() for m in self.ansatz.basis_matrices],
                dtype=jnp.complex128,
            )
            hf_jax = jnp.zeros(self.ansatz.sector_dim, dtype=jnp.complex128).at[0].set(1.0)

            def jax_loss(theta, H_mat):
                G = jnp.tensordot(theta, basis_jax, axes=1)
                U = jax.scipy.linalg.expm(G)
                psi = U @ hf_jax
                return jnp.real(jnp.vdot(psi, H_mat @ psi))

            self.ansatz._jax_val_and_grad = jax.jit(jax.value_and_grad(jax_loss))

        self._H_jax = jnp.array(self.H_sparse.toarray(), dtype=jnp.complex128)
        self._jax_val_and_grad = self.ansatz._jax_val_and_grad

    def evaluate_energy(self, theta: np.ndarray) -> float:
        """Evaluate <psi(theta)| H |psi(theta)>."""
        psi = self.ansatz.prepare_state(theta)
        H_psi = self.H_sparse.dot(psi)
        return float(np.vdot(psi, H_psi).real)

    def optimize(
        self,
        theta_init: Optional[np.ndarray] = None,
        maxiter: int = 150,
        gtol: float = 1e-6,
        ftol: float = 1e-9,
    ) -> Dict[str, Any]:
        """Run L-BFGS-B optimization."""
        if theta_init is None:
            theta0 = np.zeros(self.n_params, dtype=float)
        else:
            theta0 = np.array(theta_init, dtype=float).copy()

        iteration_energies: List[float] = []
        start_time = time.time()

        if self.use_jax:
            import jax.numpy as jnp

            def objective(theta_vec):
                val, grad = self._jax_val_and_grad(jnp.array(theta_vec), self._H_jax)
                iteration_energies.append(float(val))
                return float(val), np.array(grad, dtype=float)

            res = minimize(
                fun=objective,
                x0=theta0,
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": maxiter, "gtol": gtol, "ftol": ftol},
            )
        else:
            def objective(theta_vec):
                E = self.evaluate_energy(theta_vec)
                return E

            def callback(intermediate_theta):
                E = self.evaluate_energy(intermediate_theta)
                iteration_energies.append(E)

            iteration_energies.append(self.evaluate_energy(theta0))

            res = minimize(
                fun=objective,
                x0=theta0,
                method="L-BFGS-B",
                callback=callback,
                options={"maxiter": maxiter, "gtol": gtol, "ftol": ftol},
            )

        wall_time = time.time() - start_time
        optimal_energy = float(res.fun)
        optimal_theta = np.array(res.x, dtype=float)

        return {
            "energy": optimal_energy,
            "theta": optimal_theta,
            "nit": int(res.nit),
            "nfev": int(res.nfev),
            "success": bool(res.success),
            "message": str(res.message),
            "wall_time": wall_time,
            "iteration_energies": iteration_energies,
        }
