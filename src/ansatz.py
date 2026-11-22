"""
ansatz.py: Spin-conserving UCCSD singles and doubles excitation generators.
Constructs sparse anti-Hermitian operator G(theta) and state exp(G)|HF>.
Ordering:
1. Canonical OpenFermion uccsd_singlet_generator excitation ordering:
   Unique single excitations are ordered before double excitations.
   Singles: a^dag_{p,sigma} a_{q,sigma} - h.c.
   Doubles: a^dag_{p,alpha} a^dag_{q,beta} a_{s,beta} a_{r,alpha} - h.c.
2. Generator basis matrices are precomputed in the symmetry sector sparse representation.
"""

from typing import List
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
import openfermion
from openfermion import (
    uccsd_singlet_generator,
    uccsd_singlet_paramsize,
    get_number_preserving_sparse_operator,
)


class UCCSDAnsatz:
    """Manages UCCSD parameterization, sparse generator bases, and state preparation."""

    def __init__(self, n_qubits: int, n_electrons: int):
        self.n_qubits = n_qubits
        self.n_electrons = n_electrons
        self.n_params = uccsd_singlet_paramsize(n_qubits, n_electrons)

        # Precompute basis sparse matrices in the number and spin-preserving sector
        # Each basis matrix G_k corresponds to unit excitation parameter e_k.
        self.basis_matrices: List[sp.csc_matrix] = []
        for i in range(self.n_params):
            unit_vec = np.zeros(self.n_params, dtype=float)
            unit_vec[i] = 1.0
            gen = uccsd_singlet_generator(
                unit_vec,
                self.n_qubits,
                self.n_electrons,
                anti_hermitian=True,
            )
            gen_sparse = get_number_preserving_sparse_operator(
                gen,
                num_qubits=self.n_qubits,
                num_electrons=self.n_electrons,
                spin_preserving=True,
            )
            self.basis_matrices.append(gen_sparse)

        self.sector_dim = self.basis_matrices[0].shape[0]

        # Hartree-Fock reference state in this sector basis: index 0
        self.hf_state = np.zeros(self.sector_dim, dtype=complex)
        self.hf_state[0] = 1.0

    def get_generator_sparse(self, theta: np.ndarray) -> sp.csc_matrix:
        """Construct the anti-Hermitian generator G = sum_k theta_k * G_k."""
        G = self.basis_matrices[0] * float(theta[0])
        for i in range(1, len(theta)):
            if theta[i] != 0.0:
                G = G + self.basis_matrices[i] * float(theta[i])
        return G

    def prepare_state(self, theta: np.ndarray) -> np.ndarray:
        """Prepares state |psi(theta)> = exp(G(theta)) |HF> using scipy.sparse.linalg.expm_multiply."""
        if np.all(theta == 0.0):
            return self.hf_state.copy()
        G = self.get_generator_sparse(theta)
        return expm_multiply(G, self.hf_state)
