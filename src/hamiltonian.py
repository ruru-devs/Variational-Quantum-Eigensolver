"""
hamiltonian.py: Constructs OpenFermion molecular Hamiltonian, Jordan-Wigner qubit operator,
and maps into the particle-number and spin-preserving symmetry sector sparse matrix.
"""

from typing import Tuple
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import openfermion
from openfermion import (
    get_fermion_operator,
    jordan_wigner,
    get_number_preserving_sparse_operator,
)
from src.molecule import MoleculeDataWrapper


class MolecularHamiltonianSector:
    """Represents the molecular Hamiltonian mapped to the spin-preserving particle-number sector."""

    def __init__(self, mol_data: MoleculeDataWrapper):
        self.mol_data = mol_data
        self.n_qubits = mol_data.n_active_qubits
        self.n_electrons = mol_data.n_active_electrons

        # Get active space MolecularOperator from OpenFermion MolecularData
        self.active_ham = mol_data.mol_of.get_molecular_hamiltonian(
            occupied_indices=mol_data.occupied_indices,
            active_indices=mol_data.active_indices,
        )

        # Convert to FermionOperator
        self.fermion_ham = get_fermion_operator(self.active_ham)

        # Qubit operator via Jordan-Wigner
        self.qubit_ham = jordan_wigner(self.fermion_ham)
        self.n_pauli_terms = len(self.qubit_ham.terms)

        # Restrict sparse operator to particle number & spin-preserving sector
        # Determinant basis: reference determinant [1, 0, 0, ...] has first n_electrons occupied
        self.sparse_H = get_number_preserving_sparse_operator(
            self.fermion_ham,
            num_qubits=self.n_qubits,
            num_electrons=self.n_electrons,
            spin_preserving=True,
        )
        self.sector_dim = self.sparse_H.shape[0]

    def compute_ground_state_energy(self) -> Tuple[float, sp.csc_matrix]:
        """Compute the exact ground state eigenvalue in this symmetry sector."""
        val, vec = spla.eigsh(self.sparse_H, k=1, which="SA")
        return float(val[0]), vec[:, 0]
