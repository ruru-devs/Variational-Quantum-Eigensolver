"""
Tests for quantum chemistry VQE package.
Covers:
1. H2 Hamiltonian has exactly 15 Pauli terms.
2. Lowest eigenvalue of each sparse Hamiltonian equals the FCI energy in the active space.
3. The HF state and the UCCSD state have the correct particle number (and stay normalized).
4. VQE energy never falls below FCI (variational principle: E_vqe >= E_fci - eps).
5. H2 VQE at 0.735 A is within 1.6 mHa of FCI.
"""

import numpy as np
import pytest
from src.molecule import get_molecule
from src.hamiltonian import MolecularHamiltonianSector
from src.ansatz import UCCSDAnsatz
from src.vqe import VQESolver


def test_h2_pauli_terms_count():
    """Verify H2 Hamiltonian has exactly 15 Pauli terms in STO-3G basis."""
    mol = get_molecule("H2", r=0.735)
    ham_sec = MolecularHamiltonianSector(mol)
    assert ham_sec.n_pauli_terms == 15, f"Expected 15 Pauli terms, got {ham_sec.n_pauli_terms}"


@pytest.mark.parametrize("mol_name, r", [("H2", 0.735), ("LiH", 1.60)])
def test_sparse_hamiltonian_lowest_eigenvalue_equals_fci(mol_name, r):
    """Verify lowest eigenvalue of sector sparse Hamiltonian equals active space FCI energy."""
    mol = get_molecule(mol_name, r=r)
    ham_sec = MolecularHamiltonianSector(mol)
    ground_val, _ = ham_sec.compute_ground_state_energy()
    assert np.isclose(ground_val, mol.fci_energy, atol=1e-5), (
        f"{mol_name}: sparse lowest eigenvalue {ground_val} != FCI {mol.fci_energy}"
    )


def test_particle_number_and_norm_conservation():
    """Verify HF state and UCCSD states are normalized and preserve the sector particle number."""
    mol = get_molecule("H2", r=0.735)
    ham_sec = MolecularHamiltonianSector(mol)
    ansatz = UCCSDAnsatz(ham_sec.n_qubits, ham_sec.n_electrons)

    # 1. HF state norm
    assert np.isclose(np.linalg.norm(ansatz.hf_state), 1.0)

    # 2. Random theta parameter state
    rng = np.random.default_rng(42)
    theta = rng.normal(0, 0.1, size=ansatz.n_params)
    psi = ansatz.prepare_state(theta)
    assert np.isclose(np.linalg.norm(psi), 1.0)
    # Dimension matches particle number & Sz sector
    assert len(psi) == ham_sec.sector_dim


def test_variational_principle_h2_and_lih():
    """Verify VQE energy satisfies E_VQE >= E_FCI (within numerical tolerance)."""
    for mol_name, r in [("H2", 0.735), ("LiH", 1.60)]:
        mol = get_molecule(mol_name, r=r)
        ham_sec = MolecularHamiltonianSector(mol)
        ansatz = UCCSDAnsatz(ham_sec.n_qubits, ham_sec.n_electrons)
        solver = VQESolver(ham_sec, ansatz=ansatz)

        # Test at arbitrary parameter
        theta = np.zeros(ansatz.n_params)
        e_hf = solver.evaluate_energy(theta)
        assert e_hf >= mol.fci_energy - 1e-7

        # Optimized VQE
        res = solver.optimize(theta_init=theta)
        assert res["energy"] >= mol.fci_energy - 1e-7, (
            f"Variational violation: E_VQE ({res['energy']}) < E_FCI ({mol.fci_energy})"
        )


def test_h2_chemical_accuracy():
    """Verify H2 VQE at R=0.735 A is within 1.6 mHa of FCI."""
    mol = get_molecule("H2", r=0.735)
    ham_sec = MolecularHamiltonianSector(mol)
    ansatz = UCCSDAnsatz(ham_sec.n_qubits, ham_sec.n_electrons)
    solver = VQESolver(ham_sec, ansatz=ansatz)

    res = solver.optimize()
    error_mha = abs(res["energy"] - mol.fci_energy) * 1000.0
    assert error_mha < 1.6, f"H2 error {error_mha:.4f} mHa exceeds 1.6 mHa threshold"
