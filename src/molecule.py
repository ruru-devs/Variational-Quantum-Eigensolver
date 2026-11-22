"""
molecule.py: PySCF interface and molecular electronic structure calculation.
Computes HF energy and exact FCI reference energy over the active orbital space using CASCI.
Supports frozen core orbitals:
- H2: 0 frozen core orbitals
- LiH: 1 frozen core orbital (Li 1s)
- H2O: 1 frozen core orbital (O 1s)
- N2: 2 frozen core orbitals (N 1s for both atoms)
"""

from typing import List, Tuple, Optional
import numpy as np
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from pyscf import gto, scf, mcscf


class MoleculeDataWrapper:
    """Wrapper holding molecular data, active space partition, and reference energies."""

    def __init__(
        self,
        geometry: List[Tuple[str, Tuple[float, float, float]]],
        basis: str = "sto-3g",
        multiplicity: int = 1,
        charge: int = 0,
        n_frozen_core: int = 0,
        description: str = "",
    ):
        self.geometry = geometry
        self.basis = basis
        self.multiplicity = multiplicity
        self.charge = charge
        self.n_frozen_core = n_frozen_core
        self.description = description

        # Run OpenFermion-PySCF calculation
        self.mol_of = MolecularData(
            geometry=self.geometry,
            basis=self.basis,
            multiplicity=self.multiplicity,
            charge=self.charge,
            description=self.description,
        )
        self.mol_of = run_pyscf(self.mol_of, run_scf=True, run_fci=False)

        self.hf_energy = float(self.mol_of.hf_energy)
        self.total_orbitals = self.mol_of.n_orbitals
        self.total_electrons = self.mol_of.n_electrons

        # Determine core and active space orbital indices
        self.occupied_indices = list(range(self.n_frozen_core))
        self.active_indices = list(range(self.n_frozen_core, self.total_orbitals))

        self.n_active_orbitals = len(self.active_indices)
        self.n_active_qubits = 2 * self.n_active_orbitals
        self.n_active_electrons = self.total_electrons - 2 * self.n_frozen_core

        # Compute exact FCI energy in the active space via PySCF CASCI
        pyscf_atom = "; ".join(
            f"{atom} {coords[0]} {coords[1]} {coords[2]}"
            for atom, coords in self.geometry
        )
        spin = self.multiplicity - 1
        self.pyscf_mol = gto.M(
            atom=pyscf_atom,
            basis=self.basis,
            charge=self.charge,
            spin=spin,
            verbose=0,
        )
        self.mf = scf.RHF(self.pyscf_mol).run(verbose=0)

        # Number of active electrons in alpha and beta
        nelec_alpha = self.n_active_electrons // 2
        nelec_beta = self.n_active_electrons - nelec_alpha
        self.cas = mcscf.CASCI(
            self.mf,
            ncas=self.n_active_orbitals,
            nelecas=(nelec_alpha, nelec_beta),
        )
        self.cas.run(verbose=0)
        self.fci_energy = float(self.cas.e_tot)


def get_molecule(
    name: str,
    r: Optional[float] = None,
    angle_deg: Optional[float] = None,
) -> MoleculeDataWrapper:
    """Factory helper to build MoleculeDataWrapper with prescribed frozen core orbitals.

    Parameters:
    - name: 'H2', 'LiH', 'H2O', or 'N2'
    - r: bond distance in Angstroms (default: reference geometry)
    - angle_deg: bond angle in degrees (for H2O, default: 104.5)
    """
    mol_name = name.upper()
    if mol_name == "H2":
        dist = 0.735 if r is None else r
        geom = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, dist))]
        return MoleculeDataWrapper(
            geometry=geom,
            basis="sto-3g",
            multiplicity=1,
            charge=0,
            n_frozen_core=0,
            description=f"H2_R{dist:.4f}",
        )
    elif mol_name == "LIH":
        dist = 1.60 if r is None else r
        geom = [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, dist))]
        return MoleculeDataWrapper(
            geometry=geom,
            basis="sto-3g",
            multiplicity=1,
            charge=0,
            n_frozen_core=1,
            description=f"LiH_R{dist:.4f}",
        )
    elif mol_name == "H2O":
        dist = 0.96 if r is None else r
        deg = 104.5 if angle_deg is None else angle_deg
        rad = np.radians(deg / 2.0)
        y = dist * np.sin(rad)
        z = dist * np.cos(rad)
        geom = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, y, z)), ("H", (0.0, -y, z))]
        return MoleculeDataWrapper(
            geometry=geom,
            basis="sto-3g",
            multiplicity=1,
            charge=0,
            n_frozen_core=1,
            description=f"H2O_R{dist:.4f}_deg{deg:.1f}",
        )
    elif mol_name == "N2":
        dist = 1.098 if r is None else r
        geom = [("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, dist))]
        return MoleculeDataWrapper(
            geometry=geom,
            basis="sto-3g",
            multiplicity=1,
            charge=0,
            n_frozen_core=2,
            description=f"N2_R{dist:.4f}",
        )
    else:
        raise ValueError(f"Unknown molecule name: {name}")
