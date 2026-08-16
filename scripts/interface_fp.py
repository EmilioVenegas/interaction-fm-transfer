"""Shared machinery for the interface-fidelity analysis (FP-A, the spatial one).

FP-A is the residue-contact fingerprint: a set of receptor residues with any
heavy atom within a cutoff of any ligand heavy atom. Deliberately dependency-free
beyond rdkit/scipy so it runs in `atomica-interface`; FP-B (ProLIF) lives in the
isolated `ifp` env instead.

Receptors are parsed straight from ATOM records rather than through
`Chem.MolFromPDBFile`, which returned None for 1 of the 44 test receptors and
would have dropped it silently. See gate 7 in
`results/interface_fidelity/ANALYSIS_PLAN.md`.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
from rdkit import Chem, RDLogger
from scipy.spatial import cKDTree

RDLogger.DisableLog("rdApp.*")

CUTOFF = 4.0
RECEPTOR_DIR = "data/receptor_pdbs_test_v2"


@dataclass
class Receptor:
    name: str
    coords: np.ndarray        # (n_atoms, 3), heavy atoms only
    res_key: list             # per atom: (chain, resnum, icode)
    res_name: list            # per atom: residue name
    element: list             # per atom
    tree: cKDTree

    @property
    def n_residues(self) -> int:
        return len(set(self.res_key))


def _element_of(atom_name: str, raw_element: str) -> str:
    e = raw_element.strip()
    if e:
        return e.capitalize()
    # Fall back to the atom name's leading alpha characters (PDB cols 13-16).
    n = atom_name.strip()
    return (n[0] if n else "X").capitalize()


def load_receptor(path: str) -> Receptor:
    """Parse ATOM/HETATM records directly. Heavy atoms only, no sanitisation."""
    coords, res_key, res_name, element = [], [], [], []
    with open(path) as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom_name = line[12:16]
            el = _element_of(atom_name, line[76:78])
            if el == "H":
                continue
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            coords.append(xyz)
            res_key.append((line[21], line[22:26].strip(), line[26]))
            res_name.append(line[17:20].strip())
            element.append(el)
    if not coords:
        raise ValueError(f"no parsable heavy atoms in {path}")
    arr = np.asarray(coords, dtype=float)
    return Receptor(
        name=os.path.basename(path)[:-4],
        coords=arr,
        res_key=res_key,
        res_name=res_name,
        element=element,
        tree=cKDTree(arr),
    )


def heavy_coords(mol: Chem.Mol) -> np.ndarray:
    conf = mol.GetConformer()
    idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    return np.asarray([list(conf.GetAtomPosition(i)) for i in idx], dtype=float)


def fp_a(lig_xyz: np.ndarray, rec: Receptor, cutoff: float = CUTOFF) -> frozenset:
    """Residues with any heavy atom within `cutoff` of any ligand heavy atom."""
    if len(lig_xyz) == 0:
        return frozenset()
    hits = rec.tree.query_ball_point(lig_xyz, cutoff)
    out = set()
    for group in hits:
        for j in group:
            out.add(rec.res_key[j])
    return frozenset(out)


def tanimoto(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return float("nan")
    union = len(a | b)
    return len(a & b) / union if union else float("nan")


def recall(gen: frozenset, ref: frozenset) -> float:
    """Fraction of reference bits recovered. Sparse-robust; see the plan."""
    return len(gen & ref) / len(ref) if ref else float("nan")


def precision(gen: frozenset, ref: frozenset) -> float:
    return len(gen & ref) / len(gen) if gen else float("nan")


def read_sdf(path: str, sanitize: bool = False) -> list:
    """Molecules from an SDF, skipping unreadable records. Never raises on a
    malformed file -- the Uni-Dock run lost 33 minutes of work to exactly that.

    `sanitize=False` is the deliberate default for FP-A. FP-A reads heavy-atom
    *coordinates* only, so valence models are irrelevant to it, and sanitising
    would silently drop molecules on chemistry grounds inside a geometric
    measurement. It really does drop them: the reference ligand of
    complex_000148 carries an N with explicit valence 4 (a charged nitrogen with
    no charge block, a common CrossDocked artefact) and sanitisation rejects the
    whole record.

    Chemical validity is not being waved through -- it is measured separately and
    per arm by gate 6, which is where a differential failure rate becomes a
    finding rather than a silent subsample. FP-B (ProLIF) needs real valences and
    must pass `sanitize=True`.
    """
    try:
        supplier = Chem.SDMolSupplier(path, sanitize=sanitize, removeHs=False)
        return [m for m in supplier if m is not None]
    except OSError:
        return []


def pocket_names(arm_dir: str) -> list:
    return sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{arm_dir}/*.sdf"))


def reference_ligand(pocket: str) -> Chem.Mol:
    path = f"{RECEPTOR_DIR}/{pocket}_ref_ligand.sdf"
    mols = read_sdf(path)
    if not mols:
        raise ValueError(f"unreadable reference ligand: {path}")
    return mols[0]
