"""Validity gates for the interface-fidelity analysis -- FP-B (chemical, ProLIF).

Companion to `scripts/interface_fidelity_gates.py` (FP-A, spatial). Runs in the
isolated `ifp` conda env (prolif, MDAnalysis, rdkit, pdb2pqr -- no torch), because
FP-B needs real interaction typing (H-bond donor/acceptor, pi-stacking, salt
bridges) that FP-A's pure-distance fingerprint does not attempt. See gate 6 and
the FP-B section of `results/interface_fidelity/ANALYSIS_PLAN.md`.

Gates covered: 1 (placement null), 2 (dynamic range), 3 (ceiling), 6
(protonation), plus the FP-B seed-noise floors at both scales and the
promiscuity guard (mean interaction count per molecule). Gates 4/5/7 (size
matching, same pockets, no silent drops) are re-asserted here rather than
imported, because FP-B has its own failure modes (protonation, sanitisation)
that gate 7 in the FP-A script does not cover.

BLINDING: `results/gen_critic_lambda20_r0` is the treatment arm of the primary
comparison this analysis feeds. It is never protonated, fingerprinted, or
scored here -- it appears only in the gate-4-style heavy-atom-count listing,
which is the one exception the task spec carves out. Per the task instructions
governing this script, no control-vs-critic delta of any kind is computed or
printed anywhere below; every arm's numbers are reported side by side and left
for a human to read, not diffed in code. (The seed-noise floors are an
exception in name only -- they compare seeds *within* one arm type against
itself, control-vs-control and critic_l0.7-vs-critic_l0.7, never control against
critic.)

    ~/.conda/envs/ifp/bin/python scripts/interface_fidelity_fpb.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

import MDAnalysis as mda  # noqa: E402
import prolif  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from interface_fp import (  # noqa: E402
    RECEPTOR_DIR,
    heavy_coords,
    pocket_names,
    read_sdf,
    reference_ligand,
    recall,
    tanimoto,
)

OUT = Path("results/interface_fidelity")
PROTONATED_DIR = Path("data/protonated_receptors")
ATOMICA_PYTHON = os.path.expanduser("~/.conda/envs/atomica-interface/bin/python")
LMDB_PATH = os.path.abspath("data/crossdocked_pocket10_processed.lmdb")
# Resolve pdb2pqr30 next to this interpreter -- subprocess does not inherit the
# activated env's PATH the way an interactive shell would.
PDB2PQR = str(Path(sys.executable).parent / "pdb2pqr30")

# The six lambda=0.7 arms. critic_lambda20_r0 is excluded on purpose -- see the
# blinding note above and the 2026-08-16 amendment in ANALYSIS_PLAN.md.
CONTROL_ARMS = [f"results/gen_critic_control_r{i}" for i in range(3)]
CRITIC_ARMS = [f"results/gen_critic_graph_cosine_r{i}" for i in range(3)]
GATE_ARMS = CONTROL_ARMS + CRITIC_ARMS
HELD_OUT = "results/gen_critic_lambda20_r0"

N_NULL_PER_POCKET = 100      # placement-null draws
DISPLACEMENTS = [0.5, 1.0, 2.0, 3.0]
N_DIRECTIONS = 12            # random unit vectors per displacement
SAME_SITE_CUTOFF = 8.0       # centroid offset for a valid ceiling pair
SEED = 0
PH = 7.4

# prolif's default interaction set (Fingerprint(interactions=None)) is already
# {Anionic, CationPi, Cationic, HBAcceptor, HBDonor, Hydrophobic, PiCation,
# PiStacking, VdWContact} -- a strict superset of the plan's named list
# (Hydrophobic, HBDonor, HBAcceptor, PiStacking, CationPi, PiCation, Anionic,
# Cationic). "Default set plus" those eight therefore resolves to the default
# set, so the Fingerprint is built with interactions=None rather than a
# hand-typed list that would just reproduce it minus VdWContact.
FP = prolif.Fingerprint()
INTERACTION_NAMES = list(FP.interactions.keys())


# --------------------------------------------------------------------------- #
# Chemistry helpers
# --------------------------------------------------------------------------- #

def _repair_valence(mol: Chem.Mol):
    """One repair pass for the common CrossDocked artefact: an over-valent N/O
    with no formal-charge block (the same failure mode documented in
    `interface_fp.read_sdf`'s docstring for complex_000148's reference ligand,
    which FP-A dodges by never sanitizing and FP-B cannot). Bumps formal charge
    on every atom RDKit flags with AtomValenceException by +1 and retries once.
    Returns the sanitized mol, or None if that does not fix it.
    """
    try:
        problems = Chem.DetectChemistryProblems(mol)
    except Exception:
        return None
    fixed_any = False
    for p in problems:
        if p.GetType() == "AtomValenceException":
            atom = mol.GetAtomWithIdx(p.GetAtomIdx())
            atom.SetFormalCharge(atom.GetFormalCharge() + 1)
            fixed_any = True
    if not fixed_any:
        return None
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def read_sdf_fpb(path: str):
    """Read an SDF for FP-B: every record needs real valences and explicit H,
    the opposite of FP-A's convention (`interface_fp.read_sdf`, sanitize=False
    by default, geometry only). Every molecule lost is counted under exactly one
    of two buckets -- fail_read_sanitize or fail_addhs -- rather than silently
    absorbed the way `Chem.SDMolSupplier` would absorb it.

    Splits on the SDF record delimiter directly (rather than SDMolSupplier) so
    `attempted` is the true record count, including ones too broken to parse at
    all -- SDMolSupplier can silently swallow those before `len()` sees them.
    """
    try:
        text = Path(path).read_text()
    except OSError:
        return [], {"attempted": 0, "fail_read_sanitize": 0, "fail_addhs": 0,
                    "repaired": 0}
    blocks = [b for b in text.split("$$$$\n") if b.strip()]
    mols = []
    fail_rs = fail_ah = repaired_n = 0
    for b in blocks:
        m = Chem.MolFromMolBlock(b, sanitize=False, removeHs=False)
        if m is None:
            fail_rs += 1
            continue
        try:
            Chem.SanitizeMol(m)
        except Exception:
            # SanitizeMol can partially mutate on a failed pass; reparse clean.
            m = Chem.MolFromMolBlock(b, sanitize=False, removeHs=False)
            fixed = _repair_valence(m) if m is not None else None
            if fixed is None:
                fail_rs += 1
                continue
            m = fixed
            repaired_n += 1
        try:
            m = Chem.AddHs(m, addCoords=True)
        except Exception:
            fail_ah += 1
            continue
        mols.append(m)
    counts = {"attempted": len(blocks), "fail_read_sanitize": fail_rs,
              "fail_addhs": fail_ah, "repaired": repaired_n}
    return mols, counts


def heavy_centroid(mol: Chem.Mol) -> np.ndarray:
    conf = mol.GetConformer()
    pts = [conf.GetAtomPosition(a.GetIdx()) for a in mol.GetAtoms()
           if a.GetAtomicNum() > 1]
    return np.asarray([[p.x, p.y, p.z] for p in pts], dtype=float)


def translate_mol(mol: Chem.Mol, delta: np.ndarray) -> Chem.Mol:
    """Rigid translation of every atom (heavy + H). Never GetBestRMS -- see the
    plan's gate-2 note; that would superimpose the displacement away."""
    m = Chem.Mol(mol)
    conf = m.GetConformer()
    for i in range(m.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (p.x + delta[0], p.y + delta[1], p.z + delta[2]))
    return m


BOND_MAP = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC}


def _build_mol_from_bonds(pos, elements, bond_index, bond_type) -> Chem.Mol:
    mol = Chem.RWMol()
    for z in elements:
        mol.AddAtom(Chem.Atom(int(z)))
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, xyz in enumerate(pos):
        conf.SetAtomPosition(i, [float(c) for c in xyz])
    bidx = np.asarray(bond_index, dtype=int)
    btype = np.asarray(bond_type, dtype=int)
    seen = set()
    for k in range(bidx.shape[1]):
        i, j = int(bidx[0, k]), int(bidx[1, k])
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        bt = BOND_MAP.get(int(btype[k]))
        if bt is None:
            raise ValueError(f"unknown_bond_type_{btype[k]}")
        mol.AddBond(i, j, bt)
        if bt == Chem.BondType.AROMATIC:
            mol.GetAtomWithIdx(i).SetIsAromatic(True)
            mol.GetAtomWithIdx(j).SetIsAromatic(True)
            mol.GetBondBetweenAtoms(i, j).SetIsAromatic(True)
    mol.AddConformer(conf, assignId=True)
    return mol.GetMol()


def mol_from_lmdb_record(rec: dict) -> Chem.Mol:
    """Build an explicit-H RDKit mol from an LMDB ligand record's bond table.
    LMDB ligands carry no hydrogens and no dependable bond orders on their own
    (see the task spec) -- this only works because CrossDocked's preprocessing
    stored `ligand_bond_index`/`ligand_bond_type` alongside the coordinates.
    Raises ValueError, never returns None, so the caller can count *why*.
    """
    if "error" in rec:
        raise ValueError(rec["error"])
    if rec.get("bond_index") is None or rec.get("bond_type") is None:
        raise ValueError("no_bond_table")
    m = _build_mol_from_bonds(rec["pos"], rec["element"], rec["bond_index"],
                               rec["bond_type"])
    try:
        Chem.SanitizeMol(m)
    except Exception:
        m = _build_mol_from_bonds(rec["pos"], rec["element"], rec["bond_index"],
                                   rec["bond_type"])
        fixed = _repair_valence(m)
        if fixed is None:
            raise ValueError("sanitize_failed")
        m = fixed
    return Chem.AddHs(m, addCoords=True)


# --------------------------------------------------------------------------- #
# Fingerprinting
# --------------------------------------------------------------------------- #

def ifp_to_frozenset(ifp_dict) -> frozenset:
    """(protein residue string, interaction name) tuples -- composes directly
    with `interface_fp.tanimoto`/`recall`, which only need set operations."""
    out = set()
    for (_lig_res, prot_res), bits in ifp_dict.items():
        for i, name in enumerate(INTERACTION_NAMES):
            if bits[i]:
                out.add((str(prot_res), name))
    return frozenset(out)


def compute_fp(mol_h: Chem.Mol, prot_mol) -> frozenset:
    lig = prolif.Molecule.from_rdkit(mol_h)
    ifp = FP.generate(lig, prot_mol, residues=None)
    return ifp_to_frozenset(ifp)


# --------------------------------------------------------------------------- #
# LMDB access via a torch-capable subprocess (this env has no torch/lmdb)
# --------------------------------------------------------------------------- #

_LMDB_EXTRACT_CODE = r"""
import sys, json, pickle, lmdb

def as_list(v):
    return v.tolist() if hasattr(v, "tolist") else list(v)

req = json.loads(sys.stdin.read())
env = lmdb.open(req["lmdb_path"], readonly=True, lock=False, subdir=False)
out = {}
with env.begin() as txn:
    for k in req["keys"]:
        raw = txn.get(k.encode())
        if raw is None:
            out[k] = {"error": "missing_key"}
            continue
        try:
            rec = pickle.loads(raw)
            entry = {"pos": as_list(rec["ligand_pos"]),
                     "element": as_list(rec["ligand_element"])}
            if "ligand_bond_index" in rec and "ligand_bond_type" in rec:
                entry["bond_index"] = as_list(rec["ligand_bond_index"])
                entry["bond_type"] = as_list(rec["ligand_bond_type"])
            else:
                entry["bond_index"] = None
                entry["bond_type"] = None
            out[k] = entry
        except Exception as exc:
            out[k] = {"error": repr(exc)}
print(json.dumps(out))
"""


def extract_lmdb_records(keys: list) -> dict:
    """Reads torch-pickled LMDB records via a subprocess in the atomica-interface
    env, which has both `lmdb` and `torch` (unpickling the records' tensors
    needs torch importable in the process doing it). This script's own
    interpreter never imports torch -- the read happens in a fully separate
    process and only plain JSON-safe lists cross back over stdout.
    """
    if not keys:
        return {}
    proc = subprocess.run(
        [ATOMICA_PYTHON, "-c", _LMDB_EXTRACT_CODE],
        input=json.dumps({"lmdb_path": LMDB_PATH, "keys": keys}),
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"lmdb extraction subprocess failed:\n{proc.stderr[-4000:]}")
    return json.loads(proc.stdout)


# --------------------------------------------------------------------------- #
# Receptor protonation (gate 6, receptor half)
# --------------------------------------------------------------------------- #

def protonate_receptor(pocket: str):
    """pdb2pqr30 at pH 7.4, cached under PROTONATED_DIR, reused across every
    arm. Returns (path_or_None, stderr_tail_or_None)."""
    out_pdb = PROTONATED_DIR / f"{pocket}.pdb"
    if out_pdb.exists():
        return out_pdb, None
    PROTONATED_DIR.mkdir(parents=True, exist_ok=True)
    pqr_path = PROTONATED_DIR / f"{pocket}.pqr"
    src = Path(RECEPTOR_DIR) / f"{pocket}.pdb"
    cmd = [PDB2PQR, "--ff", "AMBER", "--with-ph", str(PH),
           "--titration-state-method", "propka", "--keep-chain",
           "--pdb-output", str(out_pdb), str(src), str(pqr_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out_pdb.exists():
        return None, proc.stderr[-2000:]
    return out_pdb, None


def _guess_bonds_h_monovalent(u) -> list:
    """MDAnalysis' distance-only bond guesser (no CONECT records exist in a
    pdb2pqr PDB) spuriously double- or triple-bonds some hydrogens on compact
    aromatic residues -- observed here on indole (TRP) ring hydrogens, which
    sit close enough to two ring carbons to pass the distance cutoff twice.
    Hydrogen is monovalent in every organic/protein context with no exception,
    so this is a safe, general repair: guess bonds normally, then for every H
    with more than one guessed bond keep only its nearest neighbour.
    """
    from collections import defaultdict

    from MDAnalysis.topology import guessers

    bonds = guessers.guess_bonds(u.atoms, u.atoms.positions)
    positions = u.atoms.positions
    types = u.atoms.types
    per_atom = defaultdict(list)
    for i, j in bonds:
        per_atom[i].append(j)
        per_atom[j].append(i)
    keep = {(min(i, j), max(i, j)) for i, j in bonds}
    for idx, neighbours in per_atom.items():
        if types[idx] == "H" and len(neighbours) > 1:
            ordered = sorted(
                neighbours,
                key=lambda n: np.linalg.norm(positions[idx] - positions[n]))
            for n in ordered[1:]:
                keep.discard((min(idx, n), max(idx, n)))
    return sorted(keep)


def load_receptor_prolif(pocket: str):
    """Protonate (or reuse cache) then wrap as a prolif.Molecule. Tries default
    bond guessing first; on a valence error, retries once with the H-monovalent
    repair above before giving up and naming the pocket as failed."""
    pdb_path, err = protonate_receptor(pocket)
    if pdb_path is None:
        return None, f"pdb2pqr30_failed: {err}"
    try:
        u = mda.Universe(str(pdb_path))
        return prolif.Molecule.from_mda(u), None
    except Exception as first_exc:
        try:
            u = mda.Universe(str(pdb_path))
            u.add_bonds(_guess_bonds_h_monovalent(u))
            return prolif.Molecule.from_mda(u), None
        except Exception as exc:
            return None, (f"prolif_load_failed: {exc!r} "
                          f"(first attempt: {first_exc!r})")


# --------------------------------------------------------------------------- #
# Setup: receptors + references, 44/44 asserted
# --------------------------------------------------------------------------- #

def load_all():
    pockets = pocket_names(GATE_ARMS[0])
    assert len(pockets) == 44, f"expected 44 pockets, got {len(pockets)}"

    receptors, rec_fail = {}, []
    for p in pockets:
        mol, err = load_receptor_prolif(p)
        if mol is None:
            rec_fail.append((p, err))
        else:
            receptors[p] = mol
    print(f"[gate 6/7] receptors protonated+loaded: {len(receptors)}/{len(pockets)}"
          f"   failures: {rec_fail}")
    assert not rec_fail, (
        "receptor protonation failed for pockets named above -- fix pdb2pqr30 "
        "input/flags, do not skip pockets")

    # Reference ligands: geometry (FP-A style, heavy atoms, no sanitize -- used
    # for centroid/displacement math) and chemistry (FP-B style, explicit H,
    # used for fingerprinting) are loaded separately and cross-checked, per the
    # complex_000148 caveat in interface_fp.read_sdf's docstring: sanitizing a
    # CrossDocked reference ligand can fail even when the coordinates are fine.
    refs_geom = {p: reference_ligand(p) for p in pockets}  # raises if any missing
    ref_centroid = {p: heavy_coords(refs_geom[p]).mean(0) for p in pockets}

    refs_mol_h, ref_fp, ref_fail = {}, {}, []
    for p in pockets:
        mols, counts = read_sdf_fpb(f"{RECEPTOR_DIR}/{p}_ref_ligand.sdf")
        if not mols:
            ref_fail.append((p, counts))
            continue
        m = mols[0]
        try:
            fp = compute_fp(m, receptors[p])
        except Exception as exc:
            ref_fail.append((p, f"fingerprint_failed: {exc!r}"))
            continue
        refs_mol_h[p] = m
        ref_fp[p] = fp
    usable_pockets = [p for p in pockets if p in ref_fp]
    print(f"[gate 6/7] reference ligands fingerprinted: "
          f"{len(usable_pockets)}/{len(pockets)}   failures: {ref_fail}")
    ref_bits = np.array([len(ref_fp[p]) for p in usable_pockets])
    print(f"           FP-B reference bits: med {np.median(ref_bits):.0f} "
          f"p10 {np.percentile(ref_bits,10):.0f} min {ref_bits.min()} "
          f"max {ref_bits.max()}")
    return pockets, usable_pockets, receptors, refs_geom, ref_centroid, \
        refs_mol_h, ref_fp, rec_fail, ref_fail


# --------------------------------------------------------------------------- #
# Per-arm processing: protonate/fingerprint every ligand, gate 6 ligand half
# --------------------------------------------------------------------------- #

def process_arm(arm: str, pockets, usable_pockets, receptors, ref_fp):
    """Returns:
      per_pocket   : pocket -> np.ndarray (n_mol, 2) columns [tanimoto, recall]
      n_interact   : pocket -> list[int] FP-B interaction count per molecule
      pool_mols    : pocket -> list[Chem.Mol] (explicit-H, un-translated) --
                     every molecule that made it through protonation, used for
                     the gate-1 placement-null pool regardless of whether its
                     own pocket has a usable reference fingerprint
      counts       : {"attempted","fail_read_sanitize","fail_addhs","repaired"}
      fp_failures  : [(pocket, mol_index, reason), ...] -- named, not dropped
    """
    arm_pockets = pocket_names(arm)
    assert arm_pockets == pockets, (
        f"[gate 5] {arm} pocket set differs from the reference set")

    per_pocket, n_interact, pool_mols = {}, {}, {}
    counts = {"attempted": 0, "fail_read_sanitize": 0, "fail_addhs": 0,
              "repaired": 0}
    fp_failures = []
    for p in pockets:
        mols, c = read_sdf_fpb(f"{arm}/{p}.sdf")
        for k, v in c.items():
            counts[k] += v
        vals, n_int, kept = [], [], []
        for i, m in enumerate(mols):
            try:
                f = compute_fp(m, receptors[p])
            except Exception as exc:
                fp_failures.append((p, i, repr(exc)))
                continue
            kept.append(m)
            n_int.append(len(f))
            if p in ref_fp:
                vals.append((tanimoto(f, ref_fp[p]), recall(f, ref_fp[p])))
        per_pocket[p] = np.asarray(vals, dtype=float) if vals else np.empty((0, 2))
        n_interact[p] = n_int
        pool_mols[p] = kept
    return per_pocket, n_interact, pool_mols, counts, fp_failures


# --------------------------------------------------------------------------- #
# Gate 1: placement null
# --------------------------------------------------------------------------- #

def gate_1_placement_null(pockets, usable_pockets, receptors, ref_centroid,
                          ref_fp, pool_mols, rng):
    rows = []
    all_pairs = [(p, i) for p in pockets for i in range(len(pool_mols[p]))]
    for p in usable_pockets:
        target = ref_centroid[p]
        pool = [q for q in all_pairs if q[0] != p]
        picks = rng.choice(len(pool), size=min(N_NULL_PER_POCKET, len(pool)),
                           replace=False)
        for k in picks:
            q, i = pool[k]
            mol = pool_mols[q][i]
            delta = target - heavy_centroid(mol).mean(0)
            moved = translate_mol(mol, delta)
            try:
                f = compute_fp(moved, receptors[p])
            except Exception as exc:
                rows.append((p, float("nan"), float("nan"), repr(exc)))
                continue
            rows.append((p, tanimoto(f, ref_fp[p]), recall(f, ref_fp[p]), None))
    return pd.DataFrame(rows, columns=["pocket", "tanimoto", "recall", "error"])


# --------------------------------------------------------------------------- #
# Gate 2: dynamic range
# --------------------------------------------------------------------------- #

def gate_2_dynamic_range(usable_pockets, receptors, refs_mol_h, ref_fp, rng):
    rows = []
    for p in usable_pockets:
        base = heavy_centroid(refs_mol_h[p])
        for d in DISPLACEMENTS:
            for _ in range(N_DIRECTIONS):
                v = rng.normal(size=3)
                v /= np.linalg.norm(v)
                delta = v * d
                moved = translate_mol(refs_mol_h[p], delta)
                rmsd = float(np.sqrt(((heavy_centroid(moved) - base) ** 2)
                                     .sum(1).mean()))
                assert abs(rmsd - d) < 1e-6, (rmsd, d)
                try:
                    f = compute_fp(moved, receptors[p])
                except Exception as exc:
                    rows.append((p, d, float("nan"), float("nan"), repr(exc)))
                    continue
                rows.append((p, d, tanimoto(f, ref_fp[p]), recall(f, ref_fp[p]),
                             None))
    return pd.DataFrame(rows, columns=["pocket", "displacement", "tanimoto",
                                       "recall", "error"])


# --------------------------------------------------------------------------- #
# Gate 3: ceiling
# --------------------------------------------------------------------------- #

def gate_3_ceiling(usable_pockets, receptors, ref_centroid, ref_fp):
    pairs = json.loads(Path(OUT / "ceiling_pairs.json").read_text())

    candidates = []  # (pocket, code, lmdb_key)
    for p in usable_pockets:
        info = pairs.get(p, {})
        for code, meta in info.get("others", {}).items():
            candidates.append((p, code, meta["lmdb_key"]))
    print(f"[gate 3] ceiling candidates: {len(candidates)} across "
          f"{len({p for p, _, _ in candidates})} pockets")

    records = extract_lmdb_records(sorted({k for _, _, k in candidates}))

    rows = []
    unusable = []
    for p, code, key in candidates:
        rec = records.get(key, {"error": "not_returned_by_subprocess"})
        if "error" in rec:
            unusable.append((p, code, rec["error"]))
            rows.append((p, code, float("nan"), None, False, rec["error"],
                        float("nan"), float("nan")))
            continue
        el = np.asarray(rec["element"])
        xyz = np.asarray(rec["pos"], dtype=float)[el > 1]
        offset = float(np.linalg.norm(xyz.mean(0) - ref_centroid[p]))
        same_site = offset <= SAME_SITE_CUTOFF
        if not same_site:
            rows.append((p, code, offset, same_site, False, "not_same_site",
                        float("nan"), float("nan")))
            continue
        try:
            mol = mol_from_lmdb_record(rec)
            f = compute_fp(mol, receptors[p])
        except Exception as exc:
            unusable.append((p, code, repr(exc)))
            rows.append((p, code, offset, same_site, False, repr(exc),
                        float("nan"), float("nan")))
            continue
        rows.append((p, code, offset, same_site, True, None,
                    tanimoto(f, ref_fp[p]), recall(f, ref_fp[p])))

    df = pd.DataFrame(rows, columns=["pocket", "ligcode", "centroid_offset",
                                     "same_site", "usable", "reason",
                                     "tanimoto", "recall"])
    print(f"[gate 3] same-site: {int(df.same_site.sum())}/{len(df)}   "
          f"usable (fingerprinted): {int(df.usable.sum())}/{len(df)}   "
          f"unusable-but-same-site: {len(unusable)} named below")
    if unusable:
        print(f"         unusable ceiling pairs: {unusable}")
    return df, unusable


# --------------------------------------------------------------------------- #
# Seed-noise floors -- exact structure of interface_fidelity_gates.seed_noise
# --------------------------------------------------------------------------- #

def seed_noise(per_arm, usable_pockets, stat):
    """Within-arm, seed-only deltas at two scales. `s_pocket` is the
    pocket-level spread (feeds the FP-B viability screen); `s_arm` is the
    spread of the mean over pockets (feeds the decision table). They differ by
    roughly sqrt(44) and are not interchangeable. Computed only from
    control-vs-control and critic_l0.7-vs-critic_l0.7 seed pairs -- never
    control vs critic.
    """
    col = {"tanimoto": 0, "recall": 1}[stat]
    out = {}
    for label, arms in (("control", CONTROL_ARMS), ("critic_l0.7", CRITIC_ARMS)):
        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                mean_d, top3_d = [], []
                for p in usable_pockets:
                    a = per_arm[arms[i]][p][:, col]
                    b = per_arm[arms[j]][p][:, col]
                    n = min(len(a), len(b))
                    if n == 0:
                        continue
                    mean_d.append(np.nanmean(a) - np.nanmean(b))
                    k = min(3, n)
                    top3_d.append(np.nanmean(np.sort(a)[-k:])
                                  - np.nanmean(np.sort(b)[-k:]))
                key = f"{label}_r{i}_vs_r{j}"
                out[key] = {
                    "mean_arm_delta": float(np.mean(mean_d)),
                    "mean_pocket_sd": float(np.std(mean_d, ddof=1)),
                    "top3_arm_delta": float(np.mean(top3_d)),
                    "top3_pocket_sd": float(np.std(top3_d, ddof=1)),
                }
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    (pockets, usable_pockets, receptors, refs_geom, ref_centroid,
     refs_mol_h, ref_fp, rec_fail, ref_fail) = load_all()

    # --- gate 4: size matching (heavy atoms) -- the ONLY place HELD_OUT appears
    sizes = {}
    for arm in GATE_ARMS + [HELD_OUT]:
        n = [len(heavy_coords(m)) for p in pockets
             for m in read_sdf(f"{arm}/{p}.sdf")]
        sizes[arm] = (float(np.mean(n)), float(np.std(n)), len(n))
    lo = min(v[0] for v in sizes.values())
    hi = max(v[0] for v in sizes.values())
    print(f"[gate 4] mean heavy atoms across all seven arms "
          f"(critic_lambda20_r0 included here ONLY, per the blinding rule): "
          f"{lo:.2f}-{hi:.2f} (spread {hi - lo:.2f})")
    for arm, (m, s, n) in sizes.items():
        print(f"         {arm}: mean {m:.2f} sd {s:.2f} n {n}")

    # --- gate 5: same pockets everywhere (also re-checked per-arm in process_arm)
    same = {arm: pocket_names(arm) == pockets for arm in GATE_ARMS + [HELD_OUT]}
    print(f"[gate 5] identical pocket sets across arms: {all(same.values())} "
          f"({len(pockets)} pockets)")
    assert all(same.values()), f"gate 5 failed: {same}"

    # --- per-arm FP-B processing (gate 6, ligand half) ------------------------
    per_arm, per_n_interact, pool_mols_by_arm = {}, {}, {}
    protonation_stats, fp_failures_all = {}, {}
    for arm in GATE_ARMS:
        pp, ni, pool, counts, fp_fail = process_arm(
            arm, pockets, usable_pockets, receptors, ref_fp)
        per_arm[arm] = pp
        per_n_interact[arm] = ni
        pool_mols_by_arm[arm] = pool
        protonation_stats[arm] = counts
        fp_failures_all[arm] = fp_fail
        n_mol = sum(len(v) for v in pool.values())
        rate = ((counts["fail_read_sanitize"] + counts["fail_addhs"])
                / counts["attempted"]) if counts["attempted"] else float("nan")
        print(f"[gate 6] {arm}: attempted {counts['attempted']}  "
              f"fail_read_sanitize {counts['fail_read_sanitize']}  "
              f"fail_addhs {counts['fail_addhs']}  repaired {counts['repaired']}  "
              f"failure_rate {rate:.4f}  fingerprinted {n_mol}  "
              f"fp_failures {len(fp_fail)}")
        if fp_fail:
            print(f"           fingerprint failures (named): {fp_fail[:20]}"
                  f"{' ...' if len(fp_fail) > 20 else ''}")

    # pool molecules across the six gate arms per pocket, for gate 1
    pool_mols = {p: [m for arm in GATE_ARMS for m in pool_mols_by_arm[arm][p]]
                for p in pockets}

    # --- gate 1 -----------------------------------------------------------
    null_df = gate_1_placement_null(pockets, usable_pockets, receptors,
                                    ref_centroid, ref_fp, pool_mols, rng)
    null_df.to_csv(OUT / "gate1_placement_null_fpb.csv", index=False)
    null_by_pocket = null_df.groupby("pocket")[["tanimoto", "recall"]].mean()
    real_by_pocket = pd.DataFrame(
        {p: np.nanmean(np.vstack([per_arm[a][p] for a in GATE_ARMS
                                  if len(per_arm[a][p])]), axis=0)
         for p in usable_pockets
         if any(len(per_arm[a][p]) for a in GATE_ARMS)},
        index=["tanimoto", "recall"]).T
    print(f"[gate 1] placement null   tanimoto {null_by_pocket.tanimoto.mean():.4f}"
          f"   recall {null_by_pocket.recall.mean():.4f}")
    print(f"         real arms (pooled) tanimoto "
          f"{real_by_pocket.tanimoto.mean():.4f}   recall "
          f"{real_by_pocket.recall.mean():.4f}")
    n_null_errors = int(null_df.error.notna().sum())
    if n_null_errors:
        print(f"         gate 1 fingerprint errors (named, counted): "
              f"{n_null_errors} -- see gate1_placement_null_fpb.csv")

    # --- gate 2 -----------------------------------------------------------
    dyn = gate_2_dynamic_range(usable_pockets, receptors, refs_mol_h, ref_fp, rng)
    dyn.to_csv(OUT / "gate2_dynamic_range_fpb.csv", index=False)
    curve = dyn.groupby("displacement")[["tanimoto", "recall"]].mean()
    print("[gate 2] self-recovery vs rigid displacement:")
    for d, row in curve.iterrows():
        print(f"         {d:>4} A   tanimoto {row.tanimoto:.4f}   "
              f"recall {row.recall:.4f}")

    # --- gate 3 -----------------------------------------------------------
    ceil, ceil_unusable = gate_3_ceiling(usable_pockets, receptors, ref_centroid,
                                         ref_fp)
    ceil.to_csv(OUT / "gate3_ceiling_fpb.csv", index=False)
    ok = ceil[ceil.usable]
    if len(ok):
        print(f"[gate 3] ceiling tanimoto  med {ok.tanimoto.median():.4f} "
              f"IQR [{ok.tanimoto.quantile(.25):.4f}, "
              f"{ok.tanimoto.quantile(.75):.4f}]")
        print(f"         ceiling recall    med {ok.recall.median():.4f} "
              f"IQR [{ok.recall.quantile(.25):.4f}, "
              f"{ok.recall.quantile(.75):.4f}]")
    else:
        print("[gate 3] no usable same-site ceiling pairs")

    # --- seed-noise floors --------------------------------------------------
    noise = {s: seed_noise(per_arm, usable_pockets, s)
             for s in ("tanimoto", "recall")}
    for stat, d in noise.items():
        s_arm_mean = max(abs(v["mean_arm_delta"]) for v in d.values())
        s_arm_top3 = max(abs(v["top3_arm_delta"]) for v in d.values())
        s_pkt_mean = max(v["mean_pocket_sd"] for v in d.values())
        s_pkt_top3 = max(v["top3_pocket_sd"] for v in d.values())
        print(f"[noise/{stat}] s_arm  mean {s_arm_mean:.4f}  top3 {s_arm_top3:.4f}")
        print(f"[noise/{stat}] s_pocket mean {s_pkt_mean:.4f}  "
              f"top3 {s_pkt_top3:.4f}")

    # --- promiscuity guard: mean FP-B interactions per molecule, per arm ----
    promiscuity = {}
    for arm in GATE_ARMS:
        all_counts = [c for p in pockets for c in per_n_interact[arm][p]]
        promiscuity[arm] = {
            "mean": float(np.mean(all_counts)) if all_counts else float("nan"),
            "sd": float(np.std(all_counts)) if all_counts else float("nan"),
            "n": len(all_counts),
        }
        print(f"[promiscuity] {arm}: mean {promiscuity[arm]['mean']:.3f} "
              f"interactions/molecule  (n={promiscuity[arm]['n']})")

    # --- write outputs --------------------------------------------------------
    summary = {
        "n_pockets": len(pockets),
        "n_usable_pockets": len(usable_pockets),
        "gate6_receptor_protonation_failures": rec_fail,
        "gate6_reference_ligand_failures": ref_fail,
        "gate6_ligand_protonation": protonation_stats,
        "gate6_fingerprint_failures": {a: len(v) for a, v in
                                       fp_failures_all.items()},
        "gate6_fingerprint_failures_named": fp_failures_all,
        "gate4_sizes": sizes,
        "gate5_same_pockets": {k: bool(v) for k, v in same.items()},
        "ref_bits": {
            "median": float(np.median([len(ref_fp[p]) for p in usable_pockets])),
            "min": int(min(len(ref_fp[p]) for p in usable_pockets)),
            "max": int(max(len(ref_fp[p]) for p in usable_pockets)),
        },
        "gate1_null": {"tanimoto": float(null_by_pocket.tanimoto.mean()),
                       "recall": float(null_by_pocket.recall.mean())},
        "gate1_null_errors": n_null_errors,
        "gate1_real_pooled": {"tanimoto": float(real_by_pocket.tanimoto.mean()),
                              "recall": float(real_by_pocket.recall.mean())},
        "gate2_curve": curve.to_dict(),
        "gate3_ceiling": {
            "n_candidate": int(len(ceil)),
            "n_same_site": int(ceil.same_site.sum()),
            "n_usable": int(ok.shape[0]),
            "n_pockets": int(ok.pocket.nunique()) if len(ok) else 0,
            "tanimoto_median": float(ok.tanimoto.median()) if len(ok) else None,
            "recall_median": float(ok.recall.median()) if len(ok) else None,
            "unusable_same_site_pairs": ceil_unusable,
        },
        "seed_noise": noise,
        "promiscuity_guard": promiscuity,
        "interaction_names": INTERACTION_NAMES,
    }
    (OUT / "gates_fpb.json").write_text(json.dumps(summary, indent=1))

    tbl = null_by_pocket.rename(columns=lambda c: f"null_{c}")
    tbl = tbl.join(real_by_pocket.rename(columns=lambda c: f"real_{c}"),
                   how="outer")
    if len(ok):
        cs = ok.groupby("pocket")[["tanimoto", "recall"]].agg(["median", "size"])
        cs.columns = ["ceiling_tanimoto", "n_pairs", "ceiling_recall", "_n2"]
        tbl = tbl.join(cs.drop(columns="_n2"), how="left")
    else:
        tbl["ceiling_tanimoto"] = np.nan
        tbl["n_pairs"] = 0
        tbl["ceiling_recall"] = np.nan
    tbl.to_csv(OUT / "gates_fpb_per_pocket.csv")
    print(f"\nwrote {OUT}/gates_fpb.json and gates_fpb_per_pocket.csv")


if __name__ == "__main__":
    main()
