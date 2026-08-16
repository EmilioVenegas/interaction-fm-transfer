"""Does Uni-Dock agree with smina closely enough to replace it here?

Cross-docking is the bottleneck in this project: 35 core-seconds per molecule at
exhaustiveness 8, 7,040 molecules per comparison, ~4 h of 16 cores. The
measurement that matters is also badly underpowered for a reason that only more
docking fixes -- 45% of the variance in the specificity effect comes from
subsampling 20 of ~95 molecules per pocket (`results/specificity/ANALYSIS_PLAN.md`).
Uni-Dock runs the same Vina scoring function on the GPU, so it could make the
full molecule set affordable.

None of that is worth anything if the scores disagree. This script docks the
same pocket/receptor/molecule triples both ways and reports whether they do.

The gate is deliberately not "the scores correlate". A high correlation across
pockets is easy and useless here, because the statistic this project reports is
a *difference of means within a pocket*: `cross - own`, a fraction of a kcal/mol
on scores that span 6. What has to survive is that per-pocket difference. So the
criteria are, in order:

  1. per-pocket specificity agrees between engines (the statistic we report),
  2. per-molecule scores correlate and share a scale (a sanity floor),
  3. Uni-Dock is actually faster on this workload (the whole point).

A systematic offset is acceptable and expected -- different search, different
random seeds -- as long as it cancels in the difference. A per-pocket
disagreement is not, however good the pooled correlation looks.

    python scripts/validate_unidock.py \\
        --sdf_dir results/gen_critic_graph_cosine_r0 \\
        --pdb_dir data/receptor_pdbs_test_v2 \\
        --n_pockets 4 --n_mols 10
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parent.parent
SMINA = os.environ.get("SMINA_BIN", str(Path.home() / ".conda/envs/smina/bin/smina"))
UNIDOCK = os.environ.get("UNIDOCK_BIN", str(Path.home() / ".conda/envs/unidock/bin/unidock"))
# obabel is not on PATH in this shell; it lives in the smina env alongside the
# docking tools. Named explicitly so this cannot silently fall back to a
# different build and produce differently protonated ligands for one engine.
OBABEL = os.environ.get("OBABEL_BIN", str(Path.home() / ".conda/envs/smina/bin/obabel"))


def read_box(box_file):
    """center_x/y/z and size_x/y/z as written by scripts/extract_pocket_pdbs.py."""
    box = {}
    for line in Path(box_file).read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            box[key.strip()] = float(value.strip())
    return box


def dock_smina(receptor, box, ligand_sdf, out_sdf, exhaustiveness, seed):
    cmd = [SMINA, "-r", str(receptor), "-l", str(ligand_sdf), "-o", str(out_sdf),
           "--center_x", str(box["center_x"]), "--center_y", str(box["center_y"]),
           "--center_z", str(box["center_z"]),
           "--size_x", str(box["size_x"]), "--size_y", str(box["size_y"]),
           "--size_z", str(box["size_z"]),
           "--exhaustiveness", str(exhaustiveness), "--seed", str(seed),
           "--num_modes", "1", "--cpu", "1", "--quiet"]
    subprocess.run(cmd, check=True, capture_output=True)
    return scores_from_sdf(out_sdf, "minimizedAffinity")


def dock_unidock(receptor_pdbqt, box, ligand_sdfs, out_dir, exhaustiveness, seed,
                 max_gpu_mb=0):
    """One GPU call for a whole batch of ligands -- the batching IS the speedup,
    so timing this one ligand at a time would measure the wrong thing.

    Ligands go in as the same SDF files smina reads, not as obabel-converted
    pdbqt: converting them would introduce a protonation difference between the
    engines and the comparison would then be measuring obabel. The receptor has
    to be converted, since Uni-Dock will not read PDB, and that difference is
    real and is part of what this validation is for.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [UNIDOCK, "--receptor", str(receptor_pdbqt),
           "--gpu_batch", *[str(p) for p in ligand_sdfs],
           "--dir", str(out_dir),
           "--center_x", str(box["center_x"]), "--center_y", str(box["center_y"]),
           "--center_z", str(box["center_z"]),
           "--size_x", str(box["size_x"]), "--size_y", str(box["size_y"]),
           "--size_z", str(box["size_z"]),
           "--exhaustiveness", str(exhaustiveness), "--seed", str(seed),
           "--num_modes", "1", "--scoring", "vina", "--verbosity", "0"]
    if max_gpu_mb:
        cmd += ["--max_gpu_memory", str(max_gpu_mb)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"unidock failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return proc


def scores_from_sdf(path, prop):
    out = []
    for mol in Chem.SDMolSupplier(str(path), sanitize=False):
        if mol is None:
            out.append(float("nan"))
            continue
        value = mol.GetPropsAsDict().get(prop)
        out.append(float(value) if value is not None else float("nan"))
    return out


def read_unidock_score(path):
    """Uni-Dock writes the Vina score into the output, as a pdbqt REMARK or an
    SDF property depending on the output format it chose."""
    path = Path(path)
    if path.suffix == ".pdbqt":
        for line in path.read_text().splitlines():
            if line.startswith("REMARK VINA RESULT"):
                return float(line.split()[3])
        return float("nan")
    for mol in Chem.SDMolSupplier(str(path), sanitize=False):
        if mol is None:
            continue
        props = mol.GetPropsAsDict()
        for key in ("Uni-Dock RESULT", "minimizedAffinity", "docking_score",
                    "ENERGY", "Energy"):
            if key in props:
                try:
                    return float(str(props[key]).split()[0])
                except (ValueError, IndexError):
                    pass
    return float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sdf_dir", default="results/gen_critic_graph_cosine_r0")
    p.add_argument("--pdb_dir", default="data/receptor_pdbs_test_v2")
    p.add_argument("--n_pockets", type=int, default=4)
    p.add_argument("--n_mols", type=int, default=10)
    p.add_argument("--n_decoy", type=int, default=2)
    p.add_argument("--exhaustiveness", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_gpu_mb", type=int, default=0,
                   help="cap Uni-Dock's GPU use; 0 lets it take what it wants")
    p.add_argument("--out", default="results/specificity/unidock_validation.csv")
    args = p.parse_args()

    os.chdir(REPO)
    for name, path in (("smina", SMINA), ("unidock", UNIDOCK), ("obabel", OBABEL)):
        if not Path(path).exists():
            raise SystemExit(f"{name} not found at {path}")

    pdb_dir = Path(args.pdb_dir)
    receptors = {p.stem: (p, p.with_suffix(".box.txt"))
                 for p in sorted(pdb_dir.glob("*.pdb"))
                 if p.with_suffix(".box.txt").exists()}
    targets_path = pdb_dir / "pocket_targets.json"
    pocket_targets = json.loads(targets_path.read_text()) if targets_path.exists() else {}

    sdfs = sorted(Path(args.sdf_dir).glob("*.sdf"))[:args.n_pockets]
    rng = np.random.default_rng(args.seed)

    rows, timing = [], {"smina": 0.0, "unidock": 0.0, "n": 0}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for sdf in sdfs:
            pocket = sdf.stem
            if pocket not in receptors:
                continue
            mols = [m for m in Chem.SDMolSupplier(str(sdf)) if m is not None][:args.n_mols]
            if not mols:
                continue

            own_target = pocket_targets.get(pocket)
            others = [q for q in receptors
                      if q != pocket and pocket_targets.get(q) != own_target]
            decoys = list(rng.choice(others, size=min(args.n_decoy, len(others)),
                                     replace=False))

            lig_dir = tmp / pocket / "lig"
            lig_dir.mkdir(parents=True, exist_ok=True)
            single = tmp / pocket / "ligands.sdf"
            writer = Chem.SDWriter(str(single))
            lig_files = []
            for i, mol in enumerate(mols):
                writer.write(mol)
                one = lig_dir / f"m{i}.sdf"
                w2 = Chem.SDWriter(str(one))
                w2.write(mol)
                w2.close()
                lig_files.append(one)
            writer.close()

            for receptor in [pocket, *decoys]:
                rec_pdb, box_file = receptors[receptor]
                box = read_box(box_file)
                rec_pdbqt = tmp / f"{receptor}.pdbqt"
                if not rec_pdbqt.exists():
                    subprocess.run([OBABEL, str(rec_pdb), "-O", str(rec_pdbqt),
                                    "-xr"], capture_output=True)

                t0 = time.time()
                out_sdf = tmp / pocket / f"{receptor}_smina.sdf"
                smina_scores = dock_smina(rec_pdb, box, single, out_sdf,
                                          args.exhaustiveness, args.seed)
                timing["smina"] += time.time() - t0

                t0 = time.time()
                out_dir = tmp / pocket / f"{receptor}_unidock"
                dock_unidock(rec_pdbqt, box, lig_files, out_dir,
                             args.exhaustiveness, args.seed,
                             max_gpu_mb=args.max_gpu_mb)
                timing["unidock"] += time.time() - t0

                uni_scores = []
                for i in range(len(mols)):
                    hit = (list(out_dir.glob(f"m{i}_out.sdf"))
                           or list(out_dir.glob(f"m{i}_out.pdbqt"))
                           or list(out_dir.glob(f"m{i}.sdf"))
                           or list(out_dir.glob(f"m{i}.pdbqt")))
                    uni_scores.append(read_unidock_score(hit[0]) if hit else float("nan"))

                timing["n"] += len(mols)
                for i in range(len(mols)):
                    rows.append({
                        "pocket": pocket, "receptor": receptor, "mol": i,
                        "is_own": receptor == pocket,
                        "smina": smina_scores[i] if i < len(smina_scores) else float("nan"),
                        "unidock": uni_scores[i]})

    frame = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)

    ok = frame.dropna(subset=["smina", "unidock"])
    print(f"\n{len(ok)} molecule-receptor pairs scored by both engines "
          f"({frame.pocket.nunique()} pockets).\n")
    if ok.empty:
        raise SystemExit("Nothing scored by both engines — Uni-Dock is not usable here.")

    print(f"{'':<22}{'smina':>10}{'unidock':>10}")
    print("-" * 42)
    print(f"{'mean score':<22}{ok.smina.mean():>10.3f}{ok.unidock.mean():>10.3f}")
    print(f"{'sd':<22}{ok.smina.std(ddof=1):>10.3f}{ok.unidock.std(ddof=1):>10.3f}")
    offset = (ok.unidock - ok.smina).mean()
    print(f"\nmean offset (unidock - smina): {offset:+.3f} kcal/mol "
          f"(a constant offset cancels in cross - own)")
    print(f"pearson  r = {ok.smina.corr(ok.unidock):.3f}")
    print(f"spearman r = {ok.smina.corr(ok.unidock, method='spearman'):.3f}")

    # The statistic this project actually reports.
    spec = {}
    for engine in ("smina", "unidock"):
        own = ok[ok.is_own].groupby("pocket")[engine].mean()
        cross = ok[~ok.is_own].groupby("pocket")[engine].mean()
        spec[engine] = (cross - own).dropna()
    joined = pd.DataFrame(spec).dropna()
    print(f"\nper-pocket specificity (cross - own), the reported statistic:\n")
    print(f"{'pocket':<20}{'smina':>10}{'unidock':>10}{'diff':>10}")
    print("-" * 50)
    for pocket, row in joined.iterrows():
        print(f"{pocket:<20}{row.smina:>+10.3f}{row.unidock:>+10.3f}"
              f"{row.unidock - row.smina:>+10.3f}")
    d = (joined.unidock - joined.smina).values
    print(f"\nmean |disagreement| {np.abs(d).mean():.3f} kcal/mol, "
          f"max {np.abs(d).max():.3f}")

    speed = timing["smina"] / timing["unidock"] if timing["unidock"] else float("nan")
    print(f"\nwall time over {timing['n']} dockings: smina {timing['smina']:.0f}s "
          f"(1 core), unidock {timing['unidock']:.0f}s (GPU) — {speed:.1f}x")

    print("\nGATE")
    checks = [
        ("per-pocket specificity agrees within 0.15 kcal/mol on average",
         np.abs(d).mean() < 0.15),
        ("per-molecule spearman >= 0.7", ok.smina.corr(ok.unidock, method="spearman") >= 0.7),
        ("unidock faster per docking than smina on one core", speed > 1.0),
    ]
    for text, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {text}")
    passed = all(c[1] for c in checks)
    print("\n" + ("Uni-Dock may replace smina for this measurement."
                  if passed else
                  "Do NOT switch: keep smina and report the bound it gives."))
    print(f"\nWrote {out}")
    # Exit code is the gate, so a driver can chain a full run on it rather than
    # a human reading a table at 3am.
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()
