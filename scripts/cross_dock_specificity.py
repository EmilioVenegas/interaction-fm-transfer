"""Cross-docking specificity: does a molecule prefer the pocket it was designed for?

The A/B ablation could not have detected pocket specificity. QED is
target-independent, so a QED gain is equally consistent with "conditioning
narrowed the model toward generically drug-like chemistry" and with
"conditioning improved pocket fit" (docs/experiment-plan.md, "Evaluation").
This is the measurement that separates them.

Each pocket's molecules are docked into **its own** pocket and into *m* other
pockets from the same evaluation set. The statistic per pocket is

    specificity = mean(score against other pockets) - mean(score against own)

With smina, lower is better, so a molecule that genuinely fits the pocket it was
generated for gives a **positive** specificity. A generically drug-like molecule
docks about as well anywhere and gives approximately zero, however good its
absolute score or its QED.

Two things this script refuses to do, both of which have already produced wrong
conclusions in this project:

- **It does not pool molecules.** The ~9,300 valid molecules per arm are nested
  within ~100 pockets; treating them as independent is pseudo-replication and
  inflates significance. Every statistic here is computed per pocket first, and
  arms are compared by a paired test *across pockets* with the fraction of
  pockets improved reported alongside the mean shift.
- **It re-docks rather than rescoring in place.** A molecule generated for
  pocket A sits in A's coordinate frame; running `--score_only` against pocket B
  would score it wherever it happens to land in B, which measures the frame
  offset, not the fit. Each pair is docked into a box centred on the target
  pocket.

Docking-based evaluation is only non-circular because the Vina expert filter was
dropped (`--no_expert_filter`). Do not reintroduce that filter and then report
docking scores.

Requires the smina binary (`conda install -c conda-forge smina`, or set
SMINA_BIN). Receptor PDBs and their boxes come from
`scripts/extract_pocket_pdbs.py`; regenerate them whenever the test split is
rebuilt, or the pocket ids here will not match the ones the molecules were
generated for.

Usage (from repo root):

    python scripts/cross_dock_specificity.py \\
        --arms baseline_A=results/baseline_A/filtered \\
               critic=results/critic/filtered \\
        --pdb_dir data/receptor_pdbs_test_v2 \\
        --out results/specificity/specificity.csv \\
        --n_decoy_pockets 3 --max_mols_per_pocket 20 --n_jobs 8
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

SMINA_BIN = os.environ.get("SMINA_BIN", "smina")
UNIDOCK_BIN = os.environ.get(
    "UNIDOCK_BIN", str(Path.home() / ".conda/envs/unidock/bin/unidock"))
OBABEL_BIN = os.environ.get(
    "OBABEL_BIN", str(Path.home() / ".conda/envs/smina/bin/obabel"))


def read_box(box_path):
    """Parse a `key = value` box file into smina arguments."""
    box = {}
    with open(box_path) as fh:
        for line in fh:
            if "=" in line:
                key, value = line.split("=", 1)
                box[key.strip()] = float(value.strip())
    return box


def dock_one(args):
    """Dock every molecule of one SDF into one receptor's box.

    Returns ``(pocket, receptor, [scores])``. A failed pair yields NaNs rather
    than raising: over thousands of pairs a handful will fail, and losing the
    whole run to one of them is worse than a gap in the table.
    """
    sdf_path, receptor_pdb, box_path, pocket, receptor, exhaustiveness, seed = args
    box = read_box(box_path)

    suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=True)
    mols = [m for m in suppl if m is not None]
    if not mols:
        return pocket, receptor, []

    with tempfile.TemporaryDirectory() as tmpdir:
        ligand_file = Path(tmpdir) / "ligands.sdf"
        writer = Chem.SDWriter(str(ligand_file))
        for mol in mols:
            writer.write(mol)
        writer.close()

        cmd = [
            SMINA_BIN,
            "--receptor", str(receptor_pdb),
            "--ligand", str(ligand_file),
            "--center_x", str(box["center_x"]),
            "--center_y", str(box["center_y"]),
            "--center_z", str(box["center_z"]),
            "--size_x", str(box["size_x"]),
            "--size_y", str(box["size_y"]),
            "--size_z", str(box["size_z"]),
            "--exhaustiveness", str(exhaustiveness),
            "--num_modes", "1",
            "--seed", str(seed),
            "--cpu", "1",
            "--out", str(Path(tmpdir) / "out.sdf"),
        ]
        try:
            # 5400s, not 1800s. One smina call docks every molecule of a pocket
            # serially -- 20 molecules at ~35 core-seconds is ~700s nominal, and
            # under contention a slow pocket approaches half an hour. At
            # n_jobs 28 on 32 cores that crossed the old 1800s limit and turned
            # whole pockets into NaN, asymmetrically between arms, because the
            # arm that happened to run second ran slower. A timeout that fires
            # in normal operation is not a safety net, it is a silent sampler.
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5400)
        except FileNotFoundError:
            raise RuntimeError(
                f"smina not found at '{SMINA_BIN}'. Install with "
                "`conda install -c conda-forge smina` or set SMINA_BIN."
            )
        except subprocess.TimeoutExpired:
            return pocket, receptor, [float("nan")] * len(mols)

        if proc.returncode != 0:
            return pocket, receptor, [float("nan")] * len(mols)

        scores = []
        out_sdf = Path(tmpdir) / "out.sdf"
        if out_sdf.exists():
            for mol in Chem.SDMolSupplier(str(out_sdf), sanitize=False):
                if mol is None:
                    scores.append(float("nan"))
                    continue
                value = mol.GetPropsAsDict().get("minimizedAffinity")
                scores.append(float(value) if value is not None else float("nan"))
    return pocket, receptor, scores


def receptor_pdbqt(receptor_pdb, cache_dir):
    """Uni-Dock will not read PDB, so each receptor is converted once and reused.

    Converting per docking call instead would repeat the conversion 44 times per
    receptor and put obabel on the critical path of a GPU run.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / (Path(receptor_pdb).stem + ".pdbqt")
    if not out.exists():
        proc = subprocess.run([OBABEL_BIN, str(receptor_pdb), "-O", str(out), "-xr"],
                              capture_output=True, text=True)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"obabel failed on {receptor_pdb}:\n{proc.stderr[-1000:]}")
    return out


def read_unidock_score(path):
    """Uni-Dock writes the Vina score as a pdbqt REMARK or an SDF property.

    Never raises. A ligand whose docking failed can leave an empty or truncated
    file behind, and RDKit raises OSError rather than returning None on those --
    which killed a validation run after 33 minutes of GPU work. One unreadable
    ligand out of thousands is a NaN, exactly as a failed smina pair is.
    """
    path = Path(path)
    try:
        if path.stat().st_size == 0:
            return float("nan")
        text_score = None
        if path.suffix == ".pdbqt":
            for line in path.read_text().splitlines():
                if line.startswith("REMARK VINA RESULT"):
                    return float(line.split()[3])
            return float("nan")
        # SDF: the score may be a tagged property or sit in the title block,
        # depending on which writer Uni-Dock used.
        for mol in Chem.SDMolSupplier(str(path), sanitize=False):
            if mol is None:
                continue
            props = mol.GetPropsAsDict()
            for key in ("Uni-Dock RESULT", "minimizedAffinity", "docking_score",
                        "ENERGY", "Energy", "Score", "score"):
                if key in props:
                    try:
                        return float(str(props[key]).split()[0])
                    except (ValueError, IndexError):
                        pass
        # Last resort: scrape a REMARK-style line out of the raw text.
        for line in path.read_text().splitlines():
            if "VINA RESULT" in line or "ENERGY=" in line:
                for token in line.replace("=", " ").split():
                    try:
                        return float(token)
                    except ValueError:
                        continue
        return float("nan")
    except (OSError, ValueError, RuntimeError):
        return float("nan")


def dock_one_unidock(args, pdbqt_cache, max_gpu_mb=0):
    """One GPU call for every molecule of one SDF against one receptor.

    Unlike the smina path this is NOT run under a process pool: the batching is
    where the speed comes from, and several processes contending for one 8 GB
    card would serialise anyway while risking an out-of-memory kill.
    """
    sdf_path, receptor_pdb, box_path, pocket, receptor, exhaustiveness, seed = args
    box = read_box(box_path)
    mols = [m for m in Chem.SDMolSupplier(str(sdf_path), sanitize=True) if m is not None]
    if not mols:
        return pocket, receptor, []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        lig_files = []
        for i, mol in enumerate(mols):
            one = tmpdir / f"m{i}.sdf"
            writer = Chem.SDWriter(str(one))
            writer.write(mol)
            writer.close()
            lig_files.append(one)
        out_dir = tmpdir / "out"
        out_dir.mkdir()

        cmd = [UNIDOCK_BIN,
               "--receptor", str(receptor_pdbqt(receptor_pdb, pdbqt_cache)),
               "--gpu_batch", *[str(f) for f in lig_files],
               "--dir", str(out_dir),
               "--center_x", str(box["center_x"]),
               "--center_y", str(box["center_y"]),
               "--center_z", str(box["center_z"]),
               "--size_x", str(box["size_x"]),
               "--size_y", str(box["size_y"]),
               "--size_z", str(box["size_z"]),
               "--exhaustiveness", str(exhaustiveness),
               "--num_modes", "1", "--seed", str(seed),
               "--scoring", "vina", "--verbosity", "0"]
        if max_gpu_mb:
            cmd += ["--max_gpu_memory", str(max_gpu_mb)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        except subprocess.TimeoutExpired:
            return pocket, receptor, [float("nan")] * len(mols)
        if proc.returncode != 0:
            return pocket, receptor, [float("nan")] * len(mols)

        scores = []
        for i in range(len(mols)):
            hit = (list(out_dir.glob(f"m{i}_out.sdf"))
                   or list(out_dir.glob(f"m{i}_out.pdbqt"))
                   or list(out_dir.glob(f"m{i}.sdf"))
                   or list(out_dir.glob(f"m{i}.pdbqt")))
            scores.append(read_unidock_score(hit[0]) if hit else float("nan"))
    return pocket, receptor, scores


def choose_decoys(receptors, n_decoy, rng, pocket_targets=None):
    """pocket -> the decoy pockets it is docked against, chosen ONCE for all arms.

    Decoy pockets are drawn from **different targets** when `pocket_targets` is
    available. CrossDocked holds many complexes per target, so without that
    constraint a "decoy" pocket can be the same protein in another docked pose --
    which a pocket-specific molecule should fit, washing out the contrast the
    metric exists to measure.

    Every arm shares this mapping. It used to be drawn per arm from an rng that
    the arms consumed in turn, so no two arms were ever docked against the same
    decoys -- 0 of 44 pockets shared a decoy set between the critic and control
    arms. That is not a bias, but it puts the whole between-decoy variance into
    a comparison meant to isolate the objective: a pocket's score varies by 0.64
    kcal/mol across decoy receptors, so two arms drawing independently differ by
    ~0.52 kcal/mol per pocket from the draw alone, against reported effects of
    -0.158 (r0) and -0.201 (r1). Sharing the draw removes that term entirely and
    costs nothing.
    """
    pockets = sorted(receptors)
    decoys = {}
    for pocket in pockets:
        own_target = (pocket_targets or {}).get(pocket)
        others = [
            p for p in pockets
            if p != pocket
            and (own_target is None
                 or (pocket_targets or {}).get(p) != own_target)
        ]
        decoys[pocket] = list(
            rng.choice(others, size=min(n_decoy, len(others)), replace=False)
        ) if others else []
    return decoys


def build_pairs(sdf_by_pocket, receptors, decoys, exhaustiveness, seed):
    """(molecule set, receptor) pairs: each pocket against its own and its decoys."""
    pairs = []
    for pocket in sorted(sdf_by_pocket):
        if pocket not in receptors:
            continue
        for receptor in [pocket, *decoys.get(pocket, [])]:
            if receptor not in receptors:
                continue
            pdb, box = receptors[receptor]
            pairs.append((sdf_by_pocket[pocket], pdb, box, pocket, receptor,
                          exhaustiveness, seed))
    return pairs


def subsample_sdf(src, dest, limit, rng):
    """Write at most `limit` molecules from `src` into `dest`.

    Cross-docking cost is (pockets x molecules x receptors), so the molecule
    count is the only term worth cutting. Sampling is random rather than
    top-of-file, since generation order is not arbitrary.
    """
    mols = [m for m in Chem.SDMolSupplier(str(src), sanitize=True) if m is not None]
    if not mols:
        return None
    if len(mols) > limit:
        mols = [mols[i] for i in sorted(rng.choice(len(mols), limit, replace=False))]
    writer = Chem.SDWriter(str(dest))
    for mol in mols:
        writer.write(mol)
    writer.close()
    return dest


def collect_arm(arm_dir, receptors, workdir, limit, rng):
    """Map pocket id -> a (possibly subsampled) SDF of that pocket's molecules."""
    arm_dir = Path(arm_dir)
    out = {}
    for sdf in sorted(arm_dir.glob("*.sdf")):
        pocket = sdf.stem.replace("_filtered", "")
        if pocket not in receptors:
            continue
        dest = subsample_sdf(sdf, workdir / f"{pocket}.sdf", limit, rng)
        if dest is not None:
            out[pocket] = dest
    return out


def per_pocket_specificity(rows):
    """Collapse raw pair scores into one specificity value per pocket."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.dropna(subset=["score"])
    matched = frame[frame["pocket"] == frame["receptor"]]
    crossed = frame[frame["pocket"] != frame["receptor"]]

    own = matched.groupby("pocket")["score"].mean().rename("own_score")
    other = crossed.groupby("pocket")["score"].mean().rename("cross_score")
    counts = matched.groupby("pocket")["score"].size().rename("n_mols")

    result = pd.concat([own, other, counts], axis=1).dropna()
    # smina: lower is better, so cross - own > 0 means the molecule prefers the
    # pocket it was designed for.
    result["specificity"] = result["cross_score"] - result["own_score"]
    return result.reset_index()


def paired_report(arms):
    """Paired comparison across the pockets common to every arm."""
    names = list(arms)
    common = set(arms[names[0]]["pocket"])
    for name in names[1:]:
        common &= set(arms[name]["pocket"])
    common = sorted(common)
    if not common:
        print("No pockets common to all arms; nothing to compare.")
        return None

    table = pd.DataFrame({"pocket": common})
    for name in names:
        sub = arms[name].set_index("pocket").loc[common]
        table[f"{name}__own"] = sub["own_score"].values
        table[f"{name}__cross"] = sub["cross_score"].values
        table[f"{name}__specificity"] = sub["specificity"].values

    print(f"\n{len(common)} pockets common to all arms. "
          f"n is the number of pockets, not molecules.\n")
    header = f"{'arm':<20} {'own':>10} {'cross':>10} {'specificity':>22} {'frac > 0':>10}"
    print(header)
    print("-" * len(header))
    for name in names:
        spec = table[f"{name}__specificity"].values
        sem = spec.std(ddof=1) / np.sqrt(len(spec)) if len(spec) > 1 else 0.0
        print(f"{name:<20} {table[f'{name}__own'].mean():>10.3f} "
              f"{table[f'{name}__cross'].mean():>10.3f} "
              f"{spec.mean():>+13.3f} +- {sem:<5.3f} "
              f"{(spec > 0).mean():>10.2f}")

    if len(names) > 1:
        try:
            from scipy.stats import wilcoxon
        except ImportError:
            print("\n(scipy not available; skipping paired tests)")
            return table
        print(f"\nPaired across pockets, against '{names[0]}':")
        print(f"{'arm':<20} {'d(specificity)':>16} {'pockets improved':>18} {'wilcoxon p':>12}")
        print("-" * 70)
        base = table[f"{names[0]}__specificity"].values
        for name in names[1:]:
            other = table[f"{name}__specificity"].values
            delta = other - base
            try:
                p = wilcoxon(other, base).pvalue
            except ValueError:
                p = float("nan")
            print(f"{name:<20} {delta.mean():>+16.3f} "
                  f"{f'{(delta > 0).sum()}/{len(delta)}':>18} {p:>12.4g}")
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--arms", nargs="+", required=True,
                   help="name=path/to/filtered pairs, one per arm")
    p.add_argument("--pdb_dir", default="data/receptor_pdbs_test_v2")
    p.add_argument("--out", default="results/specificity/specificity.csv")
    p.add_argument("--n_decoy_pockets", type=int, default=3,
                   help="other pockets each molecule set is docked into")
    p.add_argument("--max_mols_per_pocket", type=int, default=20)
    p.add_argument("--exhaustiveness", type=int, default=8)
    p.add_argument("--n_jobs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--engine", choices=("smina", "unidock"), default="smina",
                   help="unidock runs the same Vina scoring function on the GPU; "
                        "validate it with scripts/validate_unidock.py before "
                        "trusting a number it produced")
    p.add_argument("--max_gpu_mb", type=int, default=0,
                   help="cap Uni-Dock's GPU use; 0 lets it take what it wants")
    args = p.parse_args()

    pdb_dir = Path(args.pdb_dir)
    receptors = {}
    for pdb in sorted(pdb_dir.glob("*.pdb")):
        box = pdb.with_suffix(".box.txt")
        if box.exists():
            receptors[pdb.stem] = (pdb, box)
    if not receptors:
        print(f"No receptor/box pairs under {pdb_dir}. Run "
              f"scripts/extract_pocket_pdbs.py first.")
        return
    # Written by scripts/extract_pocket_pdbs.py. Without it, decoy pockets can
    # be other poses of the same protein.
    targets_path = pdb_dir / "pocket_targets.json"
    pocket_targets = None
    if targets_path.exists():
        with open(targets_path) as fh:
            pocket_targets = json.load(fh)
        print(f"{len(receptors)} receptors with boxes, "
              f"{len(set(pocket_targets.values()))} distinct targets.")
    else:
        print(f"{len(receptors)} receptors with boxes. WARNING: no "
              f"{targets_path.name}; decoy pockets may be other poses of the "
              f"same protein, which understates specificity. Regenerate with "
              f"scripts/extract_pocket_pdbs.py.")

    # Two independent streams. The decoy draw must not depend on how many
    # random numbers the molecule subsampling happened to consume first, or the
    # arms desynchronise and each gets its own decoy set -- which is exactly
    # what happened in the r0/r1/r2 runs.
    decoy_rng = np.random.default_rng(args.seed)
    rng = np.random.default_rng(args.seed + 10_000)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    decoys = choose_decoys(receptors, args.n_decoy_pockets, decoy_rng,
                           pocket_targets=pocket_targets)
    shared = sum(1 for v in decoys.values() if v)
    print(f"Decoy pockets drawn once and shared by every arm "
          f"({shared} pockets with decoys).")

    arm_specificity, raw_rows = {}, []
    with tempfile.TemporaryDirectory() as tmpdir:
        for spec in args.arms:
            if "=" not in spec:
                print(f"Skipping malformed arm {spec!r}; expected name=path")
                continue
            name, path = spec.split("=", 1)
            workdir = Path(tmpdir) / name
            workdir.mkdir(parents=True, exist_ok=True)

            sdf_by_pocket = collect_arm(path, receptors, workdir,
                                        args.max_mols_per_pocket, rng)
            if not sdf_by_pocket:
                print(f"Arm {name!r}: no usable SDFs under {path}")
                continue
            pairs = build_pairs(sdf_by_pocket, receptors, decoys,
                                args.exhaustiveness, args.seed)
            print(f"\nArm {name!r}: {len(sdf_by_pocket)} pockets, {len(pairs)} "
                  f"docking jobs ({1 + args.n_decoy_pockets} receptors each)")

            rows = []
            if args.engine == "unidock":
                pdbqt_cache = Path(tmpdir) / "receptors_pdbqt"
                for pair in tqdm(pairs, desc=f"docking {name} (gpu)"):
                    pocket, receptor, scores = dock_one_unidock(
                        pair, pdbqt_cache, max_gpu_mb=args.max_gpu_mb)
                    for i, score in enumerate(scores):
                        rows.append({"arm": name, "pocket": pocket,
                                     "receptor": receptor, "mol_idx": i,
                                     "score": score})
            else:
                with ProcessPoolExecutor(max_workers=args.n_jobs) as pool:
                    futures = [pool.submit(dock_one, pair) for pair in pairs]
                    for future in tqdm(as_completed(futures), total=len(futures),
                                       desc=f"docking {name}"):
                        pocket, receptor, scores = future.result()
                        for i, score in enumerate(scores):
                            rows.append({"arm": name, "pocket": pocket,
                                         "receptor": receptor, "mol_idx": i,
                                         "score": score})
            raw_rows.extend(rows)
            arm_specificity[name] = per_pocket_specificity(rows)

    if not arm_specificity:
        print("Nothing docked.")
        return

    pd.DataFrame(raw_rows).to_csv(out_path.with_name(out_path.stem + "_raw.csv"),
                                  index=False)
    combined = pd.concat(
        [frame.assign(arm=name) for name, frame in arm_specificity.items()],
        ignore_index=True)
    combined.to_csv(out_path, index=False)

    table = paired_report(arm_specificity)
    if table is not None:
        table.to_csv(out_path.with_name(out_path.stem + "_paired.csv"), index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
