"""The primary interface-fidelity comparison, against the pre-registered table.

Computes FP-A and FP-B recovery per molecule for the named arms, caches them to
`results/interface_fidelity/recov_<arm>.npz` so nothing is ever recomputed, and
applies the decision rules in `ANALYSIS_PLAN.md`.

    ~/.conda/envs/ifp/bin/python scripts/interface_fidelity_compare.py \
        --arms results/gen_critic_lambda20_r0 results/gen_critic_control_r0
    ~/.conda/envs/ifp/bin/python scripts/interface_fidelity_compare.py --decide

Runs in the `ifp` env (ProLIF). Parallel over pockets -- the FP-B gate run was
single-threaded and took an hour while the box was saturated by docking.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = Path("results/interface_fidelity")
CONTROL_ARMS = [f"results/gen_critic_control_r{i}" for i in range(3)]
CRITIC07_ARMS = [f"results/gen_critic_graph_cosine_r{i}" for i in range(3)]
LAMBDA20 = "results/gen_critic_lambda20_r0"

# Decision statistics, fixed in the plan: FP-A tanimoto and FP-B recall.
CELLS = [("fpa", "tanimoto"), ("fpb", "recall")]


def _one_pocket(args):
    """(pocket, arm) -> arrays of per-molecule recovery for both fingerprints."""
    pocket, arm = args
    import interface_fp as ifp
    import interface_fidelity_fpb as fpb

    rec_a = ifp.load_receptor(f"{ifp.RECEPTOR_DIR}/{pocket}.pdb")
    ref_a = ifp.fp_a(ifp.heavy_coords(ifp.reference_ligand(pocket)), rec_a)

    prot, err = fpb.load_receptor_prolif(pocket)
    if prot is None:  # gate 7: name it, never skip silently
        raise RuntimeError(f"{pocket}: receptor unusable for FP-B: {err}")
    ref_mols, _ = fpb.read_sdf_fpb(f"{ifp.RECEPTOR_DIR}/{pocket}_ref_ligand.sdf")
    ref_b = fpb.compute_fp(ref_mols[0], prot) if ref_mols else frozenset()

    rows = []
    # FP-A reads coordinates only and must not sanitize; FP-B needs valences.
    geo = ifp.read_sdf(f"{arm}/{pocket}.sdf")
    chem, counts = fpb.read_sdf_fpb(f"{arm}/{pocket}.sdf")
    for m in geo:
        f = ifp.fp_a(ifp.heavy_coords(m), rec_a)
        rows.append((ifp.tanimoto(f, ref_a), ifp.recall(f, ref_a),
                     np.nan, np.nan, np.nan))
    out_a = np.asarray(rows, dtype=float)

    rows_b = []
    for m in chem:
        f = fpb.compute_fp(m, prot)
        rows_b.append((ifp.tanimoto(f, ref_b), ifp.recall(f, ref_b), len(f)))
    out_b = (np.asarray(rows_b, dtype=float) if rows_b
             else np.empty((0, 3), dtype=float))
    return pocket, out_a[:, :2], out_b, counts


def compute_arm(arm: str, pockets: list, n_jobs: int):
    cache = OUT / f"recov_{Path(arm).name}.npz"
    if cache.exists():
        print(f"  cached: {cache.name}")
        return
    with Pool(n_jobs) as pool:
        res = pool.map(_one_pocket, [(p, arm) for p in pockets])
    store = {}
    for pocket, a, b, counts in res:
        store[f"a__{pocket}"] = a
        store[f"b__{pocket}"] = b
    np.savez_compressed(cache, **store)
    n_a = sum(v.shape[0] for k, v in store.items() if k.startswith("a__"))
    n_b = sum(v.shape[0] for k, v in store.items() if k.startswith("b__"))
    print(f"  wrote {cache.name}: FP-A {n_a} mols, FP-B {n_b} mols")


def load_arm(arm: str, pockets: list):
    z = np.load(OUT / f"recov_{Path(arm).name}.npz")
    return ({p: z[f"a__{p}"] for p in pockets},
            {p: z[f"b__{p}"] for p in pockets})


def paired(arm_a, arm_b, pockets, fp, stat, rng):
    """Per-pocket mean and top-3, subsampled to n_min with a shared rng.

    Order statistics are biased upward by larger n and the arms have unequal
    molecule counts, so both arms are cut to the same n per pocket from one
    draw -- the decoy confound, avoided in advance.
    """
    col = {"tanimoto": 0, "recall": 1}[stat]
    idx = 0 if fp == "fpa" else 1
    rows = []
    for p in pockets:
        a = arm_a[idx][p][:, col]
        b = arm_b[idx][p][:, col]
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        n = min(len(a), len(b))
        if n == 0:
            continue
        ia = rng.choice(len(a), n, replace=False)
        ib = rng.choice(len(b), n, replace=False)
        a, b = a[ia], b[ib]
        k = min(3, n)
        rows.append((p, a.mean(), b.mean(),
                     np.sort(a)[-k:].mean(), np.sort(b)[-k:].mean()))
    return pd.DataFrame(rows, columns=["pocket", "a_mean", "b_mean",
                                       "a_top3", "b_top3"])


def report(label, df, s_arm_mean, s_arm_top3, rng_range):
    out = {}
    for stat, sa in (("mean", s_arm_mean), ("top3", s_arm_top3)):
        d = (df[f"a_{stat}"] - df[f"b_{stat}"]).values
        n = len(d)
        imp = int((d > 0).sum())
        p = stats.wilcoxon(d).pvalue if n > 1 and np.any(d != 0) else float("nan")
        clears = bool(d.mean() > 0 and abs(d.mean()) > sa
                      and imp >= 30 and p < 0.05)
        out[stat] = {"delta": float(d.mean()), "sem": float(d.std(ddof=1) / np.sqrt(n)),
                     "improved": imp, "n": n, "p": float(p),
                     "s_arm": sa, "pct_of_range": 100 * float(d.mean()) / rng_range,
                     "clears": clears}
        print(f"  {label:26s} {stat:5s}  delta {d.mean():+.5f} +- "
              f"{out[stat]['sem']:.5f}  ({out[stat]['pct_of_range']:+.2f}% of range)"
              f"  improved {imp}/{n}  p {p:.4f}  s_arm {sa:.5f}  "
              f"{'CLEARS' if clears else 'does not clear'}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--decide", action="store_true")
    ap.add_argument("--n_jobs", type=int, default=8)
    args = ap.parse_args()

    import interface_fp as ifp
    pockets = ifp.pocket_names(CONTROL_ARMS[0])

    for arm in args.arms:
        print(f"computing {arm}")
        compute_arm(arm, pockets, args.n_jobs)

    if not args.decide:
        return

    gA = json.loads((OUT / "gates_fpa.json").read_text())
    gB = json.loads((OUT / "gates_fpb.json").read_text())
    ranges = {
        ("fpa", "tanimoto"): gA["gate3_ceiling"]["tanimoto_median"] - gA["gate1_null"]["tanimoto"],
        ("fpb", "recall"): gB["gate3_ceiling"]["recall_median"] - gB["gate1_null"]["recall"],
    }
    floors = {}
    for (fp, stat) in CELLS:
        sn = (gA if fp == "fpa" else gB)["seed_noise"][stat]
        floors[(fp, stat)] = (max(abs(v["mean_arm_delta"]) for v in sn.values()),
                              max(abs(v["top3_arm_delta"]) for v in sn.values()))

    l20 = load_arm(LAMBDA20, pockets)
    ctl0 = load_arm(CONTROL_ARMS[0], pockets)

    print("\n=== PRIMARY: critic_lambda20_r0 vs control_r0 ===")
    primary = {}
    for (fp, stat) in CELLS:
        rng = np.random.default_rng(0)
        df = paired(l20, ctl0, pockets, fp, stat, rng)
        sa_mean, sa_top3 = floors[(fp, stat)]
        primary[f"{fp}_{stat}"] = report(f"{fp} {stat}", df, sa_mean, sa_top3,
                                         ranges[(fp, stat)])
        df.to_csv(OUT / f"primary_{fp}_{stat}.csv", index=False)

    print("\n=== CONSISTENCY CHECK: lambda=0.7 critic vs control (3v3 seeds) ===")
    consistency = {}
    for (fp, stat) in CELLS:
        deltas = []
        for ci in range(3):
            for ti in range(3):
                rng = np.random.default_rng(100 + 10 * ci + ti)
                df = paired(load_arm(CRITIC07_ARMS[ci], pockets),
                            load_arm(CONTROL_ARMS[ti], pockets),
                            pockets, fp, stat, rng)
                deltas.append((df["a_mean"] - df["b_mean"]).mean())
        consistency[f"{fp}_{stat}"] = {
            "mean_delta": float(np.mean(deltas)),
            "max_abs_delta": float(np.max(np.abs(deltas))),
            "n_pairs": len(deltas)}
        print(f"  {fp} {stat}: mean over 9 arm pairs {np.mean(deltas):+.5f}, "
              f"max |delta| {np.max(np.abs(deltas)):.5f}")

    verdict = {"primary": primary, "consistency": consistency, "ranges":
               {f"{k[0]}_{k[1]}": v for k, v in ranges.items()}}
    (OUT / "primary_result.json").write_text(json.dumps(verdict, indent=1))
    print(f"\nwrote {OUT}/primary_result.json")


if __name__ == "__main__":
    main()
