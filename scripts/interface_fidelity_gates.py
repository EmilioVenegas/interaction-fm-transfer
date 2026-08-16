"""Validity gates for the interface-fidelity analysis -- FP-A (spatial) only.

Runs everything in `results/interface_fidelity/ANALYSIS_PLAN.md` that does not
require the primary comparison, and deliberately never touches
`critic_lambda20_r0`'s recovery values: the gates decide which decision tree we
walk into, and seeing the treatment arm first would defeat the pre-registration.

Gates covered here: 1 (headroom / placement null), 2 (dynamic range),
3 (true-binder ceiling), 4 (size matching), 5 (same pockets), 7 (no silent
drops), plus the FP-A seed-noise floors at both scales. Gate 6 (protonation) and
the FP-B viability screen need the isolated `ifp` env and run separately.

    python scripts/interface_fidelity_gates.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from interface_fp import (  # noqa: E402
    RECEPTOR_DIR,
    Receptor,
    fp_a,
    heavy_coords,
    load_receptor,
    pocket_names,
    read_sdf,
    recall,
    reference_ligand,
    tanimoto,
)

OUT = Path("results/interface_fidelity")

# The six lambda=0.7 arms. critic_lambda20_r0 is excluded on purpose -- see the
# 2026-08-16 amendment in ANALYSIS_PLAN.md.
CONTROL_ARMS = [f"results/gen_critic_control_r{i}" for i in range(3)]
CRITIC_ARMS = [f"results/gen_critic_graph_cosine_r{i}" for i in range(3)]
GATE_ARMS = CONTROL_ARMS + CRITIC_ARMS
HELD_OUT = "results/gen_critic_lambda20_r0"

N_NULL_PER_POCKET = 100      # placement-null draws
DISPLACEMENTS = [0.5, 1.0, 2.0, 3.0]
N_DIRECTIONS = 12            # random unit vectors per displacement
SAME_SITE_CUTOFF = 8.0       # centroid offset for a valid ceiling pair
SEED = 0


def load_all():
    """Receptors, reference ligands and reference fingerprints, 44/44 asserted."""
    pockets = pocket_names(GATE_ARMS[0])
    receptors, refs, ref_fps = {}, {}, {}
    failed = []
    for p in pockets:
        try:
            rec = load_receptor(f"{RECEPTOR_DIR}/{p}.pdb")
            lig = reference_ligand(p)
            receptors[p], refs[p] = rec, lig
            ref_fps[p] = fp_a(heavy_coords(lig), rec)
        except Exception as exc:  # gate 7: name it, never skip silently
            failed.append((p, repr(exc)))
    return pockets, receptors, refs, ref_fps, failed


def arm_recoveries(arm, pockets, receptors, ref_fps):
    """Per-pocket list of (tanimoto, recall) for every molecule in one arm."""
    out, dropped = {}, {}
    for p in pockets:
        mols = read_sdf(f"{arm}/{p}.sdf")
        vals = []
        n_bad = 0
        for m in mols:
            xyz = heavy_coords(m)
            if len(xyz) == 0:
                n_bad += 1
                continue
            f = fp_a(xyz, receptors[p])
            vals.append((tanimoto(f, ref_fps[p]), recall(f, ref_fps[p])))
        out[p] = np.asarray(vals, dtype=float) if vals else np.empty((0, 2))
        dropped[p] = n_bad
    return out, dropped


def gate_1_placement_null(pockets, receptors, refs, ref_fps, arm_mols, rng):
    """A correctly-sized molecule put in the right place, with no pocket-specific
    chemistry. If this matches the real arms the metric has no headroom."""
    rows = []
    all_pairs = [(p, i) for p in pockets for i in range(len(arm_mols[p]))]
    for p in pockets:
        target = np.asarray(refs[p].GetConformer().GetPositions()).mean(0)
        pool = [q for q in all_pairs if q[0] != p]
        picks = rng.choice(len(pool), size=min(N_NULL_PER_POCKET, len(pool)),
                           replace=False)
        for k in picks:
            q, i = pool[k]
            xyz = arm_mols[q][i]
            moved = xyz - xyz.mean(0) + target
            f = fp_a(moved, receptors[p])
            rows.append((p, tanimoto(f, ref_fps[p]), recall(f, ref_fps[p])))
    return pd.DataFrame(rows, columns=["pocket", "tanimoto", "recall"])


def gate_2_dynamic_range(pockets, receptors, refs, ref_fps, rng):
    """Rigidly translate the reference ligand and watch self-recovery decay.

    Translation only, so RMSD equals the displacement exactly -- verified below
    rather than assumed. GetBestRMS would superimpose it away (see the plan).
    """
    rows = []
    for p in pockets:
        base = heavy_coords(refs[p])
        for d in DISPLACEMENTS:
            for _ in range(N_DIRECTIONS):
                v = rng.normal(size=3)
                v /= np.linalg.norm(v)
                moved = base + v * d
                rmsd = float(np.sqrt(((moved - base) ** 2).sum(1).mean()))
                assert abs(rmsd - d) < 1e-6, (rmsd, d)
                f = fp_a(moved, receptors[p])
                rows.append((p, d, tanimoto(f, ref_fps[p]),
                             recall(f, ref_fps[p])))
    return pd.DataFrame(rows, columns=["pocket", "displacement",
                                       "tanimoto", "recall"])


def gate_3_ceiling(receptors, refs, ref_fps):
    """Two distinct true binders of the same receptor, same site."""
    import lmdb
    import pickle

    pairs = json.loads(Path(OUT / "ceiling_pairs.json").read_text())
    env = lmdb.open("data/crossdocked_pocket10_processed.lmdb", readonly=True,
                    lock=False, subdir=False)
    rows = []
    with env.begin() as txn:
        for p, info in pairs.items():
            if p not in receptors:
                continue
            ref_centroid = np.asarray(
                refs[p].GetConformer().GetPositions()).mean(0)
            for code, meta in info["others"].items():
                rec = pickle.loads(txn.get(meta["lmdb_key"].encode()))
                el = np.asarray(rec["ligand_element"])
                xyz = np.asarray(rec["ligand_pos"], dtype=float)[el > 1]
                offset = float(np.linalg.norm(xyz.mean(0) - ref_centroid))
                f = fp_a(xyz, receptors[p])
                rows.append((p, code, offset, offset <= SAME_SITE_CUTOFF,
                             tanimoto(f, ref_fps[p]), recall(f, ref_fps[p])))
    return pd.DataFrame(rows, columns=["pocket", "ligcode", "centroid_offset",
                                       "same_site", "tanimoto", "recall"])


def seed_noise(per_arm, pockets, stat):
    """Within-arm, seed-only deltas at two scales.

    `s_pocket` is the pocket-level spread (used by the FP-B viability screen);
    `s_arm` is the spread of the mean over pockets (used by the decision table).
    They differ by roughly sqrt(44) and are not interchangeable.
    """
    col = {"tanimoto": 0, "recall": 1}[stat]
    out = {}
    for label, arms in (("control", CONTROL_ARMS), ("critic_l0.7", CRITIC_ARMS)):
        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                mean_d, top3_d = [], []
                for p in pockets:
                    a, b = per_arm[arms[i]][p][:, col], per_arm[arms[j]][p][:, col]
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


def main():
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    pockets, receptors, refs, ref_fps, failed = load_all()
    print(f"[gate 7] receptors+references parsed: "
          f"{len(receptors)}/{len(pockets)}   failures: {failed}")
    assert not failed, "gate 7 failed -- fix the parser, do not skip pockets"

    ref_bits = np.array([len(ref_fps[p]) for p in pockets])
    print(f"          FP-A reference bits: med {np.median(ref_bits):.0f} "
          f"p10 {np.percentile(ref_bits,10):.0f} min {ref_bits.min()} "
          f"max {ref_bits.max()}")

    # --- per-arm recoveries and raw coordinates (gate arms only) --------------
    per_arm, per_arm_xyz, dropped_tot = {}, {}, {}
    for arm in GATE_ARMS:
        per_arm[arm], dropped = arm_recoveries(arm, pockets, receptors, ref_fps)
        dropped_tot[arm] = int(sum(dropped.values()))
        print(f"  scored {arm}: "
              f"{sum(len(v) for v in per_arm[arm].values())} molecules, "
              f"{dropped_tot[arm]} unusable")

    # coordinates pooled across the six gate arms, for the placement null
    for p in pockets:
        pool = []
        for arm in GATE_ARMS:
            for m in read_sdf(f"{arm}/{p}.sdf"):
                xyz = heavy_coords(m)
                if len(xyz):
                    pool.append(xyz)
        per_arm_xyz[p] = pool

    # --- gate 5: same pockets everywhere -------------------------------------
    same = {arm: pocket_names(arm) == pockets for arm in GATE_ARMS + [HELD_OUT]}
    print(f"[gate 5] identical pocket sets across arms: {all(same.values())} "
          f"({len(pockets)} pockets)")

    # --- gate 4: size matching ------------------------------------------------
    sizes = {}
    for arm in GATE_ARMS + [HELD_OUT]:
        n = [len(heavy_coords(m)) for p in pockets
             for m in read_sdf(f"{arm}/{p}.sdf")]
        sizes[arm] = (float(np.mean(n)), float(np.std(n)), len(n))
    lo = min(v[0] for v in sizes.values())
    hi = max(v[0] for v in sizes.values())
    print(f"[gate 4] mean heavy atoms across arms: {lo:.2f}-{hi:.2f} "
          f"(spread {hi-lo:.2f})")

    # --- gate 1 ---------------------------------------------------------------
    null_df = gate_1_placement_null(pockets, receptors, refs, ref_fps,
                                    per_arm_xyz, rng)
    null_df.to_csv(OUT / "gate1_placement_null.csv", index=False)
    null_by_pocket = null_df.groupby("pocket")[["tanimoto", "recall"]].mean()
    real_by_pocket = pd.DataFrame(
        {p: np.nanmean(np.vstack([per_arm[a][p] for a in GATE_ARMS]), axis=0)
         for p in pockets}, index=["tanimoto", "recall"]).T
    print(f"[gate 1] placement null   tanimoto {null_by_pocket.tanimoto.mean():.4f}"
          f"   recall {null_by_pocket.recall.mean():.4f}")
    print(f"         real arms (pooled) tanimoto "
          f"{real_by_pocket.tanimoto.mean():.4f}"
          f"   recall {real_by_pocket.recall.mean():.4f}")

    # --- gate 2 ---------------------------------------------------------------
    dyn = gate_2_dynamic_range(pockets, receptors, refs, ref_fps, rng)
    dyn.to_csv(OUT / "gate2_dynamic_range.csv", index=False)
    curve = dyn.groupby("displacement")[["tanimoto", "recall"]].mean()
    print("[gate 2] self-recovery vs rigid displacement:")
    for d, row in curve.iterrows():
        print(f"         {d:>4} A   tanimoto {row.tanimoto:.4f}   "
              f"recall {row.recall:.4f}")

    # --- gate 3 ---------------------------------------------------------------
    ceil = gate_3_ceiling(receptors, refs, ref_fps)
    ceil.to_csv(OUT / "gate3_ceiling.csv", index=False)
    ok = ceil[ceil.same_site]
    print(f"[gate 3] ceiling pairs: {len(ceil)} candidate, {len(ok)} same-site, "
          f"over {ok.pocket.nunique()}/{len(pockets)} pockets")
    print(f"         ceiling tanimoto  med {ok.tanimoto.median():.4f} "
          f"IQR [{ok.tanimoto.quantile(.25):.4f}, "
          f"{ok.tanimoto.quantile(.75):.4f}]")
    print(f"         ceiling recall    med {ok.recall.median():.4f} "
          f"IQR [{ok.recall.quantile(.25):.4f}, "
          f"{ok.recall.quantile(.75):.4f}]")

    # --- seed-noise floors ----------------------------------------------------
    noise = {s: seed_noise(per_arm, pockets, s) for s in ("tanimoto", "recall")}
    for stat, d in noise.items():
        s_arm_mean = max(abs(v["mean_arm_delta"]) for v in d.values())
        s_arm_top3 = max(abs(v["top3_arm_delta"]) for v in d.values())
        s_pkt_mean = max(v["mean_pocket_sd"] for v in d.values())
        s_pkt_top3 = max(v["top3_pocket_sd"] for v in d.values())
        print(f"[noise/{stat}] s_arm  mean {s_arm_mean:.4f}  top3 {s_arm_top3:.4f}")
        print(f"[noise/{stat}] s_pocket mean {s_pkt_mean:.4f}  "
              f"top3 {s_pkt_top3:.4f}")

    summary = {
        "n_pockets": len(pockets),
        "gate7_parse_failures": failed,
        "gate7_unusable_molecules": dropped_tot,
        "gate5_same_pockets": {k: bool(v) for k, v in same.items()},
        "gate4_sizes": sizes,
        "ref_bits": {"median": float(np.median(ref_bits)),
                     "min": int(ref_bits.min()), "max": int(ref_bits.max())},
        "gate1_null": {"tanimoto": float(null_by_pocket.tanimoto.mean()),
                       "recall": float(null_by_pocket.recall.mean())},
        "gate1_real_pooled": {"tanimoto": float(real_by_pocket.tanimoto.mean()),
                              "recall": float(real_by_pocket.recall.mean())},
        "gate2_curve": curve.to_dict(),
        "gate3_ceiling": {
            "n_candidate": int(len(ceil)), "n_same_site": int(len(ok)),
            "n_pockets": int(ok.pocket.nunique()),
            "tanimoto_median": float(ok.tanimoto.median()),
            "recall_median": float(ok.recall.median())},
        "seed_noise": noise,
    }
    (OUT / "gates_fpa.json").write_text(json.dumps(summary, indent=1))

    # per-pocket table the viability screen consumes
    tbl = null_by_pocket.rename(columns=lambda c: f"null_{c}")
    tbl = tbl.join(real_by_pocket.rename(columns=lambda c: f"real_{c}"))
    cs = ok.groupby("pocket")[["tanimoto", "recall"]].agg(["median", "size"])
    cs.columns = ["ceiling_tanimoto", "n_pairs", "ceiling_recall", "_n2"]
    tbl = tbl.join(cs.drop(columns="_n2"))
    tbl.to_csv(OUT / "gates_fpa_per_pocket.csv")
    print(f"\nwrote {OUT}/gates_fpa.json and gates_fpa_per_pocket.csv")


if __name__ == "__main__":
    main()
