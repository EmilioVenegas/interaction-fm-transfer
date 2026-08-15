"""Compare the critic and control training arms across seed replicates.

The r0 comparison (`results/critic_arms/README.md`) reported a +0.6% diffusion
loss penalty for the critic arm from a single pair of runs, with a Wilcoxon test
over the 47 validation points. That test is not one to lean on: successive
validation points are checkpoints of one trajectory and are strongly
autocorrelated, so the effective sample size is nearer 1 than 47. This script
therefore reports two things and keeps them apart:

  * **Within a seed**, the per-seed summary — descriptive only, no test.
  * **Across seeds**, the run-level paired difference with n = number of seeds.
    That is the unit of analysis that answers "is this bigger than run-to-run
    noise", and with three seeds it can only ever be a coarse answer.

Each arm's summary is the mean of its last `--window` validation points, which
is a converged value rather than a single noisy final checkpoint.

    python scripts/compare_seed_arms.py
    python scripts/compare_seed_arms.py --seeds r0 r1 r2 --window 10
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_val(run_dir: Path, column: str) -> pd.DataFrame:
    """Validation points for one metric of one run, indexed by step."""
    csv = run_dir / "metrics.csv"
    if not csv.exists():
        return pd.DataFrame(columns=["step", column])
    df = pd.read_csv(csv)
    if column not in df.columns:
        return pd.DataFrame(columns=["step", column])
    return df[["step", column]].dropna().reset_index(drop=True)


def quartile_means(values: np.ndarray) -> list:
    """Mean of each quarter of a trajectory, in order."""
    return [chunk.mean() for chunk in np.array_split(values, 4)]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--logdir", default="my_logs")
    p.add_argument("--seeds", nargs="+", default=["r0", "r1", "r2"])
    p.add_argument("--critic_prefix", default="critic_graph_cosine")
    p.add_argument("--control_prefix", default="critic_control")
    p.add_argument("--metric", default="loss_diffusion/val",
                   help="the metric both arms log identically, so they compare")
    p.add_argument("--window", type=int, default=10,
                   help="validation points averaged at each end")
    p.add_argument("--out", default="results/critic_arms/seed_training.csv")
    args = p.parse_args()

    logdir = Path(args.logdir)
    rows = []
    for seed in args.seeds:
        critic_dir = logdir / f"{args.critic_prefix}_{seed}"
        control_dir = logdir / f"{args.control_prefix}_{seed}"
        critic = load_val(critic_dir, args.metric)
        control = load_val(control_dir, args.metric)
        if critic.empty or control.empty:
            print(f"seed {seed}: missing {args.metric} "
                  f"(critic {len(critic)} pts, control {len(control)} pts) — skipped")
            continue

        # Only steps both arms validated at are comparable.
        merged = critic.merge(control, on="step", suffixes=("_critic", "_control"))
        c = merged[f"{args.metric}_critic"].values
        k = merged[f"{args.metric}_control"].values
        w = min(args.window, len(c))
        rows.append({
            "seed": seed,
            "n_val_points": len(merged),
            "critic_end": c[-w:].mean(),
            "control_end": k[-w:].mean(),
            "delta_end": c[-w:].mean() - k[-w:].mean(),
            "critic_best": c.min(),
            "control_best": k.min(),
            "critic_worse_at": int((c > k).sum()),
        })

    if not rows:
        print("Nothing to compare.")
        return
    table = pd.DataFrame(rows)

    print(f"\n{args.metric} — mean of the last {args.window} matched validation "
          f"points per run. Lower is better.\n")
    header = (f"{'seed':<6}{'n pts':>7}{'critic':>10}{'control':>10}"
              f"{'delta':>10}{'critic worse at':>18}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['seed']:<6}{r['n_val_points']:>7}{r['critic_end']:>10.5f}"
              f"{r['control_end']:>10.5f}{r['delta_end']:>+10.5f}"
              f"{f'{r_critic_worse(r)}':>18}")

    d = table["delta_end"].values
    print(f"\nAcross {len(d)} seeds, run-level paired difference "
          f"(critic - control):")
    print(f"  mean {d.mean():+.5f}   per-seed {np.array2string(d, precision=5)}")
    if len(d) > 1:
        sem = d.std(ddof=1) / np.sqrt(len(d))
        print(f"  sd {d.std(ddof=1):.5f}, sem {sem:.5f}, "
              f"sign {'+' if (d > 0).all() else ('-' if (d < 0).all() else 'mixed')}")
        print(f"  seeds where the critic is worse: {(d > 0).sum()}/{len(d)}")
        if (d > 0).any() and (d < 0).any():
            print("  The sign is not consistent across seeds, so the difference "
                  "is within run-to-run noise.")
    print("\nNo p-value is reported. With three seeds a paired test has no "
          "power worth quoting; the sign consistency above is the whole signal.")

    # The critic arm's own objective, for the arms that log it.
    print(f"\ncritic_distance/val — the quantity the critic loss penalises "
          f"(critic arm only):\n")
    print(f"{'seed':<6}{'Q1':>11}{'Q2':>11}{'Q3':>11}{'Q4':>11}{'change':>10}"
          f"{'monotone':>10}")
    print("-" * 70)
    crit_rows = []
    for seed in args.seeds:
        cd = load_val(logdir / f"{args.critic_prefix}_{seed}", "critic_distance/val")
        if cd.empty:
            continue
        v = cd["critic_distance/val"].values
        q = quartile_means(v)
        w = min(args.window, len(v) // 2)
        pct = 100.0 * (v[-w:].mean() - v[:w].mean()) / abs(v[:w].mean())
        mono = all(q[i] >= q[i + 1] for i in range(3))
        print(f"{seed:<6}{q[0]:>11.6f}{q[1]:>11.6f}{q[2]:>11.6f}{q[3]:>11.6f}"
              f"{pct:>+9.1f}%{('yes' if mono else 'no'):>10}")
        crit_rows.append({"seed": seed, "critic_distance_pct": pct,
                          "critic_distance_monotone": mono})

    if crit_rows:
        table = table.merge(pd.DataFrame(crit_rows), on="seed", how="left")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\nWrote {out}")


def r_critic_worse(r):
    return f"{r['critic_worse_at']}/{r['n_val_points']}"


if __name__ == "__main__":
    main()
