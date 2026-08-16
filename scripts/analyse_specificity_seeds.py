"""Cross-docking specificity across seed replicates.

The r0 comparison came out at -0.158 +- 0.111 kcal/mol with 18/44 pockets
improved and p = 0.299 -- branch D of `results/specificity/ANALYSIS_PLAN.md`,
inconclusive, which is what put two further seeds on the GPU. This script turns
those replicates into the comparison the plan asked for, and it is deliberately
structured so that the decision rules stay the ones fixed in advance.

Two levels, kept apart because they answer different questions:

  * **Per seed** -- the r0 statistic recomputed for each replicate. Shows
    whether the lean against the critic is a property of the method or of one
    training run.
  * **Across seeds** -- the per-pocket delta averaged over seeds, then tested
    across the 44 pockets. Averaging first is what buys the precision: pocket
    identity is the dominant variance term, and it is shared by every seed.

The between-seed spread is reported alongside every mean. A mean of three
replicates whose signs disagree is a null however small its standard error, and
that distinction is the whole reason the replicates were run.

    python scripts/analyse_specificity_seeds.py
    python scripts/analyse_specificity_seeds.py \\
        --paired results/specificity/specificity_paired.csv \\
                 results/specificity/specificity_r1_paired.csv \\
                 results/specificity/specificity_r2_paired.csv
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Fixed in ANALYSIS_PLAN.md before the r0 numbers existed. Not to be adjusted
# after seeing the replicates -- that is precisely the failure this project has
# already had once (`results/pose_scorer/README.md`, the retracted 22-target
# result).
DECISION_RULES = [
    ("A. Clear positive", lambda n_imp, n, p: n_imp >= 30 * n / 44 and p < 0.05,
     "The critic improves pocket specificity"),
    ("B. Null", lambda n_imp, n, p: 19 * n / 44 <= n_imp <= 25 * n / 44 and p > 0.3,
     "The critic optimises its own objective without changing what is generated"),
    ("C. Clear negative", lambda n_imp, n, p: n_imp <= 14 * n / 44 and p < 0.05,
     "The critic actively costs sample quality"),
]


def seed_of(column: str) -> str:
    """'critic_r1__specificity' -> 'r1'; r0's columns carry no suffix."""
    match = re.search(r"_(r\d+)__", column)
    return match.group(1) if match else "r0"


def arm_of(column: str) -> str:
    """'control_r1__specificity' -> 'control'."""
    return re.sub(r"_r\d+$", "", column.split("__")[0])


def load_paired(paths):
    """One tidy frame: pocket x seed x arm -> specificity, own, cross."""
    records = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            print(f"  {path} not found — skipped")
            continue
        frame = pd.read_csv(path)
        for column in frame.columns:
            if "__" not in column:
                continue
            arm, measure = arm_of(column), column.split("__")[1]
            for pocket, value in zip(frame["pocket"], frame[column]):
                records.append({"pocket": pocket, "seed": seed_of(column),
                                "arm": arm, "measure": measure, "value": value})
    if not records:
        raise SystemExit("No paired specificity files found.")
    tidy = pd.DataFrame(records)
    return tidy.pivot_table(index=["pocket", "seed"], columns=["arm", "measure"],
                            values="value").reset_index()


def wilcoxon_p(a, b):
    try:
        from scipy.stats import wilcoxon
        return wilcoxon(a, b).pvalue
    except Exception:
        return float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--paired", nargs="+", default=[
        "results/specificity/specificity_paired.csv",
        "results/specificity/specificity_r1_paired.csv",
        "results/specificity/specificity_r2_paired.csv"])
    p.add_argument("--out", default="results/specificity/specificity_seeds.csv")
    args = p.parse_args()

    table = load_paired(args.paired)
    delta = pd.DataFrame({
        "pocket": table["pocket"],
        "seed": table["seed"],
        "critic_spec": table[("critic", "specificity")],
        "control_spec": table[("control", "specificity")],
        "critic_own": table[("critic", "own")],
        "control_own": table[("control", "own")],
        "critic_cross": table[("critic", "cross")],
        "control_cross": table[("control", "cross")],
    })
    delta["delta_spec"] = delta["critic_spec"] - delta["control_spec"]
    delta["delta_own"] = delta["critic_own"] - delta["control_own"]
    delta["delta_cross"] = delta["critic_cross"] - delta["control_cross"]

    seeds = sorted(delta["seed"].unique())
    print(f"\n{len(seeds)} seed replicate(s): {', '.join(seeds)}\n")

    print("Absolute specificity per arm (positive = molecules prefer their own "
          "pocket).\nA null here would be a null about the harness, not the "
          "critic.\n")
    header = (f"{'seed':<6}{'pockets':>9}{'critic':>10}{'control':>10}"
              f"{'critic>0':>10}{'control>0':>11}")
    print(header)
    print("-" * len(header))
    for seed in seeds:
        sub = delta[delta["seed"] == seed]
        print(f"{seed:<6}{len(sub):>9}{sub['critic_spec'].mean():>+10.3f}"
              f"{sub['control_spec'].mean():>+10.3f}"
              f"{f'{(sub.critic_spec > 0).sum()}/{len(sub)}':>10}"
              f"{f'{(sub.control_spec > 0).sum()}/{len(sub)}':>11}")

    print("\nPaired per pocket, delta = specificity(critic) - "
          "specificity(control). Positive favours the critic.\n")
    header = (f"{'seed'    :<6}{'n':>5}{'mean delta':>13}{'sem':>8}"
              f"{'median':>9}{'improved':>11}{'wilcoxon p':>12}")
    print(header)
    print("-" * len(header))
    per_seed = []
    for seed in seeds:
        sub = delta[delta["seed"] == seed]
        d = sub["delta_spec"].values
        p_value = wilcoxon_p(sub["critic_spec"].values, sub["control_spec"].values)
        per_seed.append({"seed": seed, "n": len(d), "mean": d.mean(),
                         "improved": int((d > 0).sum()), "p": p_value})
        print(f"{seed:<6}{len(d):>5}{d.mean():>+13.3f}"
              f"{d.std(ddof=1) / np.sqrt(len(d)):>8.3f}{np.median(d):>+9.3f}"
              f"{f'{(d > 0).sum()}/{len(d)}':>11}{p_value:>12.3g}")

    # Averaging over seeds first: pocket identity is the dominant variance term
    # and is shared by every replicate, so this is the precise comparison. It is
    # only legitimate when every seed contributes every pocket, which is checked.
    counts = delta.groupby("pocket")["seed"].nunique()
    complete = counts[counts == len(seeds)].index
    if len(complete) < len(counts):
        print(f"\n{len(counts) - len(complete)} pocket(s) missing from some "
              f"seed — excluded from the pooled test.")
    pooled = (delta[delta["pocket"].isin(complete)]
              .groupby("pocket")[["delta_spec", "delta_own", "delta_cross"]]
              .mean())

    if len(seeds) > 1:
        d = pooled["delta_spec"].values
        p_value = wilcoxon_p(d, np.zeros_like(d))
        print(f"\nPooled: per-pocket delta averaged over {len(seeds)} seeds, "
              f"then tested across {len(d)} pockets.\n")
        print(f"  mean delta   {d.mean():+.3f} +- {d.std(ddof=1) / np.sqrt(len(d)):.3f} (sem)")
        print(f"  median       {np.median(d):+.3f}")
        print(f"  improved     {(d > 0).sum()}/{len(d)} pockets")
        print(f"  wilcoxon p   {p_value:.3g}")

        means = np.array([r["mean"] for r in per_seed])
        signs = "all negative" if (means < 0).all() else (
            "all positive" if (means > 0).all() else "MIXED")
        print(f"\n  between-seed spread of the mean delta: "
              f"{np.array2string(means, precision=3)}  ({signs})")
        if signs == "MIXED":
            print("  The sign is not stable across replicates, so the r0 lean "
                  "was run-to-run noise.")
        else:
            print(f"  sd across seeds {means.std(ddof=1):.3f}; the effect keeps "
                  f"its sign in every replicate.")

        print("\nDecomposition — where any difference comes from "
              "(negative = the critic docks better):\n")
        print(f"  own pocket    {pooled['delta_own'].mean():+.3f}")
        print(f"  other pockets {pooled['delta_cross'].mean():+.3f}")
        print("  Specificity falls when the gain against other pockets exceeds "
              "the gain against its own —\n  a molecule that docks better "
              "everywhere, which is the failure mode this metric exists to "
              "detect.")

        # The noise floor, measured rather than assumed. Two runs of the SAME
        # arm differ only by seed, so the spread of specificity(arm, seed i) -
        # specificity(arm, seed j) is what this pipeline produces when nothing
        # about the method has changed. If the critic-vs-control difference is
        # not larger than that, there is nothing to explain.
        print("\nNoise floor: the same arm at two different seeds, paired per "
              "pocket.\nNothing differs but the seed, so this is what the "
              "pipeline produces from no effect at all.\n")
        print(f"  {'comparison':<28}{'mean delta':>12}{'sem':>8}{'improved':>11}")
        print("  " + "-" * 59)
        floor = []
        for arm in ("critic", "control"):
            column = f"{arm}_spec"
            wide = delta[delta["pocket"].isin(complete)].pivot(
                index="pocket", columns="seed", values=column)
            for i, seed_a in enumerate(seeds):
                for seed_b in seeds[i + 1:]:
                    diff = (wide[seed_b] - wide[seed_a]).values
                    floor.append(diff.mean())
                    print(f"  {f'{arm} {seed_b} - {seed_a}':<28}"
                          f"{diff.mean():>+12.3f}"
                          f"{diff.std(ddof=1) / np.sqrt(len(diff)):>8.3f}"
                          f"{f'{(diff > 0).sum()}/{len(diff)}':>11}")
        if floor:
            print(f"\n  Same-arm differences span "
                  f"{min(floor):+.3f} to {max(floor):+.3f}; the critic-vs-control "
                  f"difference is {d.mean():+.3f}.")
            print("\n  READ THIS BEFORE COMPARING THOSE TWO NUMBERS. They do not\n"
                  "  have the same structure, and the difference flatters the\n"
                  "  critic-vs-control result:\n"
                  "\n"
                  "    * same-arm, across seeds -- decoys are MATCHED. Every\n"
                  "      replicate was given the identical rng stream so that only\n"
                  "      the trained model differed, so critic_r0 and critic_r1 drew\n"
                  "      the same decoy pockets. This floor therefore contains model\n"
                  "      seed and molecule-subsample noise but NO decoy variance.\n"
                  "    * critic-vs-control -- decoys are UNMATCHED in these runs, and\n"
                  "      that term alone contributes sd ~0.078 to the mean.\n"
                  "\n"
                  "  So the floor bounds the seed and subsample terms (<= "
                  f"{max(np.abs(floor)):.3f}), which is\n"
                  "  useful, but it does NOT bound this comparison. Nor does the\n"
                  "  agreement across seeds argue for a real effect: all three share\n"
                  "  the same decoy assignment, so a decoy artefact would reproduce\n"
                  "  exactly this consistently. Only the matched-decoy re-measurement\n"
                  "  separates the two.")

        # What this design could have detected. A null is only informative if
        # the experiment had the power to see the effect it is nulling, which is
        # the same logic as the buriedness control in the hotspot work: show the
        # measurement can detect a positive before reporting that it did not.
        mde = 2.80 * d.std(ddof=1) / np.sqrt(len(d))  # 1.96 + 0.84, two-sided
        print(f"\nDetectable effect: with {len(d)} pockets at a per-pocket sd of "
              f"{d.std(ddof=1):.3f},\nthe smallest difference this design "
              f"resolves at 80% power is {mde:.3f} kcal/mol.")
        if abs(d.mean()) < mde:
            print(f"The observed {d.mean():+.3f} is below that, so this is "
                  f"'no effect large enough to see',\nnot 'no effect'. Reporting "
                  f"it as a bound is honest; reporting it as zero is not.")

        n = len(d)
        n_improved = int((d > 0).sum())
        verdict = next((name for name, rule, _ in DECISION_RULES
                        if rule(n_improved, n, p_value)), "D. Ambiguous")
        reading = next((text for name, _, text in DECISION_RULES
                        if name == verdict), "Underpowered, not resolved")
        print(f"\nPre-registered decision (ANALYSIS_PLAN.md), applied to the "
              f"pooled result:\n  {verdict} — {reading}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    delta.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
