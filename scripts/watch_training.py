"""Live view of a training run from its metrics.csv.

The obvious thing -- tailing the driver's stdout -- does not work here: runs are
launched through `conda run`, which captures stdout and releases it only when
the process exits, so that log stays empty for hours. The CSVLogger's
`metrics.csv` is written and flushed during the run, so it is the thing to
watch.

Compares any number of runs side by side, which is the point when a critic arm
and its control have to be judged against each other.

    python scripts/watch_training.py                       # all runs under my_logs
    python scripts/watch_training.py critic_graph_cosine_r0 critic_control_r0
    python scripts/watch_training.py --follow              # refresh every 30 s
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

# Metrics worth showing, in display order. Anything absent is skipped, so the
# same call works for a critic arm and for a control that logs no critic keys.
TRACKED = [
    ("loss_diffusion", "diffusion loss"),
    ("critic_distance", "critic dist"),
    ("critic_weighted", "critic weighted"),
    ("critic_frac_applied", "frac applied"),
    ("error_t_lig", "error_t_lig"),
    ("loss", "total loss"),
]


def summarise(run_dir: Path, split: str, window: int):
    csv = run_dir / "metrics.csv"
    if not csv.exists():
        return None
    try:
        df = pd.read_csv(csv)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return None  # mid-write; try again next tick

    rows = []
    for key, label in TRACKED:
        col = f"{key}/{split}"
        if col not in df.columns:
            continue
        s = df[["step", col]].dropna()
        if s.empty:
            continue
        v = s[col].values
        # A trend needs two non-overlapping windows. With fewer points the two
        # slices share most of their data and the percentage is meaningless --
        # report it as unavailable rather than as a small number.
        if len(v) >= 2 * window:
            first, last = v[:window].mean(), v[-window:].mean()
            pct = 100.0 * (last - first) / abs(first) if first else float("nan")
        else:
            last, pct = v[-min(window, len(v)):].mean(), float("nan")
        rows.append((label, v[-1], last, pct, len(v)))

    step = int(df["step"].max()) if "step" in df and not df["step"].isna().all() else 0
    return step, rows


def render(runs, split, window, max_steps):
    out = []
    stamp = time.strftime("%H:%M:%S")
    out.append(f"[{stamp}]  split={split}  trend = mean of last {window} vs first {window}")
    for run_dir in runs:
        got = summarise(run_dir, split, window)
        if got is None:
            out.append(f"\n{run_dir.name}: no metrics yet")
            continue
        step, rows = got
        progress = f"step {step}"
        if max_steps:
            pct = 100.0 * step / max_steps
            progress += f"/{max_steps} ({pct:.0f}%)"
        out.append(f"\n{run_dir.name}  --  {progress}")
        if not rows:
            out.append(f"  (no {split} metrics yet)")
            continue
        out.append(f"  {'metric':<18}{'latest':>12}{'recent mean':>14}{'trend':>10}{'n':>6}")
        for label, latest, last, pct, n in rows:
            if pd.isna(pct):
                trend = f"{'n/a':>9}      "
            else:
                arrow = "down" if pct < 0 else ("up" if pct > 0 else "flat")
                trend = f"{pct:>+8.1f}% {arrow:<5}"
            out.append(f"  {label:<18}{latest:>12.5f}{last:>14.5f}{trend}{n:>4}")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("runs", nargs="*", help="run names under --logdir; default all")
    p.add_argument("--logdir", default="my_logs")
    p.add_argument("--split", default="train", choices=("train", "val"))
    p.add_argument("--window", type=int, default=10,
                   help="points averaged at each end for the trend")
    p.add_argument("--max_steps", type=int, default=3000,
                   help="for the progress percentage; 0 to hide")
    p.add_argument("--follow", action="store_true", help="refresh until interrupted")
    p.add_argument("--interval", type=float, default=30.0)
    args = p.parse_args()

    logdir = Path(args.logdir)
    if args.runs:
        runs = [logdir / r for r in args.runs]
    else:
        runs = sorted(d for d in logdir.iterdir()
                      if d.is_dir() and (d / "metrics.csv").exists())
    if not runs:
        print(f"No runs with metrics.csv under {logdir}")
        return

    while True:
        text = render(runs, args.split, args.window, args.max_steps)
        if args.follow:
            os.system("clear")
        print(text, flush=True)
        if not args.follow:
            return
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()
