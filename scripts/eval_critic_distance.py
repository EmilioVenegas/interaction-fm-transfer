"""Measure the critic's own objective for BOTH arms, paired per sample.

`results/critic_arms/README.md` reports that `critic_distance/val` fell 37.9%
over a critic run and reads that as "the critic optimises its own objective".
Two things are wrong with resting on it:

  * **There is no control measurement.** The control arm has the critic
    disabled, so it never logs `critic_distance` at all. A fall in the critic
    arm is therefore uncontrolled: a better denoiser predicts a better `x0_hat`,
    which lowers the ATOMICA distance whether or not the critic is in the loss.
    This project has already been caught assuming a control rather than
    measuring one (`docs/experiment-plan.md`, the `pocket_pool` gate).
  * **The logged metric is noisy enough to swamp the trend.** Each validation
    point averages the distance over only the ~25 of 100 sampled complexes
    whose timestep falls under the ramp, at a *random* `t` each time. Its
    point-to-point sd is about half its mean, and the trend is seed-dependent:
    -37.9% at r0 (p = 0.013) but -7.5% at r1 (p = 0.78).

    (`critic_frac_applied` reads ~0.57 rather than ~0.25 because batches where
    no sample qualifies return before logging anything, so the average is taken
    over non-empty batches only. At batch size 2 a per-sample rate of 0.25 gives
    exactly the 0.571 observed. The metric is conditional, not wrong.)

This script removes both problems. It replays a fixed slice of the validation
set through each arm's checkpoint from a **single shared seed**, so every arm
sees identical complexes, identical timesteps and identical noise; the critic
distance is then paired per sample and the arms differ only in their weights.

    python scripts/eval_critic_distance.py \\
        --config DiffSBDD/configs/crossdock_fullatom_critic.yml \\
        --arms critic_r0=my_logs/critic_graph_cosine_r0/checkpoints/last.ckpt \\
               control_r0=my_logs/critic_control_r0/checkpoints/last.ckpt \\
        --n_batches 250

The config only supplies the architecture and the critic settings; the critic is
forced on for every arm regardless of what that arm trained with, because here
it is a *measurement*, not a loss.
"""

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "DiffSBDD"))

import numpy as np
import pandas as pd
import torch
import yaml

torch.set_float32_matmul_precision("medium")

from lightning_modules import LigandPocketDDPM  # noqa: E402


def build_module(config_path, device):
    with open(config_path) as fh:
        config = yaml.safe_load(fh)
    args = Namespace(**{k: (Namespace(**v) if isinstance(v, dict) else v)
                        for k, v in config.items()})

    # The critic is the instrument here, so it is on for every arm.
    critic_params = getattr(args, "critic_params", None)
    if critic_params is not None:
        critic_params.enabled = True

    histogram = np.load(Path(args.datadir, "size_distribution.npy")).tolist()
    module = LigandPocketDDPM(
        outdir=Path("/tmp/eval_critic_distance"),
        dataset=args.dataset,
        datadir=args.datadir,
        batch_size=args.batch_size,
        lr=args.lr,
        adapter_lr=getattr(args, "adapter_lr", args.lr * 0.01),
        freeze_backbone=getattr(args, "freeze_backbone", False),
        egnn_params=args.egnn_params,
        diffusion_params=args.diffusion_params,
        num_workers=0,          # deterministic ordering, no worker seeding
        augment_noise=args.augment_noise,
        augment_rotation=args.augment_rotation,
        clip_grad=args.clip_grad,
        eval_epochs=args.eval_epochs,
        eval_params=args.eval_params,
        visualize_sample_epoch=args.visualize_sample_epoch,
        visualize_chain_epoch=args.visualize_chain_epoch,
        auxiliary_loss=args.auxiliary_loss,
        loss_params=args.loss_params,
        mode=args.mode,
        node_histogram=histogram,
        pocket_representation=args.pocket_representation,
        virtual_nodes=getattr(args, "virtual_nodes", False),
        critic_params=critic_params,
    )
    module.to(device)
    module.eval()
    return module


def load_arm(module, ckpt_path):
    """Load one arm's weights, leaving the frozen critic untouched.

    `on_save_checkpoint` strips the critic keys, so `strict=False` is required
    and the critic is reported missing. Anything *else* missing means the
    checkpoint does not match this architecture and the numbers would be
    meaningless, so that is fatal rather than a warning.
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = module.load_state_dict(state, strict=False)
    unexpected_real = [k for k in unexpected if not k.startswith("critic.")]
    missing_real = [k for k in missing if not k.startswith("critic.")]
    if missing_real or unexpected_real:
        raise SystemExit(
            f"{ckpt_path}: checkpoint does not match the config architecture.\n"
            f"  missing (non-critic): {missing_real[:5]}\n"
            f"  unexpected (non-critic): {unexpected_real[:5]}")


def run_arm(module, loader, n_batches, seed, device, repeats=1):
    """Per-sample critic distances over a fixed slice of the validation set.

    The critic's own `critic_term` is left untouched; its per-sample distances
    are captured by wrapping the critic call, so this measures exactly what
    training penalises rather than a reimplementation of it.
    """
    captured = []

    original_forward = module.critic.forward

    def capture(*call_args, **call_kwargs):
        out = original_forward(*call_args, **call_kwargs)
        captured.append(out.detach().float().cpu())
        return out

    module.critic.forward = capture
    rows = []
    # Seeded ONCE for the whole pass, not per batch. Reseeding every batch with
    # consecutive integers would make the first draw of each batch -- which is
    # the timestep -- correlated across batches, biasing the very quantity being
    # measured. A single seed is safe because the forward pass consumes random
    # numbers in a way that depends only on tensor shapes, never on the weights,
    # so both arms walk the same sequence. If that ever stopped holding the
    # arms' kept-sample counts would diverge and the merge below would drop the
    # samples rather than pair the wrong ones.
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    try:
        # Only ~25% of complexes draw a timestep under the ramp cutoff, so one
        # pass over the validation set yields a quarter of its size in usable
        # samples. Repeats re-draw the timestep and the noise for the same
        # complexes, which is the cheapest way to buy precision here -- and it
        # stays paired, because both arms replay the identical sequence.
        for rep in range(repeats):
            for i, data in enumerate(loader):
                if i >= n_batches:
                    break
                captured.clear()
                with torch.no_grad():
                    module.forward(data)
                if not captured:
                    continue  # no sample in this batch fell under the ramp
                values = torch.cat(captured).numpy()
                for j, value in enumerate(values):
                    rows.append({"batch": rep * n_batches + i, "slot": j,
                                 "distance": float(value)})
    finally:
        module.critic.forward = original_forward
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="DiffSBDD/configs/crossdock_fullatom_critic.yml")
    p.add_argument("--arms", nargs="+", required=True,
                   help="name=path/to/checkpoint.ckpt, one per arm")
    p.add_argument("--n_batches", type=int, default=250,
                   help="validation batches per arm (batch_size from the config)")
    p.add_argument("--repeats", type=int, default=1,
                   help="passes over the validation slice; each re-draws the "
                        "timestep and noise, multiplying the paired sample size")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/critic_arms/critic_distance_paired.csv")
    args = p.parse_args()

    os.chdir(REPO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = build_module(args.config, device)
    if module.critic is None:
        raise SystemExit("The config built no critic; nothing to measure.")
    module.setup("fit")
    loader = module.val_dataloader()

    frames = {}
    for spec in args.arms:
        name, ckpt = spec.split("=", 1)
        load_arm(module, ckpt)
        frame = run_arm(module, loader, args.n_batches, args.seed, device,
                        repeats=args.repeats)
        print(f"{name:<16} {len(frame):>6} samples   "
              f"mean {frame['distance'].mean():.6f}")
        frames[name] = frame

    names = list(frames)
    # Pairing is by (batch, slot): every arm saw the same batches in the same
    # order at the same seed, so the same complexes at the same timesteps land
    # in the same slots. If an arm's keep-set ever differs, the merge drops that
    # sample rather than silently comparing two different complexes.
    merged = frames[names[0]].rename(columns={"distance": names[0]})
    for name in names[1:]:
        merged = merged.merge(
            frames[name].rename(columns={"distance": name}),
            on=["batch", "slot"], how="inner")
    dropped = {n: len(frames[n]) - len(merged) for n in names}
    if any(dropped.values()):
        print(f"\nDropped unpaired samples: {dropped}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)

    print(f"\nCritic distance on {len(merged)} paired validation samples "
          f"(same complexes, same timesteps, same noise). Lower is better.\n")
    print(f"{'arm':<16}{'mean':>12}{'sem':>12}")
    print("-" * 40)
    for name in names:
        v = merged[name].values
        print(f"{name:<16}{v.mean():>12.6f}{v.std(ddof=1) / np.sqrt(len(v)):>12.6f}")

    if len(names) > 1:
        from scipy.stats import wilcoxon
        base = merged[names[0]].values
        print(f"\nPaired against '{names[0]}':\n")
        print(f"{'arm':<16}{'delta':>12}{'sem':>12}{'lower at':>14}{'wilcoxon p':>13}")
        print("-" * 68)
        for name in names[1:]:
            v = merged[name].values
            d = v - base
            sem = d.std(ddof=1) / np.sqrt(len(d))
            print(f"{name:<16}{d.mean():>+12.6f}{sem:>12.6f}"
                  f"{f'{(d < 0).sum()}/{len(d)}':>14}"
                  f"{wilcoxon(v, base).pvalue:>13.3g}")
        print("\nSamples within a batch share a timestep draw and the pockets "
              "repeat across batches, so treat the p-value as descriptive.")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
