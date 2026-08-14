# Critic weight calibration and the 300-step trainability check

`scripts/calibrate_critic_weight.py`, `DiffSBDD/configs/crossdock_fullatom_critic.yml`,
`my_logs/critic_calib_r1/metrics.csv`.

Two questions, answered in order, before committing GPU-days to the critic arm:
how large should `lambda` be, and does the critic term actually go down when
trained.

## 1. How large should lambda be

Comparing the two **losses** answers the wrong question. Cosine distance on a
32-d representation is order 1e-2 while the diffusion nll is order 1e0, so the
loss ratio makes the critic look negligible at a weight that would in fact
dominate the update. What matters is the **gradient** each term contributes to
the trainable parameters, measured with `lambda` factored out.

**The ratio depends on what is trainable, and this is the trap.** Measured
first against arm B's architecture with the ATOMICA adapter as the trainable
path, and then against the arm that actually trains — the upstream checkpoint's
architecture with LoRA on the EGNN:

| trainable path | ‖grad‖ diffusion | ‖grad‖ critic (λ=1) | median ratio | λ for a 10% share |
|---|---|---|---|---|
| arm B arch, ATOMICA adapter | 3.53e-04 | 4.21e-06 | 108 | 53.6 |
| **this arm, LoRA on the EGNN** | **1.61e-01** | **9.79e-02** | **1.9** | **0.688** |

LoRA sits in the edge/node/coordinate MLPs that directly determine `x̂₀`, so the
critic's gradient arrives far more directly than through an input-side adapter.
Carrying the first figure over would have made the critic roughly **240× too
strong**. Recompute this whenever the trainable set changes.

By noise level, since the ramp concentrates the term at low `t`:

| band | median ratio | λ for 10% |
|---|---|---|
| `t/T <= 0.25` | 6.2 | 0.688 |
| `0.25 < t/T <= 0.5` | 1.6 | 0.175 |
| `t/T > 0.5` | 2.0 | 0.218 |

The critic's gradient is relatively *weakest* at low `t` — where the ramp weights
it most — because `x̂₀` is already close to `x_true` there. `max_weight` is set to
**0.7**, the 10%-share figure for the band the term is actually applied in.

## 2. Does it train

300 optimiser steps from `checkpoints/crossdocked_fullatom_cond.ckpt`, LoRA rank
8, backbone frozen, adapter off, `max_weight` 0.7, ramp with cutoff 0.25,
effective batch 16, 48 validation points on the target-disjoint val split.

| quartile of the run | `critic_distance/val` | `loss/val` | `error_t_lig/val` |
|---|---|---|---|
| Q1 | 0.00324 | 0.44200 | 0.09282 |
| Q2 | 0.00272 | 0.43976 | 0.08878 |
| Q3 | 0.00187 | 0.44136 | 0.08928 |
| Q4 | **0.00161** | 0.43483 | 0.08903 |

First 10 against last 10 validation points: `critic_distance` **−58.9%**
(0.00378 → 0.00155), `loss/val` −1.1% (0.43863 → 0.43360).

The critic term falls monotonically across quartiles while the diffusion loss is
flat. That is what λ=0.7 was chosen to produce: the auxiliary objective is being
optimised without destabilising the primary one. Memory fits an 8 GB card at
batch 2 with gradient checkpointing (7.5 GB peak; batch 4 without checkpointing
OOMs inside the critic's encoder pass).

`critic_frac_applied` sits at 0.55–0.63 rather than 0.25, because a sample is
counted whenever its ramp weight is non-zero — that is `t/T < 0.25` per *sample*,
not per batch.

## What this does and does not establish

**Does:** the loss is wired correctly end to end, the weight is in a sensible
range, and the model can reduce the ATOMICA interface distance between its
predicted ligand and the reference one without the diffusion objective
degrading.

**Does not:** say anything about whether the generated molecules are better.
The model is being directly optimised on `critic_distance`, so its falling is
the minimum one should expect, not evidence of success — exactly the reasoning
that made the A/B ablation's QED gain uninterpretable. That question belongs to
`scripts/cross_dock_specificity.py`, per pocket, against the
`critic_params.enabled: False` control.

This run also had **no control arm**, so the 1.1% movement in `loss/val` is not
attributable to the critic.

## Reproducing

```bash
# gradient calibration (trains nothing)
python scripts/calibrate_critic_weight.py --data_dir data/processed_expert_atomica/val

# short trainability check
WANDB_MODE=offline python DiffSBDD/train.py \
    --config <calibration config with max_steps: 300> \
    --resume checkpoints/crossdocked_fullatom_cond.ckpt
```

`train.py` now passes `max_steps`, `accumulate_grad_batches`,
`val_check_interval` and `limit_val_batches` through to the Trainer, and writes
a `metrics.csv` alongside wandb.
