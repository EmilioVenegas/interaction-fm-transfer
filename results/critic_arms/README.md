# The two training arms: critic against control

`DiffSBDD/configs/crossdock_fullatom_critic.yml` and
`..._critic_control.yml`, `my_logs/critic_graph_cosine_r0/metrics.csv`,
`my_logs/critic_control_r0/metrics.csv`.

Two runs differing in exactly one thing, the objective:

| | critic arm | control arm |
|---|---|---|
| loss | `L_diffusion + λ(t)·d(ATOMICA(pocket, x̂₀), ATOMICA(pocket, x_true))` | `L_diffusion` |
| checkpoint | `crossdocked_fullatom_cond.ckpt` | same |
| trainable | LoRA rank 8, 82,120 params | same |
| data / budget / monitor | 83,921 complexes, 3,000 steps, `loss_diffusion/val` | same |

The configs differ in two lines: the run name and `critic_params.enabled`.

Both completed **3,000 optimiser steps** with **47 validation points at
identical steps**. λ = 0.7, ramp with cutoff 0.25, effective batch 32.

## The critic term is optimised

`critic_distance/val`, the quantity the loss actually penalises:

| quartile | value |
|---|---|
| Q1 | 0.005954 |
| Q2 | 0.004367 |
| Q3 | 0.004150 |
| Q4 | **0.003973** |

Monotone across quartiles; −37.9% comparing the first ten validation points to
the last ten. The model does learn to reduce the ATOMICA interface distance
between its predicted ligand and the reference one.

## It costs a little on the diffusion objective

`loss_diffusion/val` — the diffusion loss alone, logged identically by both arms
so the two are comparable (`loss/val` would not be, since it contains the critic
term for one arm only):

| quartile | critic | control | difference |
|---|---|---|---|
| Q1 | 0.47369 | 0.47185 | +0.00185 |
| Q2 | 0.47067 | 0.46885 | +0.00182 |
| Q3 | 0.46910 | 0.46338 | +0.00572 |
| Q4 | 0.46623 | 0.46480 | +0.00143 |

Mean paired difference **+0.00273 ± 0.00174** (sem), critic worse at **28 of 47**
validation points, Wilcoxon p = 0.119. Best value reached: critic 0.45368,
control 0.45034.

The penalty is small — about 0.6% relative — and positive in every quartile, but
it is not significant, and the significance test should not be leaned on anyway
(see below).

## What this does not establish

- **One seed per arm.** Run-to-run variance is unestimated, so a difference of
  0.6% cannot be separated from seed noise. Repeating both arms at two or three
  seeds is the only way to know.
- **The 47 validation points are not independent.** They are successive
  checkpoints of a single training trajectory and are strongly autocorrelated,
  so the Wilcoxon p = 0.119 is optimistic. Treat it as descriptive, not as a
  test.
- **Neither number says the molecules are better.** The critic arm is directly
  optimised on `critic_distance`, so its falling is the minimum to expect. This
  is the same trap as the A/B ablation's QED gain: an improvement in the thing
  being optimised is not evidence about the thing being asked. Pocket
  specificity is measured by `scripts/cross_dock_specificity.py`, per pocket,
  against the control.

## The first attempt at this comparison was invalid

Worth recording, because the numbers it produced looked reportable.

The critic arm crashed at step 2,599 of 3,000 with `IndexError: invalid index of
a 0-dim tensor` in `critic_term`. `t_int` arrives squeezed, so a batch of
exactly one complex makes it 0-dim; the train split holds 83,921 complexes, an
odd number, so at batch size 2 the last batch of each epoch is a singleton and
the crash landed exactly at the first epoch boundary. The control never enters
`critic_term` and ran to 3,000. The driver script continued past the failure,
producing a comparison of 2,599 steps against 3,000 — precisely the
unequal-budget confound the fixed step count exists to prevent.

That truncated run reported `critic_distance` falling only 20.5% and
non-monotonically (Q1 0.0044 → Q2 0.0058 → Q3 0.0064 → Q4 0.0035). After the
fix it falls 37.9% and monotonically. The artefact was in the run, not the
method.

The control arm was not repeated: none of the fixes touch a code path it
executes, and it completed its full budget.
