# The two training arms: critic against control, at three seeds

`DiffSBDD/configs/crossdock_fullatom_critic{,_control}{,_r1,_r2}.yml`,
`my_logs/critic_{graph_cosine,control}_r{0,1,2}/metrics.csv`,
`scripts/compare_seed_arms.py`, `scripts/eval_critic_distance.py`,
`seed_training.csv`, `critic_distance_paired.csv`.

Two arms differing in exactly one thing, the objective:

| | critic arm | control arm |
|---|---|---|
| loss | `L_diffusion + λ(t)·d(ATOMICA(pocket, x̂₀), ATOMICA(pocket, x_true))` | `L_diffusion` |
| checkpoint | `crossdocked_fullatom_cond.ckpt` | same |
| trainable | LoRA rank 8, 82,120 params | same |
| data / budget / monitor | 83,921 complexes, 3,000 steps, `loss_diffusion/val` | same |

The configs differ in two lines: the run name and `critic_params.enabled`. All
six runs completed 3,000 optimiser steps with 47 validation points at identical
steps. λ = 0.7, ramp with cutoff 0.25, effective batch 32. Seeds 0, 1, 2.

## Headline: at this λ the critic does not measurably do anything

**Not "it optimised its proxy without moving the target" — it did not reliably
move its proxy either.** Both claims in the single-seed version of this document
were seed artefacts, and the correction runs in the same direction for both.

## The critic does not reduce its own objective against a control

This is the measurement that decides it, and it did not exist before:
`scripts/eval_critic_distance.py` replays the validation set through both arms'
checkpoints from **one shared seed**, so every arm sees identical complexes,
identical timesteps and identical noise, and the distance pairs per sample.

475 paired samples. Lower is better; negative delta favours the critic.

| seed | critic | control | delta | sem | critic lower at | wilcoxon p |
|---|---|---|---|---|---|---|
| r0 | 0.002972 | 0.003168 | **−0.000197** | 0.000063 | 261/475 | 0.006 |
| r1 | 0.003109 | 0.002976 | **+0.000133** | 0.000073 | 223/475 | 0.030 |
| r2 | 0.003067 | 0.003112 | −0.000045 | 0.000059 | 243/475 | 0.849 |

**The sign flips across seeds.** The critic beats its control at r0, *loses* to
it at r1 — both nominally significant — and ties at r2. Mean −1.2% relative,
which is a number with no sign stability behind it.

Why this measurement and not the logged metric: the control arm has the critic
disabled and therefore **never logs `critic_distance` at all**, so the reported
fall had never been compared with anything. That gap matters because a better
denoiser predicts a better `x̂₀`, which lowers the ATOMICA distance whether or
not the critic is in the loss. This project has been caught assuming a control
rather than measuring one before — the `pocket_pool` gate, which outscored the
real metric until a permuted-weight control settled it.

## The logged fall was one seed, and too noisy to carry a trend

`critic_distance/val`, first ten validation points against last ten:

| seed | change | Welch p | Spearman(step) | monotone across quartiles |
|---|---|---|---|---|
| r0 | **−37.9%** | 0.013 | −0.310 (p = 0.034) | yes |
| r1 | −7.5% | 0.782 | −0.084 (p = 0.573) | no |
| r2 | −3.6% | 0.887 | −0.114 (p = 0.444) | no |

The −37.9% that this document previously led with replicates at neither other
seed. The metric's point-to-point sd is about half its mean, because each
validation point averages only ~25 complexes at a *random* `t`. r2 illustrates
the consequence directly: read at 41 validation points it stood at −36.3%, and
its final six points took it to −3.6%.

(`critic_frac_applied` reads ~0.57 rather than the true ~0.25 per-sample rate
because batches where no sample qualifies return before logging anything, so the
average is conditional on non-empty batches. At batch size 2 a rate of 0.25
gives exactly the 0.571 observed. The metric is conditional, not wrong.)

## The diffusion-loss penalty was also noise

`loss_diffusion/val`, mean of the last ten matched validation points — the
metric both arms log identically, so the two are comparable:

| seed | critic | control | delta |
|---|---|---|---|
| r0 | 0.46682 | 0.46476 | +0.00207 |
| r1 | 0.46177 | 0.46186 | −0.00009 |
| r2 | 0.46774 | 0.46663 | +0.00111 |

Sign mixed, critic worse at 2 of 3 seeds. The "+0.6% relative penalty, positive
in every quartile" reported from r0 alone does not survive replication.

No p-value is quoted. The earlier Wilcoxon over 47 validation points treated
successive checkpoints of one trajectory as independent; the unit of analysis
that answers "is this bigger than run-to-run noise" is the **run**, and with
three of them the sign consistency is the whole signal.

## Why this is a null about the dose

`max_weight` was calibrated to give the critic ~10% of the gradient at low
noise, and the ramp applies it to only the ~25% of samples with `t` under the
cutoff. Those multiply: **the critic contributed roughly 2.5% of the total
gradient over training.** A null at that dose is a null about the dose, not
about the method — and it is consistent with everything above, including the
guardrails, which are indistinguishable across all six arms.

The untested variable is therefore λ itself, not the architecture. A sweep at
5× and 20× would separate "the signal is absent" from "the signal was never
applied", and is ~3.3 h per arm.

## What this does not establish

- **λ = 0.7 only.** Nothing here says a stronger critic would fail; it says this
  one did not act.
- **3,000 steps only.** LoRA on 82,120 parameters from a converged checkpoint.
- The paired critic-distance samples share timestep draws within a batch and the
  pockets repeat across batches, so those p-values are descriptive.
- Neither number says anything about the molecules. Pocket specificity is
  measured separately, in `results/specificity/`.

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
non-monotonically. After the fix it fell 37.9% and monotonically, which is what
this document reported for months. The three-seed result above shows both
figures were noise; the artefact was real, but so was the trend it appeared to
repair.
