# Cross-docking specificity: the critic does not improve pocket fit

`scripts/cross_dock_specificity.py`, `specificity.csv`, `specificity_raw.csv`,
`specificity_paired.csv`. Analysis rules were fixed in advance in
`ANALYSIS_PLAN.md`, committed before this run finished.

**Outcome: branch D, inconclusive — and it reads as a null leaning against the
critic.** No evidence that the ATOMICA critic improves pocket specificity.

## Setup

Two arms differing only in the training objective (`results/critic_arms/`), each
sampled for the same 44 held-out pockets over 44 distinct targets. 20 molecules
per pocket docked into its own pocket and into 3 pockets from *other* targets,
smina exhaustiveness 8, 7,040 docking pairs.

```
specificity(pocket) = mean score against other targets - mean score against own
```

smina is lower-is-better, so positive means the molecules prefer the pocket they
were designed for.

## Validity checks (all pass)

| Check | Result |
|---|---|
| NaN rate below 10% | **0.00%** of 7,040 pairs |
| Both arms show positive absolute specificity | critic +0.232, control +0.390 |
| Same pockets in both arms | 44 / 44 |

The second is the one that matters: molecules in both arms genuinely prefer
their own pocket, so the harness detects pocket fit. A null here is a null about
the critic, not about the measurement.

## Result

| arm | own | cross | specificity | pockets with specificity > 0 |
|---|---|---|---|---|
| critic | −7.596 | −7.364 | **+0.232 ± 0.128** | 29/44 |
| control | −7.508 | −7.118 | **+0.390 ± 0.142** | 28/44 |

Paired across the 44 pockets, Δ = specificity(critic) − specificity(control):

| | |
|---|---|
| mean Δ | **−0.158 ± 0.111** (sem) |
| median Δ | −0.222 |
| pockets where critic > control | **18 / 44 (41%)** |
| Wilcoxon signed-rank | **p = 0.299** |

Against the pre-registered thresholds this is branch D by a single pocket:
branch B (null) required 19–25 of 44 and p > 0.3; this gives 18 and p = 0.2994.
Substantively it is a null with a slight lean against the critic, and it is
nowhere near either significance threshold.

## The informative part: the critic made molecules that dock better *everywhere*

The critic arm docks **better in absolute terms** than the control — into its own
pocket (−7.596 vs −7.508, Δ = −0.088, p = 0.20) *and* into other targets'
pockets (−7.364 vs −7.118, a larger gap). Because the improvement against other
pockets is bigger than the improvement against its own, specificity goes **down**.

That is exactly the failure mode this metric exists to detect. It is the same
shape as the A/B ablation's QED gain: an improvement that is real on a
target-independent axis and absent on the target-dependent one. Had we reported
mean docking score, the critic would have looked mildly good. Neither difference
is significant, so this is a direction, not a finding — but it is the direction
the whole evaluation was designed to expose.

## Guardrails: chemistry is unchanged

| arm | n | QED | SA | MW | diversity |
|---|---|---|---|---|---|
| critic | 4,176 | 0.505 ± 0.190 | 4.690 ± 1.177 | 298.7 ± 101.8 | 0.894 |
| control | 4,184 | 0.514 ± 0.189 | 4.601 ± 1.215 | 297.5 ± 104.1 | 0.894 |

Indistinguishable. The critic did not buy specificity with degenerate chemistry,
nor cost diversity. It simply did not change what was generated in any way this
measures. Generation validity was 97.8% (critic) and 97.6% (control).

## Reading this alongside the training result

The critic term itself was optimised: `critic_distance/val` fell **37.9%**
monotonically over 3,000 steps (`results/critic_arms/`). So the model did learn
to reduce the ATOMICA interface distance between its predicted ligand and the
reference one — and that reduction **did not transfer** into pocket specificity.

Two nominal costs, neither significant on its own: +0.6% on the diffusion loss
during training, −0.158 kcal/mol on specificity here.

The coherent reading, consistent with the rest of the project: ATOMICA's
within-system discrimination is real (Phase 0, AUROC 1.000; the gate, 0.926
against a 0.697 permuted-weight control) but reducing its distance metric is not
the same as producing a better-fitting ligand. The critic optimised the proxy
without moving the target.

## What would settle it

**Seeds, not more training.** One seed per arm cannot separate a −0.158
difference from run-to-run noise, and the question is variance rather than
convergence. Two further seeds per arm are running per the pre-registered plan
for branch D.

If the lean against the critic holds across three seeds, this becomes a fourth
well-controlled negative — and a sharper one than the others, because the
critic's own objective demonstrably improved while the outcome did not. If it
washes out, the honest statement is that the critic is neutral at λ = 0.7.

## What this cannot establish

- 44 pockets, one held-out set, one seed per arm so far. The pose-scorer
  retraction happened at this sample size.
- 20 of ~95 molecules per pocket, randomly subsampled at seed 0.
- λ = 0.7 and the ramp cutoff of 0.25 were calibrated, not tuned for outcome. A
  larger λ was never tried, and the critic's influence may simply be too small
  to matter — its gradient share was set to ~10% at low noise.
- Cross-docking specificity is a proxy for pocket fit, not a binding assay.

---

# Retracted: `specificity_r0_matched*.csv` is an invalid run, kept deliberately

**Do not use these three files as evidence.** `specificity_r0_matched.csv`,
`_paired.csv` and `_raw.csv` are retained because the way they failed is a
methodological result, not because their numbers mean anything. They are the
matched-decoy re-measurement of the λ = 0.7 arms, and they failed two of the
three validity checks pre-registered in `ANALYSIS_PLAN.md`.

## What it reported, and why that is a trap

It produced **+0.157 ± 0.064, 24/38 pockets improved, p = 0.043** — the first
nominally significant positive in this project and a sign flip of the −0.158
headline. It is not usable.

Note also that the addendum to `ANALYSIS_PLAN.md` had already pre-registered
that the point estimate of this run was *not to be read*: the subsampling rng
moved to its own stream, so the 20 molecules are redrawn and the estimate is a
near-independent draw rather than a correction. "Returning at +0.1 is not a
reversal" is in the plan, written before the run finished. Only the error bar
and the fraction improved were ever readable. The number was tempting precisely
because it was significant, and the pre-registration is what disarmed it.

## Why it is invalid

Failure rates by arm across every docking run:

| run | `n_jobs` | NaN by arm |
|---|---|---|
| r0 original | 16 | 0.00% / 0.00% |
| r1 | 16 | 2.84% / 1.70% |
| r2 | 16 | 0.00% / 0.00% |
| λ = 20 (matched) | 16 | 0.00% / 0.00% |
| **r0 matched** | **28** | **15.34% / 1.14%** |

One smina call docks all 20 of a pocket's molecules serially, ~700 s nominal.
Raising `n_jobs` from 16 to 28 on a 32-core machine pushed the slower pockets
past the 1800 s timeout, and a timeout returns NaN for the **entire cell**.

The damage landed asymmetrically. Ligand size and flexibility are effectively
identical between the arms (20.55 vs 20.58 heavy atoms, 4.29 vs 4.39 rotatable
bonds), so the 13× gap is not the molecules — it tracks **which arm ran second**,
later on a machine that had been at full load for hours. Five pockets lost their
own-pocket docking in the control arm and none in the critic arm, so the paired
comparison ran on 38 of 44 pockets with attrition concentrated in one arm.

That fails validity check 2 (NaN below 10%) and validity check 3 (both arms
contributing the same pockets). By the plan's own rule the result is about the
harness, not the critic.

## Two lessons, both carried forward

**The NaN criterion was wrong, not just the timeout.** "NaN below 10%" was
written assuming failures are random. They are not — they concentrate in
whichever arm runs second under drifting load. A 6% rate split 5.5% / 0.5% would
pass that gate and be just as biased. **Any future docking run should require 0%
NaN, or per-arm equality, rather than a pooled ceiling.**

**Raising the timeout is necessary but not sufficient.** `77d76e0` raised it to
5400 s and returned `n_jobs` to 16. Because smina at `--seed 0 --cpu 1` is
deterministic, contention cannot change scores — only whether a cell finishes —
so a timeout that never fires does fix this. But the *ordering* asymmetry remains
latent for any future time-varying condition. The structural fix, not yet
implemented, is to **interleave the work queue by (pocket, arm)** instead of
running arm A then arm B, so that drift hits both arms equally.

A timeout that fires in normal operation is not a safety net; it is a silent
sampler.

## Status

The requeued run (`specificity_r0_matched2.csv`, `n_jobs` 16, timeout 5400 s) was
killed ~36 minutes in when the ATOMICA direction closed, so it was never written.
The λ = 0.7 matched-decoy comparison is therefore **unresolved**, and the r0
figure quoted above in this document retains its independent-decoy confound.

**This costs the write-up nothing.** The paper's specificity claim rests on the
λ = 20 arm (−0.055 ± 0.047 against an MDE of 0.131), which ran at `n_jobs` 16
with 0.00% NaN in both arms, and on the variance decomposition, which is
computed from the r0 raw scores independently of this run.
