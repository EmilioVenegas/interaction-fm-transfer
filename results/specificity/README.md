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
