# Overnight run plan, 2026-08-15 → 16

Everything below is launched and chained on sentinel files in
`~/.claude/jobs/68fc1d63/tmp/sentinels/`. Nothing needs a human. Times are
estimates from the runs already measured.

## Check it in the morning with

```bash
cd ~/Documents/atomica-diff-antibiotic/ATOMICA-Diffusion-Antibiotic-design
ls ~/.claude/jobs/68fc1d63/tmp/sentinels/          # what has finished
tail -3 ~/.claude/jobs/68fc1d63/tmp/{lambda20,dock_seeds,dock_r0_matched,unidock_pipeline}.log
```

Sentinels in the order they should appear: `DOCK_R2_DONE`,
`LAMBDA20_TRAIN_DONE`, `LAMBDA20_GEN_DONE`, `UNIDOCK_VALIDATED`,
`LAMBDA20_DONE`, `R0_MATCHED_DONE`, then either `UNIDOCK_FULL_DONE` or
`UNIDOCK_REJECTED`.

## The sequence

| # | step | resource | est. finish | driver |
|---|---|---|---|---|
| 1 | r2 cross-docking, smina | 16 cores | ~03:00 | `dock_seeds.sh` |
| 2 | λ20 training, 3,000 steps | GPU | ~00:50 | `lambda20.sh` |
| 3 | λ20 generation, 100/pocket | GPU | ~02:40 | `lambda20.sh` |
| 4 | Uni-Dock validation vs smina | GPU | ~03:00 | `unidock_pipeline.sh` |
| 5 | λ20 cross-docking vs control_r0, smina | 16 cores | ~07:15 | `lambda20.sh` |
| 6 | r0 matched-decoy re-measurement, smina | 28 cores | ~09:45 | `dock_r0_matched.sh` |
| 7 | Uni-Dock full re-dock, **all molecules**, 7 arms | GPU | only if 4 passes | `unidock_pipeline.sh` |

Steps 2–3 and 5–6 are sequential on purpose: one 8 GB card, and the smina runs
would otherwise fight each other for cores.

## What each step answers

**1. r2 cross-docking** completes the three-seed specificity comparison.
Analyse with `scripts/analyse_specificity_seeds.py` — it reproduces r0 exactly,
reports per-seed and pooled results, measures the noise floor from same-arm seed
pairs, and applies the pre-registered decision rules rather than restating them.
Note these three runs all carry the decoy confound (below); they are internally
consistent and that is what the seed comparison needs.

**2–3, 5. The λ20 arm** is the stopping rule for experimental work. At 20× the
calibrated weight the critic takes roughly twice the diffusion term's gradient
in the low-noise band, which is well past where it should be visible if it acts
at all. Three seeds at the calibrated weight showed it does not reduce its own
objective against a matched control (sign flips across seeds), so the open
question is whether that was a null about the method or about the dose: the
critic contributed ~2.5% of the gradient over training. Either outcome is
useful. If the diffusion loss degrades and specificity still does not move, the
negative upgrades to "even at a dose that damages sample quality, this buys no
pocket specificity".

**4. The Uni-Dock gate** decides step 7 by exit code. It compares per-pocket
`cross - own` between engines, not a pooled correlation — the reported statistic
is a difference of means within a pocket, a fraction of a kcal/mol on scores
spanning 6, so a good correlation is easy and uninformative. A constant offset
between engines is fine and cancels; a per-pocket disagreement is not.

**6. The r0 matched-decoy re-measurement** corrects the confound found tonight:
the two arms were never docked against the same decoy pockets (0 of 44 shared a
set), which alone injects ~0.52 kcal/mol per pocket against reported effects of
−0.158 and −0.201. Predictions are registered in
`results/specificity/ANALYSIS_PLAN.md` — read the error bar and the fraction
improved, not the point estimate.

**7. The Uni-Dock full re-dock** is the one that could change what this project
can conclude. Docking all ~95 molecules per pocket instead of 20 removes 45% of
the variance; with matched decoys the MDE goes from 0.31 to ~0.12 kcal/mol.
Every effect discussed in this project so far sits below 0.31.

## Findings from today, already committed

- **The critic does not reduce its own objective against a matched control.**
  475 paired samples, identical complexes/timesteps/noise: −0.000197 at r0
  (p = 0.006), **+0.000133 at r1** (p = 0.030), −0.000045 at r2. Sign flips.
  `results/critic_arms/README.md`, commit `c9b654d`.
- The logged −37.9% fall replicates at neither other seed (−7.5%, −3.6%), and
  the control never logged that metric at all, so it had never been compared
  with anything.
- The +0.6% diffusion-loss penalty is noise: +0.00207, −0.00009, +0.00111.
- **94% of the variance in the specificity effect is harness sampling noise** —
  49% independent decoy draws, 45% the 20-molecule subsample. Commit `3c10da7`.
- Guardrails identical across all six arms (QED 0.505–0.516, validity
  97.0–97.7%, diversity 0.784–0.788).

## If something has failed

Every driver aborts loudly rather than continuing past a failure, which is the
mistake that produced the invalid 2,599-vs-3,000-step comparison. Grep the logs
for `ABORTING`. A failed Uni-Dock gate is **not** a failure — it writes
`UNIDOCK_REJECTED` and stops, leaving the smina results as the record.
