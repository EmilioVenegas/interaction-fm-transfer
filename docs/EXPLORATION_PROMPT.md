# Exploration prompt — paste into a fresh session

Written 2026-08-16, deliberately for a session with no memory of this one. It
carries the evidence but tries not to carry the pessimism: everything below that
is a *prior* rather than a *measurement* is labelled as such, so it can be
discounted. If the reasoning is wrong, say so — the last two days consisted
largely of finding that confidently-held conclusions in this repository were
artefacts.

---

## The task

You are looking for a variation of this project that produces a **positive**
result, or for a well-argued case that none exists and the work should be
written up as negatives. Do not start by reproducing what is below; start by
deciding which of the open directions is worth an experiment, or by proposing
one that is not listed.

Repository: `~/Documents/atomica-diff-antibiotic/ATOMICA-Diffusion-Antibiotic-design`.
Read `docs/experiment-plan.md` for full state, then `results/critic_arms/README.md`
and `results/specificity/README.md` + `ANALYSIS_PLAN.md`.

## The one measurement that governs everything

ATOMICA is a pretrained interaction foundation model. Two results that look
contradictory and are not:

| task | result |
|---|---|
| rank poses **within one pocket** | AUROC 1.000, clash- and composition-controlled |
| rank poses **across unseen pockets** | 63.9% docking power vs smina's 59.7%, p = 0.65 |

It discriminates interaction geometry sharply *inside* a system and barely
transfers *between* systems. Every design that has worked here was
within-system; every one that failed was cross-system.

**Note the tension this creates, because it may be the whole story.** Pocket
specificity — "does this molecule prefer the pocket it was designed for" — is
inherently a *cross-system* property. Any objective built on ATOMICA is
constrained to the within-system regime. So the current experiment may be
structurally incapable of producing the outcome it measures. That is a
hypothesis, not a finding.

## What is resolved negative (do not redo)

1. **Cross-system pose scoring** — 72 targets, out-of-fold by target. 63.9% vs
   smina 59.7%, p = 0.65. An earlier 22-target positive was retracted after it
   reversed at 72.
2. **Interaction hotspot fields** — median percentile 52.4 against a random floor
   of 52.2; a buriedness control reaches 98.2, proving the harness could detect a
   real field.
3. **Pocket-only encodings are degenerate** — mean pairwise cosine between
   different pockets 1.0000; fixing the block vocabulary gives 0.9917; only
   supplying the second segment (the ligand) gives 0.9248.
4. **The training-time critic**, `L_diffusion + λ(t)·d(ATOMICA(pocket, x̂₀),
   ATOMICA(pocket, x_true))`, frozen ATOMICA, LoRA on the EGNN. Three seeds at
   the calibrated λ = 0.7 and one arm at λ = 14. See below — this one is
   subtler than a flat negative.

Each is paired with a control proving the measurement could have detected the
positive. That property is what makes them evidence rather than absence of
evidence, and it should be preserved in anything new.

## What the critic experiment actually established

At λ = 0.7 (~2.5% of the training gradient) the critic does essentially nothing
measurable. At λ = 14 it does three things and not the fourth:

| | measurement |
|---|---|
| **acts** | reduces its own objective: 2.8 control-seed sd below the control mean, p = 7e-7, measured paired on identical complexes/timesteps/noise |
| **costs** | diffusion loss +0.0078 (~7 sd of the seed spread) |
| **costs, in real units** | docks **0.232 kcal/mol worse** into its own pocket, p = 0.001 |
| **does not help** | pocket specificity −0.055 ± 0.047, 17/44 pockets, p = 0.21, against an MDE of 0.131 |

So the ATOMICA distance is genuinely steerable by gradient descent, and steering
it degrades the molecules without improving pocket discrimination. **The
interesting open question is why**: minimising `d(pred, true)` is regression
toward the reference ligand, which the diffusion loss already performs in
coordinate space. The critic may be adding a noisier copy of an objective the
model already has, rather than new information.

## The measurement you will be judged on, and its precision

`scripts/cross_dock_specificity.py`. Per pocket: mean docking score against 3
decoy pockets from *different targets*, minus the score against its own. Paired
across 44 held-out pockets; report the fraction of pockets improved, never a
pooled per-molecule test.

Current precision, after a fix made on 2026-08-15 (decoys are now drawn once and
shared across arms, rather than each arm drawing its own):

| | per-pocket sd | sem | MDE at 80% power |
|---|---|---|---|
| before the fix | 0.738 | 0.111 | 0.312 |
| **after** | **0.310** | **0.047** | **0.131** |

Docking 40 or all ~95 molecules per pocket instead of 20 would take the MDE to
~0.10; it costs 2.4 h per arm pair per 20 molecules at 28 cores.

**Every effect this project has discussed is below 0.3 kcal/mol.** If a new
direction cannot plausibly clear ~0.15, the experiment will not resolve it and
the design should change before the GPU is spent.

## Open directions, with my priors labelled as priors

**A. Contrastive critic.** Replace `d(pocket, x̂₀) → d(pocket, x_true)` with a
margin: push `d(pocket_own, x̂₀)` below `d(pocket_decoy, x̂₀)`. This is the only
formulation that optimises what the evaluation measures. *Prior: likely fails,
because it needs exactly the cross-system comparison measured at chance in
Phase 2 — but its failure would be sharp and publishable, and my prior could be
wrong since the decoy pocket is a real second segment rather than a pocket-only
encoding.*

**B. Change the metric to match the objective.** The critic is within-system, so
judge it within-system: does the generated ligand reproduce the *reference
ligand's interaction pattern* (PLIP/ProLIF fingerprint recovery, contact-type
agreement)? Cheap — re-analysable from molecules already generated in
`results/gen_*`. *Prior: this is the most likely place a positive is hiding,
because it is the only regime ATOMICA has ever measured well in. It also
reframes the contribution from "better drugs" to "better interface fidelity",
which may or may not be what you want.*

**C. Sampling-time guidance (never tried).** At low noise, take gradients of the
ATOMICA distance through `x̂₀` during sampling rather than during training. It
was gated on the training critic showing signal, which it did not — but the dose
response shows the distance *is* steerable, and guidance acts per-sample instead
of shifting shared weights. Falls back to exact baseline at strength 0. *Prior:
genuinely uncertain, and the cheapest untried thing in the repository.*

**D. Per-target fitting.** The regime that measured at AUROC 1.000 is a probe fit
on the system it scores. That is lead optimisation: given a target with known
actives, fit a per-target model. *Prior: this is what the evidence actually
supports, and it is a different (smaller, more defensible) paper.*

**E. Different readout.** Unit-level rather than graph-level representations; a
learned metric rather than cosine. *Prior: weak — the featurization probe
suggests the level is not the binding constraint.*

**Ruled out, with reasons in the plan:** re-running conditioning with fixed
pocket featurization (Phase 3b), distillation to a pocket-only encoder (dead for
the same reason), and any further probing of the pretrained denoising heads as a
scoring function (three independent measurements put them at a trivial
baseline).

## What would count as a breakthrough

A specificity gain above ~0.15 kcal/mol that survives (a) three seeds, (b) a
matched control differing only in the objective, and (c) the pre-registered
decision rules in `results/specificity/ANALYSIS_PLAN.md`. Anything smaller is
below what this design resolves, and anything unreplicated has a poor track
record here — two headline numbers in this repository were seed artefacts.

## Environment and constraints

- `conda activate ~/.conda/envs/atomica-interface` runs both ATOMICA and DiffSBDD
  (torch 2.0.1 / CUDA 11.8, pytorch-lightning 2.3.3). **Never `pip install`
  anything depending on torch without a constraints file pinning
  `torch==2.0.1`** — it silently swaps in a CPU wheel.
- smina at `~/.conda/envs/smina/bin/smina`; Uni-Dock 1.2.0 at
  `~/.conda/envs/unidock/bin/unidock` — **rejected** for this measurement
  (per-pocket disagreement 1.237 kcal/mol vs smina, likely from `obabel -xr`
  receptor prep; Meeko would be the fix, and it is 35× faster if repaired).
- One 8 GB GPU. The critic arm peaks at ~7.6 GB at batch 2 with gradient
  checkpointing; batch 4 OOMs. Run GPU work sequentially; docking is CPU-only
  and overlaps fine on 32 cores.
- `WANDB_MODE=offline`. Commits: Emilio as sole author, no `Co-Authored-By`.
- Do not delete anything under `data/` without checking.

## Traps, all of which have already cost time here

Nearly every bug in this project was **silent** — code that ran and produced
plausible numbers. Assume the next one is too.

- **A control has to be measured, not assumed.** The critic gate's intended
  negative control outscored the real metric until a permuted-weight run settled
  it. The critic's own objective was reported as improving for months without
  the control arm ever measuring that quantity.
- **Check the unit of analysis.** Validation points within a run are one
  trajectory, not independent samples; molecules are nested within pockets.
- **Check what varies between arms besides the thing you are testing.** The two
  arms were docked against *different decoy pockets* for three seed replicates,
  which alone injects ~0.52 kcal/mol per pocket against effects of ~0.18.
- **Seeds sharing an rng stream replicate the confound, not the effect.**
- `pgrep -f "<pattern>"` matches the shell whose command line contains the
  pattern — including your own. This has now cost time twice.
- `conda run` buffers stdout; watch `metrics.csv` via `scripts/watch_training.py`.
- Gradient calibration depends on what is trainable: λ measured against the
  ATOMICA adapter was 53.6, against LoRA on the EGNN 0.688.
- `rdMolAlign.GetBestRMS` superimposes before measuring and deletes the
  displacement you are testing. Use `CalcRMS`.

## How to start

Argue for one direction in a paragraph, state what would falsify it, write the
decision rule down *before* running anything, and only then use the GPU. If your
read is that no variation clears the bar, say that plainly — the four
controlled negatives are a real result and the honest recommendation may be to
write them up rather than continue.
