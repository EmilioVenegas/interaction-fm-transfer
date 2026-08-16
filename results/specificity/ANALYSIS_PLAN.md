# Pre-registered analysis plan for the cross-docking comparison

**Written and committed before the docking run finished, deliberately.** This
project has already retracted one result that reversed between 22 and 72 targets
(`results/pose_scorer/README.md`), and the cause was partly that six feature sets
were evaluated and the best reported. Fixing the decision rules before seeing the
numbers is cheap insurance against doing that again.

Run in flight: `scripts/cross_dock_specificity.py`, 2 arms × 44 pockets ×
(own + 3 decoy pockets from *different targets*) × 20 molecules, exhaustiveness
8, seed 0. Started 2026-08-14 22:36.

## The measurement

Per pocket *p*:

```
specificity(p) = mean smina score against 3 other targets' pockets
               - mean smina score against p's own pocket
```

smina is lower-is-better, so **positive specificity means the molecules prefer
the pocket they were designed for**. A generically drug-like molecule docks about
as well anywhere and scores ≈ 0, however good its absolute affinity or QED. That
is the whole point: it is the measurement the A/B ablation lacked, which is why
its QED +13.9% could not distinguish "better pocket fit" from "more generically
drug-like".

Primary comparison: Δ = specificity(critic) − specificity(control), **paired per
pocket**, n = 44 pockets. Not per molecule — the ~4,300 molecules per arm are
nested within 44 pockets and treating them as independent is pseudo-replication.

Reported: mean Δ ± sem, **fraction of pockets improved**, Wilcoxon signed-rank p.

## Validity checks, run first

If any of these fails, the comparison is not interpretable and the finding is
about the harness, not the critic:

1. **Both arms must show positive absolute specificity.** If molecules do not
   prefer their own pocket in *either* arm, the harness is not measuring pocket
   fit. The smoke test gave +0.428 kcal/mol over 3 pockets, so this is expected
   to pass.
2. **NaN rate below ~10%.** Failed docking pairs return NaN; a high rate means
   the scores are a biased subsample.
3. **Both arms must contribute the same pockets.** Paired statistics require it.

## Decision rules, fixed in advance

n = 44 pockets. "Improved" = Δ > 0 for that pocket.

| Outcome | Criterion | Reading | What I do |
|---|---|---|---|
| **A. Clear positive** | ≥ 30/44 improved **and** p < 0.05 | The critic improves pocket specificity | Write up as **suggestive, pending replication**, then launch 2 further seeds per arm |
| **B. Null** | 19–25/44 improved, p > 0.3 | The critic optimises its own objective (−37.9%) without changing what is generated | Write up as a **controlled negative**; stop this direction |
| **C. Clear negative** | ≤ 14/44 improved **and** p < 0.05 | The critic actively costs sample quality, consistent with its +0.6% diffusion-loss penalty | Write up as a negative; stop |
| **D. Ambiguous** | anything else — most likely | Underpowered, not resolved | Write up as **inconclusive**, then launch 2 further seeds per arm |

**A is not a positive result on its own.** One seed per arm cannot separate a
small effect from seed noise, and the training comparison already showed the
arms differing by only 0.6% on the diffusion loss. Branch A is written up as
"suggestive, pending replication" and nothing stronger.

## Why seeds rather than more training, in branches A and D

The open question in both is whether a small difference is real, and that is a
question about variance, not about convergence. Two more seeds per arm gives 3
independent replicates and lets the comparison be made between-run rather than
between two single trajectories. Another 3,000 steps would not.

Cost: ~3.3 h per critic arm, ~1.1 h per control, so ~8.8 h for two more seeds of
both — overnight GPU that is otherwise idle. Seeds go in
`crossdock_fullatom_critic{,_control}.yml` copies with `run_name` suffixed `_r1`,
`_r2` and a distinct `seed`, everything else identical.

**I will launch these automatically in branches A and D** and will not in B or C.
Kill them with `pkill -f train.py` if they are not wanted.

## Regardless of outcome

- **Ligand-only guardrails** (QED, SA, Lipinski, validity, diversity) computed
  for both arms with `scripts/evaluate.py`. Guardrails only: they are
  target-independent and cannot evidence pocket specificity. Reported so a
  specificity gain bought by degenerate chemistry is visible.
- **Absolute docking scores** per arm alongside specificity, since an arm could
  improve specificity while docking worse overall.
- Everything written to `results/specificity/README.md` with the raw per-pair
  scores kept in `specificity_raw.csv`.

## What this cannot establish either way

- One seed per arm (until the replicates land).
- 44 pockets, one held-out set. The pose-scorer retraction happened at this
  sample size.
- 20 of ~95 molecules per pocket are docked, randomly subsampled at seed 0.
- Cross-docking specificity is a proxy for pocket fit, not a measure of binding.
  Nothing here is an affinity claim.

---

# Addendum, 2026-08-15: the decoy confound, and predictions registered before
# the re-measurement

Written and committed **before** the matched-decoy run finished, for the same
reason as the original plan.

## What was wrong

`cross_dock_specificity.py` drew decoy pockets inside the per-arm loop, from one
rng the arms consumed in turn. How many random numbers the molecule subsampling
happened to use therefore decided what the next arm got, and **0 of 44 pockets
shared a decoy set between the critic and the control** — in the r0, r1 and r2
runs alike.

Variance decomposition of the r0 per-pocket delta, computed from
`specificity_raw.csv`:

| source | per-pocket sd | share of variance |
|---|---|---|
| independent decoy draws (3 per pocket) | 0.519 | **49%** |
| the 20-molecule subsample (of ~95) | 0.495 | **45%** |
| observed total | 0.738 | 100% |

Molecule-to-molecule score sd within one pocket-receptor cell is 1.356 kcal/mol.
**94% of the variance in the reported effect is harness sampling noise**, leaving
sd ≈ 0.18 for any real difference between arms.

The reported effects are −0.158 (r0) and −0.201 (r1). Both sit inside what the
decoy assignment alone produces (~0.52 per pocket between two arms drawing
independently).

**The seed replicates could not have caught this.** They were given the identical
rng stream deliberately, so only the trained model would differ — which also
handed every replicate the *same* decoy assignment. r0 and r1 agreeing to within
0.04 kcal/mol replicates the confound, not the effect. That agreement is what
made it visible: the paired critic-distance measurement shows these two arms are
nearly indistinguishable, and nearly indistinguishable models should not produce
a stable specificity gap.

Fixed in `af0ebde`: decoys are drawn once, before any arm, from an rng
independent of the subsampling stream, and shared by every arm.

## Predictions, registered in advance

For `specificity_r0_matched.csv` — same molecules, same pockets, same
parameters, decoys now shared between the arms:

1. **The error bar shrinks by ~30%.** Per-pocket sd 0.738 → ~0.525, sem
   0.111 → ~0.079. This is close to arithmetic and is the point of the fix.
2. **The point estimate is a near-independent redraw, not a correction.** The
   subsampling rng moved to its own stream, so the 20 molecules are redrawn too;
   between them the two noise terms are 94% of the variance. Expected delta in
   **[−0.2, +0.2]**, pockets improved **~22/44**, **p > 0.2**.
3. **Returning near −0.16 is not replication** — it is one draw agreeing with
   another. **Returning at +0.1 is not a reversal.** Only the error bar and the
   fraction improved should be read.
4. **The own-pocket difference is unaffected by the decoy fix** (both arms always
   docked into their own pocket); any change in the delta must come from the
   cross term.

## How this changes the decision rules

The branch thresholds in the original plan stand, but they are now applied to a
measurement whose MDE is known: **0.31 kcal/mol at 80% power** with 44 pockets
and 20 molecules. Every effect discussed in this project so far is below that.
The correct statement for any null here is "no effect large enough for this
design to see", with the bound quoted — never "no effect".

Getting under that bound needs molecules, not seeds: docking ~95 per pocket
instead of 20 cuts the molecule term 2.2× and takes the MDE to ~0.12. That is
4.75× the docking cost, which is why GPU docking is being set up.
