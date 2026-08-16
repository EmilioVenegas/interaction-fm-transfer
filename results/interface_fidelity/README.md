# Interface fidelity: the critic does not improve it, and at λ = 20 it costs

`scripts/interface_fidelity_compare.py`, `primary_result.json`,
`primary_fpa_tanimoto.csv`, `primary_fpb_recall.csv`. Decision rules were fixed
in `ANALYSIS_PLAN.md` and committed before any of these numbers existed; the
gates are in `GATES.md` and all pass.

**Outcome: branch N.** Neither fingerprint clears. The reframe this analysis
existed to test — that the critic works within-system and cross-docking
specificity was simply the wrong readout — **is not supported**.

## The question

The critic is a within-system objective, judged until now only by cross-docking
specificity, which is a cross-system comparison and therefore the regime measured
at chance in Phase 2. So the experiment may have been structurally unable to
reward its own objective. This asks the within-system question directly: do the
generated molecules reproduce the *reference ligand's interaction pattern*?

## Result

`critic_lambda20_r0` vs `control_r0`, paired over 44 held-out pockets. Δ > 0
favours the critic. "Clears" required Δ > 0, |Δ| > `s_arm`, ≥ 30/44 pockets
improved, and p < 0.05.

| statistic | Δ | % of range | improved | p | `s_arm` | |
|---|---|---|---|---|---|---|
| **FP-A Tanimoto, mean** | **−0.00800 ± 0.00214** | −3.22% | **12/44** | **0.0002** | 0.00756 | does not clear |
| FP-A Tanimoto, top-3 | −0.00738 ± 0.00406 | −2.97% | 17/44 | 0.134 | 0.01011 | does not clear |
| FP-B recall, mean | +0.00237 ± 0.00428 | +0.84% | 20/44 | 0.690 | 0.00538 | does not clear |
| FP-B recall, top-3 | +0.00258 ± 0.00415 | +0.92% | 10/44 | 0.639 | 0.00827 | does not clear |

Δ is quoted against the null→ceiling range (FP-A 0.2489, FP-B recall 0.2804), not
against 1.0, because a molecule of the right size dropped in the right place
already scores 0.51 — see `GATES.md`.

## The dose-response is what makes this readable

The λ = 0.7 arms are a matched negative control: three seeds per arm, nine
cross-arm pairs, at a dose known to contribute ~2.5% of the training gradient and
to move the critic's own objective not at all (the paired distance sign-flips
across seeds, `results/critic_arms/`).

| | FP-A Tanimoto | FP-B recall |
|---|---|---|
| λ = 0.7, mean over 9 arm pairs | **−0.00009** | −0.00352 |
| λ = 0.7, largest \|Δ\| of the 9 | 0.00609 | 0.00798 |
| **λ = 20, one seed** | **−0.00800** | +0.00237 |

At λ = 0.7 the critic-vs-control difference is essentially exactly zero. **The
pipeline carries no systematic bias between arms**, so the λ = 20 shift is not an
artefact of arm ordering, molecule counts, or the fingerprint itself — the check
that would have voided this analysis had it failed.

The λ = 20 FP-A shift clears three independent floors: seed-only noise (×1.06),
the largest λ = 0.7 cross-arm excursion (×1.31), and the λ = 0.7 mean (×89).

## What this establishes, and what it does not

**Established: no interface-fidelity gain.** The critic does not make molecules
reproduce the reference interaction pattern better, spatially or chemically, at
either dose. This is a null and needs no replication to state.

**Suggestive, one seed: at λ = 20 it makes contact recovery slightly worse.**
32 of 44 pockets degrade, p = 0.0002, and the dose-response is clean. But it is
one λ = 20 seed, and this project has retracted two headline numbers that were
seed artefacts. The magnitude is small in absolute terms — interpolated onto the
displacement curve in `GATES.md`, −0.008 Tanimoto is about 0.07 Å of equivalent
rigid displacement. Claiming *active degradation* requires the two further λ = 20
seeds; claiming *no improvement* does not.

**FP-B is flat at both doses**, so the critic does not change interaction
chemistry in either direction. There is no chemical collapse (branch C) — FP-A
did not clear, so that branch never applied — and gate 6 independently shows
λ = 20 sits 0.07 pp from the control on chemical validity.

## Why this is the mechanism the other negatives lack

The critic's objective is demonstrably steerable: at λ = 20 it reduces
`d(ATOMICA(pocket, x̂₀), ATOMICA(pocket, x_true))` by 2.8 control-seed sd,
p = 7e-7, paired on identical complexes, timesteps and noise. The question was
always what that buys. The answer is now measured on three independent readouts:

| readout | λ = 20 vs control |
|---|---|
| its own ATOMICA objective | **improves**, p = 7e-7 |
| docking into its own pocket | 0.232 kcal/mol **worse**, p = 0.001 |
| pocket specificity | −0.055 ± 0.047, null against MDE 0.131 |
| **interface fidelity (this work)** | **−0.008 Tanimoto, no gain; contacts slightly worse** |

So minimising the ATOMICA interface distance is not a weak proxy for fitting the
pocket better — it is optimising a quantity that trades *against* it. The
distance falls, the molecule's interface gets no closer to the reference one, and
it docks worse. That is a mechanism, not another absence, and it is what makes
the four negatives publishable as a unit.

The likely reason, stated as the hypothesis it is: `d(pred, true)` is regression
toward the reference ligand in embedding space, which the diffusion loss already
performs in coordinate space. The critic adds a noisier copy of an objective the
model already has, and the noise costs geometry.

## Consequences

- **The critic direction is closed.** No further GPU on it unless someone wants
  the two λ = 20 seeds purely to firm up the degradation claim, which changes no
  decision.
- **Direction C (sampling-time guidance) should not be run** in its current form.
  It was gated on this analysis identifying a target worth guiding toward, and
  the target it would have used is the quantity just shown to trade against
  pocket fit. It is also under-specified: there is no reference ligand at
  sampling time, and the only target-free alternative is the training-free
  denoising energy, at a trivial baseline in three independent measurements.
- **The harness is the asset.** Between `GATES.md` and
  `results/specificity/`, this project now has a specificity metric with a
  measured MDE, matched decoys, seed replication, and a fidelity readout whose
  floor, ceiling, and noise are all quantified — resolving ~3% of its dynamic
  range against cross-docking specificity's ~40%.

## What this cannot establish

- One λ = 20 seed. Two more are needed for any claim of active degradation.
- 44 pockets, one held-out set. The pose-scorer retraction happened at this n.
- Fidelity to one reference ligand per pocket. A molecule binding the same pocket
  by a genuinely different and better mode scores as a failure here. The metric
  rewards imitation; that limit is intrinsic.
- Nothing here is an affinity claim.
