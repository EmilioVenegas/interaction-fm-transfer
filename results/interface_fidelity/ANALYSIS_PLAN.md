# Pre-registered plan: does the critic improve *interface fidelity*?

**Written 2026-08-16, before any number in this directory exists.** Same reason as
`results/specificity/ANALYSIS_PLAN.md`: this project has already retracted one
result that reversed between 22 and 72 targets, and two headline numbers that
turned out to be seed artefacts. Fixing the readout and the thresholds before
seeing them is the cheapest insurance available.

Nothing here needs the GPU. It re-analyses molecules that already exist in
`results/gen_*`, on CPU, in minutes.

## The argument for this direction

The governing measurement is that ATOMICA discriminates interaction geometry
sharply *within* a complex (AUROC 1.000, Phase 0; 0.926 against a 0.697
permuted-weight control, the critic gate) and barely *between* complexes (63.9%
vs smina's 59.7%, p = 0.65). The critic was designed to stay on the within-system
side of that line, and it does. **But it has only ever been judged by a
cross-system readout** — cross-docking specificity asks whether a molecule
prefers its own pocket over three other targets' pockets, which is exactly the
comparison measured at chance in Phase 2. So the experiment as it stands may be
structurally incapable of rewarding the objective it trains.

At λ = 20 the critic is known to do three things and not a fourth: it reduces its
own objective (2.8 control-seed sd, p = 7e-7, paired on identical
complexes/timesteps/noise), costs 0.0078 on the diffusion loss, docks 0.232
kcal/mol worse into its own pocket, and does not move specificity
(−0.055 ± 0.047 against an MDE of 0.131). The open question the exploration
prompt identifies is *why*, and the leading hypothesis is that minimising
`d(ATOMICA(pocket, x̂₀), ATOMICA(pocket, x_true))` is regression toward the
reference ligand — a noisier copy of something the diffusion loss already does in
coordinate space.

That hypothesis is directly testable, within-system, for free, on molecules
already on disk. **Ask whether the critic arm's molecules reproduce the reference
ligand's interaction pattern better than the control's.** It is the only readout
in this project aligned with the regime ATOMICA actually works in, and both
outcomes are informative:

- **Positive** — the critic does buy interface fidelity, and specificity was the
  wrong readout. The contribution reframes from "better molecules" to "better
  interface reproduction", there is a real fidelity-vs-docking trade-off to
  report, and a dose-response between λ = 0.7 and λ = 20 becomes worth GPU time.
- **Negative** — reducing the ATOMICA distance does not make the generated
  interface resemble the reference interface *even within one complex*, despite
  the distance provably falling. That is the sharpest negative in the project:
  not "the representation does not transfer" but "the quantity we optimised was
  never a proxy for the thing we wanted, in the one regime where the
  representation works." It supplies the mechanism behind negative #4 rather
  than another null.

**Stated against it, plainly:** fidelity to a *known* reference ligand's contacts
is imitation, not design. A positive result here is a weaker claim than a
specificity win and is not the breakthrough defined in the exploration prompt. It
is worth running because it costs nothing and because it decides what happens
next, not because it is the paper.

### Why this before direction C (sampling-time guidance)

C is currently under-specified. Guidance takes gradients of "the ATOMICA
distance" through `x̂₀` at sampling time — but *distance to what*? The reference
ligand is unavailable at sampling time for a novel target, and the only
target-free alternative is the training-free denoising energy, which three
independent measurements place at a trivial baseline (`results/hotspot/`,
`results/pose_scorer/`) and which the plan explicitly says to stop probing.
This analysis decides what target C would even use, and whether the quantity is
worth guiding on. Running C first would spend GPU on an objective we have not
established corresponds to anything.

## The measurement: two co-primary fingerprints, spatial and chemical

A purely geometric cutoff fingerprint is vulnerable to **chemical collapse**: if
the critic merely condenses the molecule into a denser mass of carbon against the
pocket wall, it trips the distance flags without reproducing any of the reference
interaction's chemistry — a hydrophobic ring shoved where a hydrogen bond
belongs. A geometric metric alone would score that as success. So the readout is
two fingerprints, and **both are primary**:

- **FP-A — spatial.** Binary vector over receptor residues; bit set if any ligand
  heavy atom is within 4.0 Å of any heavy atom of that residue. Heavy atoms only,
  computed locally on rdkit + scipy, no dependencies beyond the existing env.
  Transparent and auditable; this is the geometric proxy.
- **FP-B — chemical.** A true interaction fingerprint from **ProLIF**: bits over
  (protein residue × interaction type) for hydrogen bonds (donor and acceptor),
  hydrophobic contacts, π-stacking (face-to-face and edge-to-face), cation-π, and
  salt bridges — with real distance *and angle* criteria, not a distance cutoff.

Recovery = **Tanimoto(FP_gen, FP_ref)** against the reference ligand's
fingerprint in the same receptor, computed separately for FP-A and FP-B.

Requiring both to move is an **intersection** rule, so it is strictly
conservative — it cannot inflate the false-positive rate the way reporting the
better of two endpoints would. The four-way reading is fixed in advance:

| FP-A | FP-B | Reading |
|---|---|---|
| ↑ | ↑ | Interface fidelity. The positive. |
| ↑ | flat/↓ | **Chemical collapse** — spatial proximity bought at chemical cost. A negative for fidelity, and a substantive finding about what the critic does. |
| flat | ↑ | Chemistry retyped without spatial rearrangement. Interesting, but ambiguous pending replication; not claimed on one seed. |
| flat | flat | Negative #4, with mechanism. |

### FP-B is sparse, and sparsity is measured before it is trusted

Reference fingerprints here are small. Measured on the 44 reference ligands:
FP-A carries a **median of 11 bits** (p10 8, min 6), and residues in polar
contact — a generous *upper bound* on what ProLIF will type as a real
interaction, since proximity is necessary but not sufficient for an H-bond —
run **median 8, min 4**, with 11/43 pockets at ≤6. FP-B will be sparser still.

At 4 reference bits a single missed interaction moves Tanimoto by 0.25. Left
unaddressed this holds the whole analysis hostage: an intersection rule
requiring FP-B to clear a noise floor it is too volatile to clear would
*guarantee* a null, and the null would be an artefact of the measuring stick.

**Two statistics, because Tanimoto is the wrong one here.** Tanimoto penalises
*extra* interactions as heavily as missing ones, so at 3–8 reference bits it
conflates "found the reference anchors" with "made additional contacts." FP-B's
decision statistic is therefore **interaction recall**,
`|FP_gen ∩ FP_ref| / |FP_ref|`, which isolates the anchors. Tanimoto and
precision are reported alongside, and so is the **mean interaction count per
arm** — because recall is trivially bought by promiscuity, and a recall gain
accompanied by an interaction-count rise outside the seed-noise band is the FP-B
analogue of chemical collapse, read the same way.

**Viability pre-screen, arm-blind and pre-registered.** Per pocket define
`Range = Ceiling_FP-B − Null_FP-B` from gate 3 and gate 1. A pocket whose range
does not exceed the per-pocket seed-noise scale cannot express a detectable
effect, and is excluded from FP-B's denominator.

Two things make this legitimate, and both must hold:

- **The filter never sees the arms.** Ceiling comes from true binders, and the
  placement null is pooled across arms so no arm's inclusion depends on itself. A
  filter computed from arm-independent quantities cannot select for an effect.
  *This property is the whole justification — if the null is ever computed from
  one arm, the filter becomes a selection bias and the screen is void.*

  **Amendment, 2026-08-16, before the gates were run:** the null pools the **six
  λ = 0.7 arms only** (3 control + 3 critic), not all seven. Two reasons, both
  stricter than the original rule: it is balanced between arm *types*, and it
  excludes `critic_lambda20_r0` — the treatment arm of the primary comparison —
  entirely, so the screen cannot see the arm whose effect it gates. Sizes are
  matched across all seven (20.28–20.58 heavy atoms), so the pooled null remains
  representative of λ = 20's molecules. Recorded here rather than in the results,
  because a rule changed after seeing numbers is not a rule.
- **The comparison is against a per-pocket scale, not the arm-level one.**
  `s_arm(cell)` — the noise floor in the decision table — is the spread of a mean
  over 44 pockets and is smaller by roughly √44. Screening a per-pocket range
  against it would pass essentially every pocket and do nothing. The screen uses
  `s_pocket(cell)`, the spread of within-arm seed-only deltas **at pocket level**,
  pooled over pockets and all 6 comparisons. Both are reported.

**If more than 50% of pockets fail the screen, FP-B is downgraded to a secondary
observation and FP-A resumes sole primary status.** The cost of that is stated
here rather than discovered later: FP-B exists to detect chemical collapse, so an
analysis running on FP-A alone **cannot exclude it**. Branch P reached that way
is labelled "positive on spatial fidelity, chemical collapse not excluded" — a
strictly weaker claim than the intersection rule was built to deliver, and not
one that would on its own justify GPU time on direction C.

### Two statistics per pocket, both pre-registered

Generative models do not fail or succeed uniformly. A critic that forces 5 of 95
molecules into near-perfect reference alignment while leaving the rest untouched
is, for drug design, a better model than one that shifts all 95 by a hair — and
the mean would wash the first out entirely. So per pocket, per fingerprint:

- **mean recovery** over the molecules, and
- **top-3 mean recovery** (the best three).

**The top-K statistic needs a bias correction that the mean does not.** Molecule
counts differ across arms per pocket (90 vs 95 in the first pocket alone), and
order statistics are biased upward by larger *n* — top-3-of-95 beats top-3-of-90
for identical distributions. So for every pocket all arms are **subsampled to
n_min(pocket)**, the smallest count among the arms being compared, using a single
rng drawn **once and shared across arms**. This is the decoy-confound lesson
(`specificity/ANALYSIS_PLAN.md` addendum) applied before rather than after: 0 of
44 pockets shared a decoy set between arms there, injecting 0.52 kcal/mol of
noise against effects of 0.18.

Primary statistic: Δ = value(critic) − value(control), **paired across the 44
pockets**, for each of the 4 cells (2 fingerprints × 2 statistics). Never pooled
per molecule — molecules are nested within pockets.

Reported per cell: mean Δ ± sem, **fraction of pockets improved**, Wilcoxon
signed-rank p, and the full per-pocket recovery *distribution* (deciles), so a
non-uniform shift is visible directly rather than inferred from two summaries.

## Validity gates, run and reported first

If any fails, the finding is about the harness and the comparison is not
interpretable.

1. **Headroom.** A *placement null* — molecules generated for other pockets,
   rigidly translated so their centroid matches the reference ligand's — must
   score materially below the real arms. This is the buriedness lesson: protein
   neighbour count alone reached the 98.2nd percentile on the hotspot protocol
   and would have made a method measuring nothing look excellent. If a
   correctly-sized blob in the right place recovers as much of the fingerprint as
   the real molecules, the metric has no headroom and nothing downstream means
   anything.
2. **Dynamic range.** The reference ligand rigidly displaced by 0.5, 1, 2, 3 Å
   must produce a monotone decay in self-recovery. This calibrates one Tanimoto
   point against Ångströms and shows the metric responds at the scale in
   question. Displacement is applied with `CalcRMS`-style bookkeeping — never
   `GetBestRMS`, which superimposes first and deletes the displacement.
3. **Ceiling — what does a good Tanimoto actually look like?** Gate 2 shows the
   metric responds to displacement; it says nothing about the scale. Tanimoto on
   sparse bit vectors is low even between genuinely similar molecules, so a Δ of
   0.02 is uninterpretable without knowing whether the attainable range is
   [0, 1] or [0, 0.45]. **Two distinct true binders of the same receptor**
   supply that ceiling: CrossDocked entries sharing a receptor stem but carrying
   different ligand codes are different actives posed in the same frame. Compute
   FP-A and FP-B Tanimoto between such pairs, per pocket, and report the ceiling
   alongside every Δ.

   Feasibility measured, not assumed. Across the 5,637 LMDB entries on the 44
   holdout targets, 32/44 targets carry ≥2 distinct ligand codes (median 3, max
   80). Restricting to the *same receptor stem*, so the comparison stays in one
   coordinate frame, leaves **22/44 pockets and 99 candidate pairs** — cached in
   `ceiling_pairs.json`.

   **A same-site filter is mandatory and is not optional bookkeeping.** Two
   ligands of one receptor need not occupy one site: candidate pair centroid
   offsets run 0.43 Å (p10) to **34.81 Å** (max), and 7 of 99 pairs sit at a
   different site entirely. Those would score near-zero Tanimoto and *deflate the
   ceiling*, making any Δ look larger as a fraction of range — the failure would
   have been silent and in the flattering direction. Requiring the partner
   ligand's centroid within 8 Å of the reference's leaves **92 pairs across 20/44
   pockets** (median offset 1.59 Å).

   The ceiling is therefore an estimate on **20/44 pockets**, and at a median of
   1 usable pair per pocket a *per-pocket* ceiling is a single Tanimoto value —
   far too noisy to divide by. So: report the **pooled distribution over the 92
   pairs** (median and IQR) as the scale reference, quote per-pocket ceilings only
   where ≥3 pairs exist, and never silently average over the 24 pockets that have
   none. *Fallback for those:* alternate poses of the same ligand within 2 Å,
   which bounds the ceiling from below rather than estimating it, flagged per
   pocket.
4. **Size matching.** Heavy-atom counts per arm, already checked:
   20.28–20.58 across all seven arms. Contact counts scale with size, so a
   divergence here voids the comparison.
5. **Same pockets, both arms.** 44/44, as paired statistics require.
6. **Protonation succeeded, and failed symmetrically.** FP-B needs explicit
   hydrogens on both partners — the receptor PDBs are heavy-atom ATOM records and
   the generated SDFs carry no H. Receptors are protonated once, at pH 7.4, and
   reused across every arm; ligands are protonated per molecule. Report the
   per-arm failure rate. A *differential* rate between arms is a confound (an arm
   whose molecules protonate less often is being scored on a biased subsample),
   so require the arms to agree within 2 percentage points and the absolute rate
   to stay under 10%, matching the NaN gate used for docking.

   **Amendment, 2026-08-16, written before the FP-B numbers existed.** The
   implementation applies one valence-repair pass before giving up on a molecule
   — over-valent N/O with no formal-charge block is the standard CrossDocked
   artefact (it is what `complex_000148`'s *reference* ligand suffers from), and
   refusing to repair it would reject good molecules. Repairs are counted
   separately from hard failures.

   That creates a loophole in the gate as originally written: if the critic arm
   emits more over-valent nitrogens and every one is repaired, the hard-failure
   rate stays equal across arms and the signal hides in the repair count. So the
   **≤10% absolute and ≤2 percentage-point differential thresholds apply to the
   union — `fail_read_sanitize + fail_addhs + repaired`, every molecule that did
   not sanitize as written** — and the three buckets are also reported
   separately. Chemical degradation that is *repairable* is still chemical
   degradation, and the gate must be able to see it.

   **A failure of this gate in the critic arm is a finding, not an abort.** If
   λ = 20 protonates at 11% failure against the control's 2%, the critic is
   producing molecules that strict cheminformatics rejects — strained valences,
   clashes, geometries RDKit or OpenBabel will not process — and it is buying its
   lower ATOMICA distance by breaking chemistry the diffusion loss was holding
   together. Report it as **branch V: the critic degrades chemical validity**,
   with the per-arm rates and a sample of the rejected molecules. It terminates
   the direction immediately and needs no further comparison.

   Prior evidence says this is unlikely but does not cover it: λ = 20's molecules
   show QED 0.506 vs the control's 0.514 and **zero** multi-fragment structures,
   so gross breakage is absent. Valence and protonation are a different and
   stricter check than QED, which is why the gate stays.
7. **No pocket silently dropped.** `MolFromPDBFile` returned `None` for 1 of the
   44 receptors during the sparsity measurement above. Every stage asserts it
   processed 44/44 pockets and names any it could not; a receptor that fails to
   parse is fixed with a fallback parser, never skipped. This project's
   pose-scorer run silently dropped target `9WT9` to a tokenizer `KeyError`, and
   silent drops are how a biased subsample becomes a result.

## The noise floor is measured, not assumed

The run-to-run null is available directly: `control_r{0,1,2}` and
`graph_cosine_r{0,1,2}` differ from each other **only by seed**. The 6 within-arm
pairwise per-pocket deltas give the distribution that a real effect must exceed.
No effect is claimed that does not clear the largest of them.

**Two scales, both reported.** `s_arm(cell)` is the spread of the *arm-level*
statistic (the mean over 44 pockets) and sets the bar in the decision table.
`s_pocket(cell)` is the spread of the same deltas **at pocket level**, pooled
over pockets and all 6 comparisons, and is what the FP-B viability screen uses.
They differ by roughly √44; using one where the other belongs either passes every
pocket or fails every arm.

**Computed separately for each of the 4 cells.** Order statistics are noisier
than means, so the top-3 endpoint gets its own floor rather than borrowing the
mean's — otherwise the noisier statistic is judged against the quieter one's bar,
which is how a null becomes a finding.

This is the check the specificity seed replicates could not perform, because they
shared an rng stream and therefore replicated the decoy confound rather than the
effect.

## Decision rules, fixed in advance

**Primary comparison: `critic_lambda20_r0` vs `control_r0`** — the only arm whose
objective demonstrably moved. n = 44 pockets. Decision statistics: **FP-A
Tanimoto** and **FP-B recall**. "Clears" = Δ > 0 **and** |Δ| > `s_arm(cell)`
**and** ≥ 30/44 pockets improved **and** Wilcoxon p < 0.05.

Gates run first and can terminate before any of this: **branch V** (gate 6,
critic degrades chemical validity) ends it outright, and gate 1 or the FP-B
viability screen can void or downgrade an endpoint.

| Outcome | Criterion | Reading | What I do |
|---|---|---|---|
| **P. Positive** | **FP-A and FP-B both clear**, on the mean **or** both on top-3 | The critic buys interface fidelity; specificity was the wrong readout | Write up as **suggestive, pending replication**. Launch 2 further λ = 20 seeds (~3.3 h each) before any claim. Only then consider C, with the target this analysis identifies |
| **C. Chemical collapse** | FP-A clears, FP-B flat or negative — **or** FP-B recall clears while interaction count rises beyond `s_arm` | The critic buys proximity, or promiscuity, at chemical cost | A negative for fidelity **and** a mechanism worth reporting. Do not run C |
| **N. Negative** | neither clears | Reducing the ATOMICA distance does not improve interface fidelity even within-system | Write up as **negative #4 with mechanism**; do **not** run C in its current form; recommend writing the project up as four controlled negatives |
| **A. Ambiguous** | anything else — including FP-B alone clearing, or mean and top-3 disagreeing | Underpowered or inconsistent | Report with bounds. Do **not** spend GPU on the strength of it |

**If FP-B was downgraded by the viability screen**, branches P and C collapse
into one another — FP-A alone cannot distinguish fidelity from chemical
collapse. The only outcomes available are then "positive on spatial fidelity,
chemical collapse not excluded" and N, and the former does not justify GPU time
on direction C by itself.

Δ is reported **as a fraction of the pooled true-binder ceiling from gate 3**
(median over the 92 same-site pairs), not in raw Tanimoto units, so that its size
is interpretable against the range the metric can actually express. The ceiling
rests on 20 of 44 pockets and is a *context* figure, not part of any test
statistic — the paired comparison stands on all 44 regardless, and no branch of
the decision table is conditioned on it.

**Consistency check, not a test.** The λ = 0.7 arms (3 seeds vs 3 seeds) are
expected to be null — the paired critic-distance measurement already shows those
models are near-indistinguishable, sign-flipping across seeds. If λ = 0.7 shows a
*larger* effect than λ = 20, the metric is reading noise and this entire analysis
is void regardless of what the primary comparison says.

**One seed at λ = 20.** Branch P is written up as "suggestive, pending
replication" and nothing stronger, for the same reason branch A was in the
specificity plan.

## What this cannot establish either way

- Fidelity to one reference ligand per pocket. A generated molecule that binds
  the same pocket by a genuinely different and better mode scores as a failure.
  This metric rewards imitation and that limitation is intrinsic, not fixable by
  more samples.
- 44 pockets, one held-out set. The pose-scorer retraction happened at this
  sample size.
- FP-B captures interaction *geometry* — distances and angles for H-bonds, salt
  bridges, π-stacking — but no desolvation and no energetics. It is a
  better-typed proxy, not a physics model.
- FP-B's H-bond and salt-bridge terms depend on the protonation assigned at pH
  7.4 by a titration predictor. Histidine tautomers in particular are a known
  weak point. Gate 6 checks the failure rate is low and symmetric across arms;
  it cannot check that the assignments are *right*.

## Environment: isolation, and why not Docker

The earlier draft avoided ProLIF/PLIP because installing them risks pulling
MDAnalysis into `atomica-interface` and letting pip swap the CUDA torch build for
a CPU wheel. That reason does not survive scrutiny: this analysis reads molecules
off disk, runs on CPU, and never imports torch. Avoiding a better metric to
protect an environment it need not touch is infrastructure fear, not a
methodological constraint.

The fix is a **separate conda prefix**, `~/.conda/envs/ifp`, built from
conda-forge with python 3.10, prolif, mdanalysis, rdkit, openbabel, plip and
pdb2pqr. Nothing is installed into `atomica-interface`; the two share only the
filesystem, and inputs are read-only there.

Docker would also work but buys nothing here. The failure mode being defended
against is "pip resolves torch inside the env you are standing in" — a distinct
conda prefix is already complete isolation for that, since python package
resolution never crosses prefixes. A container would add image builds and volume
mounts to defend against a risk that a second prefix already eliminates. If a
future step needs OS-level libraries or a pinned CUDA userspace, revisit.

**PLIP is kept as an auditor, not the primary.** ProLIF produces a per-pose
bit vector directly and vectorises over many poses against one receptor, which is
the shape of this workload (~29,400 pose-receptor pairs across seven arms). PLIP
is re-run on a random 200-molecule subsample and its interaction calls compared
with ProLIF's; a systematic disagreement means FP-B is measuring the
implementation rather than the chemistry, and it is reported as a gate.
- Nothing here is an affinity claim, and nothing here measures pocket
  specificity — that is `results/specificity/`, and it stays the primary
  endpoint for any claim about design.
