# Prompt: write the preprint

Hand this to the drafting agent. It is self-contained — it assumes no memory of
the project and carries every number the manuscript needs. Where a claim is a
*prior* rather than a *measurement* it is labelled as such, and the manuscript
must preserve that distinction.

---

## Your task

Write a complete preprint as a **Quarto markdown file**,
`paper/atomica_transfer_preprint.qmd`, plus `paper/references.bib`.

**Match the conventions of `/home/emilio/Documents/hairpin/mantis/paper/mantis_preprint.qmd`
exactly.** Read that file first. Specifically reuse: the YAML front matter shape
(title / shorttitle / author with ORCID `0000-0002-7689-9185` and affiliation
Tecnológico de Monterrey, Escuela de Medicina y Ciencias de la Salud, Monterrey
N.L. 64700, México; block `abstract: |`; `keywords`; the bioRxiv metadata block
— `article-category`, `subject-area`, `author-approvals`, `competing-interests`,
`funding`, `distribution-reuse`, `external-data`), the Quarto rendering options
(`bibliography: references.bib`, `number-sections: true`, 11pt, 1in margins,
`linestretch: 1.25`, letter, Times New Roman / Arial / Courier New, NavyBlue
links, `pdf` via `xelatex` with the `lineno`/`unicode-math`/`booktabs`/
`microtype`/`siunitx`/`caption` header block and the custom `\maketitle`, plus
the `html` format), cross-reference idioms (`{#sec-...}`, `{#tbl-...}`,
`{#fig-...}`, `@tbl-`, `@fig-`, `Section [-@sec-...]`), table captions written
as a `: caption {#tbl-id}` line beneath the table, and the unnumbered back
matter (Data/scripts/code availability, Funding, Conflict of interest,
**Artificial intelligence disclosure**, Author contributions, Acknowledgements,
References).

Set `subject-area` to "Bioinformatics" or "Biophysics" and `article-category` to
"New Results". Author: Emilio A. Venegas, sole author, corresponding,
emiliovenegas10@gmail.com.

---

## The thesis — read this before writing a word

**Do not write a "negative results" paper.** The results support three positive
claims, and the manuscript is organised around them:

1. **A sharp regime boundary.** A pretrained molecular-interaction foundation
   model discriminates interaction geometry almost perfectly *within* a single
   complex and fails to transfer *between* complexes. Applications divide cleanly
   along that line, and the boundary predicts which succeed.
2. **A mechanism, with a dose-response.** Optimising the model's interface
   distance as a training-time critic does not weakly help pocket fit — it
   *trades against* it. The objective provably falls while the molecules dock
   worse and reproduce fewer reference contacts.
3. **Detection limits.** Standard evaluations for structure-based generative
   models cannot resolve the effects they are used to claim, and the paper
   quantifies by how much.

Working title, adapt freely but keep the shape:
*"Within-system discrimination does not imply cross-system transfer: interaction
foundation models in structure-based generative design."*

**Framing rule:** every negative is paired with a control proving the measurement
*could* have detected the positive. That pairing is what makes them evidence
rather than absence of evidence, and it must be explicit in the text every time.

---

## Positioning against prior art (essential — do not skip)

These exist and overlap. Cite them as the evaluation framework the paper *uses*,
never as competitors, and never claim the metrics as novel:

- **PoseCheck** (Harris et al., NeurIPS 2023 GenBio workshop / OpenReview
  `xoUUCS9IGl`, https://github.com/cch1999/posecheck) — benchmarks SBDD
  generative models with redocking, strain energy, and ProLIF-derived
  **interaction fingerprints**, reporting an "interaction recovery" metric.
  Our FP-B is the same construct with the same tool. Say so plainly, and use
  their qualitative finding (generated molecules make fewer key interactions
  than reference ligands) as independent corroboration of our null and ceiling.
- **Delta Score** — "Rethinking Specificity in SBDD: Leveraging Delta Score and
  Energy-Guided Diffusion", arXiv:2403.12987. Introduces a binding-specificity
  metric on the premise that generated molecules bind almost every pocket. Our
  cross-docking specificity is the same idea; cite it as precedent.
- **PoseBusters** (Buttenschoen et al., *Chem. Sci.* 2024) — physical/chemical
  validity checks.
- **GenBench3D** (arXiv:2407.04424) and **CBGBench** — benchmark suites.
- **PharmacoNet** (*Chem. Sci.* 2024) — supervised pharmacophore/hotspot
  prediction from protein structure; cite when explaining why we did not pursue a
  trained readout on probe representations.

Also cite: DiffSBDD (Schneuing et al.), CrossDocked2020 (Francoeur et al.),
smina (Koes et al. 2013), AutoDock Vina, ProLIF (Bouysset & Fiorucci),
RDKit, CASF-2016 (Su et al.), Radoux et al. 2016 (hotspot validation protocol),
and the ATOMICA model itself.

**What the paper claims as novel:** claims 1 and 2 above, and the *detection-limit
analysis applied to those established metrics* (claim 3, as a methods section,
not as a new benchmark).

---

## Structure

Follow `# Introduction {#sec-intro}` / `# Methods` / `# Results` /
`# Discussion` numbering as in the reference file.

**Introduction.** Foundation models for molecular interactions are widely assumed
to transfer to downstream design. State the regime boundary as the paper's
finding. End with a short roadmap referencing sections by `[-@sec-...]`.

**Methods.** ATOMICA representations (two-segment interface encoding, PS_300
fragmentation, graph vs unit representations); the pose benchmark; the diffusion
generator and the critic objective; the evaluation metrics and — importantly —
the *controls*: permuted-weight, buriedness, placement null, true-binder ceiling,
matched decoys, seed-only noise floors. Include the pre-registration practice as
a methods statement: analysis plans were committed before results existed
(`results/specificity/ANALYSIS_PLAN.md`, `results/interface_fidelity/ANALYSIS_PLAN.md`).

**Results**, one subsection per block below.

**Discussion.** The boundary; why within-system success does not imply
transferability; the mechanism hypothesis; what this implies for using
interaction foundation models in generative design; limitations; and the
recommendation that evaluations report their detection limit.

---

## The evidence, with every number

### R1 — Within-system discrimination is near-perfect

Clash- and composition-controlled pose benchmark, 30 poses per class:

| scorer | AUROC | Spearman vs RMSD | needs fitting |
|---|---|---|---|
| min contact distance (trivial baseline) | 0.727 | — | no |
| training-free denoising energy | 0.787 | +0.476 | no |
| linear probe on the representation | **1.000** | +0.927 | yes, per system |

The 1.000 is an *upper bound on extractable signal*, fitted on the system it
scores — say this explicitly, it is not a competing number. The rotation head is
anti-correlated: displaced poses score *lower* rotational correction than native
ones.

Independent confirmation on a larger set (the critic gate): 92 targets, 1,662
poses; every pose of a target is the same molecule rigidly displaced, so
composition is controlled by construction.

| metric | ρ(all) | ρ(<4 Å) | AUROC |
|---|---|---|---|
| `graph_cosine` (pretrained) | +0.386 | +0.558 | **0.926** |
| `graph_cosine` (permuted weights) | +0.149 | +0.061 | 0.697 |
| contacts (no-learning floor) | +0.253 | +0.355 | 0.837 |
| smina (reference) | +0.281 | +0.465 | 0.844 |

Pretraining carries it: in the low-RMSD regime the permuted-weight control has
essentially no signal.

**Gotcha worth a sentence in Methods:** `pocket_pool` scored the best raw AUROC of
any variant (0.949) but scores 0.923 with *random* weights — almost all
architecture and geometry, not learned chemistry. It was nearly used as the
negative control. This is why every representation claim needs a
same-architecture, same-scale, no-learned-information comparison.

### R2 — Cross-system transfer fails

100-target CASF-style benchmark built from open RCSB data, 72 solvable targets,
1,674 poses, out-of-fold under `GroupKFold` **by target**, ridge α by inner CV.
Metric: docking power (is the top-ranked pose within 2 Å of crystal).

| scorer | docking power | hits | mean per-target Spearman | vs smina |
|---|---|---|---|---|
| random floor | 15.7% | — | — | — |
| **smina** | **59.7%** | 43/72 | — | — |
| graph (32-d) | 55.6% | 40/72 | +0.371 ± 0.084 | p = 0.70 |
| pocket_pool (96-d) | 41.7% | 30/72 | +0.344 ± 0.077 | p = 0.019, worse |
| all-block (288-d) | 63.9% | 46/72 | +0.400 ± 0.074 | p = 0.65 |

Every variant sits far above the 15.7% floor on unseen targets, so the head
learns something real and transferable — it is simply not competitive with a
fast, free, established baseline.

**Report the retraction in the main text.** An earlier 22-target result had
`pocket_pool` at 68.2% with a paired Spearman gain of +0.127 (p ≈ 0.07); at 72
targets it measures 41.7% and −0.027 (p = 0.52). The effect *reversed*, not
merely shrank. Causes: six feature sets evaluated and the best reported, and 22
targets could not resolve two- or three-target differences (every McNemar
p ≥ 0.69). Recording a selection bias does not remove it. This belongs in the
paper — it is the strongest possible argument for the design discipline used
later.

### R3 — Interaction hotspot fields carry no signal

Chemical probes on a grid, scored with the training-free denoising energy;
validation protocol from Radoux et al. 2016. CDK2 / NU6102 (1H1S chain A), 5 Å
site, 1.5 Å grid, 1,683 accessible non-clashing points, 6 probes × 2
orientations, 28 ligand atoms.

| measure | value | reference |
|---|---|---|
| median percentile, matched probe | 52.4 | Radoux: 97 (fragments), 72 (leads) |
| median percentile, **buriedness control** | **98.2** | the confound |
| median percentile, random placement | 52.2 | the floor |
| type specificity | 0.107 | chance = 0.167 |

The buriedness control is the point: protein neighbour count alone beats the
field and would beat a published fragment number. Without it, "ligand atoms land
in the 98th percentile" would have read as success.

### R4 — Pocket-only encodings are degenerate

99 pockets, identical setup, differing **only** in block vocabulary.

| | old `[GLB, UNK]` | per-residue |
|---|---|---|
| blocks per pocket | 2.0 | 56.3 (17.9 distinct types) |
| **mean pairwise cosine between different pockets (graph)** | **1.0000** | **0.9917** |
| mean pairwise cosine (unit) | 1.0000 | 0.9999 |
| composition probe R² (graph) | 0.201 | 0.176 |

Cosine 1.0000 means every pocket mapped to the same direction: conditioning on
pocket identity was not attenuated, it was *absent*. Fixing the vocabulary barely
moves it; only supplying the second segment (the ligand) does, at 0.9248 over six
unrelated targets. **The missing interaction partner is the binding defect, not
the block vocabulary.**

Consequence for the earlier ablation (100 pockets, 9,328 vs 9,246 valid
molecules): QED 0.424 → 0.483 (+13.9% relative), diversity 0.731 → 0.684 (−6.4%),
no target-aware gain. A generic drug-likeness prior, exactly as the degenerate
encoding predicts — and a worked example of a target-independent metric being
mistaken for evidence of pocket fit.

### R5 — The critic: the objective moves, the molecules do not

Setup: `L = L_diffusion + λ(t)·d(ATOMICA(pocket, x̂₀), ATOMICA(pocket, x_true))`,
ATOMICA frozen, LoRA rank 8 on the EGNN (82,120 of 1,087,538 parameters, 7.55%),
`graph_cosine` distance, λ ramped off at high noise, 3,000 steps, 83,921 training
complexes, 44 held-out pockets over 44 targets with **zero targets shared with
train**.

**λ calibrated by gradient norm, not loss ratio** — and it depends on what is
trainable: λ for a 10% gradient share is 53.6 through the input-side adapter but
0.688 through LoRA on the EGNN. Carrying the first figure over would have made
the critic ~240× too strong. Worth a Methods paragraph.

**At λ = 0.7 (~2.5% of the total training gradient) nothing happens, including to
its own objective.** Paired replay of the validation set through both arms from
one shared seed — identical complexes, timesteps and noise, 475 paired samples:

| seed | critic | control | Δ | critic lower at | p |
|---|---|---|---|---|---|
| r0 | 0.002972 | 0.003168 | −0.000197 | 261/475 | 0.006 |
| r1 | 0.003109 | 0.002976 | +0.000133 | 223/475 | 0.030 |
| r2 | 0.003067 | 0.003112 | −0.000045 | 243/475 | 0.849 |

**The sign flips across seeds**, two of them nominally significant in opposite
directions. The previously reported −37.9% fall in the logged metric replicates
at neither other seed (−7.5%, −3.6%); the diffusion-loss penalty likewise
(+0.00207, −0.00009, +0.00111). Both headline numbers were seed artefacts. Note
that the control arm never logs `critic_distance` at all, so the reported fall
had never been compared against anything — the paired replay is what settled it.

**At λ = 20 the critic acts, and every downstream readout is worse or flat:**

| readout | λ = 20 vs control |
|---|---|
| its own ATOMICA objective | **improves**, 2.8 control-seed sd, p = 7e-7, paired |
| diffusion loss | +0.0078 (~7 sd of the seed spread) |
| docking into its own pocket | **0.232 kcal/mol worse**, p = 0.001 |
| pocket specificity | −0.055 ± 0.047, 17/44 pockets, p = 0.21 (MDE 0.131) |
| interface fidelity (below) | no gain; contact recovery slightly worse |

### R6 — Interface fidelity, and the dose-response

Two co-primary fingerprints against the reference ligand, per pocket, paired over
44 pockets. **FP-A** = residue-contact Tanimoto (4.0 Å, heavy atoms). **FP-B** =
ProLIF interaction fingerprint, decision statistic = recall.

Scale, measured before the comparison:

| | FP-A Tanimoto | FP-A recall | FP-B Tanimoto | FP-B recall |
|---|---|---|---|---|
| placement null | 0.5107 | 0.6261 | 0.3115 | 0.4696 |
| real arms, pooled | 0.6277 | 0.7786 | 0.3764 | 0.5769 |
| true-binder ceiling | 0.7596 | 0.8333 | 0.5251 | 0.7500 |
| usable range | 0.2489 | 0.2073 | 0.2136 | 0.2804 |

Rigid-displacement calibration (FP-A Tanimoto): 0.8733 / 0.7610 / 0.5542 / 0.4499
at 0.5 / 1 / 2 / 3 Å. Interpolating: the placement null ≈ 2.4 Å of displacement,
generated molecules ≈ 1.64 Å, a genuine second binder ≈ 1.01 Å. **The entire span
under study is ~0.6 Å of equivalent rigid displacement.**

Primary comparison, `critic_lambda20_r0` vs `control_r0`:

| statistic | Δ | % of range | improved | p | s_arm |
|---|---|---|---|---|---|
| **FP-A Tanimoto, mean** | **−0.00800 ± 0.00214** | −3.22% | **12/44** | **0.0002** | 0.00756 |
| FP-A Tanimoto, top-3 | −0.00738 ± 0.00406 | −2.97% | 17/44 | 0.134 | 0.01011 |
| FP-B recall, mean | +0.00237 ± 0.00428 | +0.84% | 20/44 | 0.690 | 0.00538 |
| FP-B recall, top-3 | +0.00258 ± 0.00415 | +0.92% | 10/44 | 0.639 | 0.00827 |

**The dose-response is what makes it readable.** At λ = 0.7 — three seeds per arm,
nine cross-arm pairs, a dose that moves the critic's own objective not at all —
the critic-vs-control difference is **−0.00009** on FP-A (largest single |Δ| of
the nine: 0.00609); FP-B recall −0.00352 (largest 0.00798). The pipeline
therefore carries no systematic bias between arms, and the λ = 20 shift clears
seed noise (×1.06), the largest λ = 0.7 excursion (×1.31) and the λ = 0.7 mean
(×89).

**State the two claims separately and do not merge them.** *No fidelity gain* is
established and needs no replication. *Active degradation at λ = 20* rests on one
seed and must be labelled suggestive — this project has already retracted two
seed artefacts.

### R7 — Detection limits (the methods contribution)

- Cross-docking specificity, 44 pockets × (own + 3 decoys from different targets)
  × 20 molecules: per-pocket sd 0.738, sem 0.111, **MDE 0.312** before matched
  decoys; 0.310 / 0.047 / **0.131** after. Variance decomposition: independent
  decoy draws 0.519 sd (49%), the 20-molecule subsample 0.495 (45%) — **94% of the
  reported effect's variance was harness sampling noise**, leaving sd ≈ 0.18 for
  any real difference. Every effect in the literature-relevant range here is
  below that.
- **Seed replicates cannot catch a shared-stream confound.** The replicates were
  given an identical rng stream so only the trained model would differ, which
  handed every replicate the *same* decoy assignment. Agreement across seeds
  replicated the confound, not the effect.
- Interface fidelity resolves far finer: MDE 0.0072 Tanimoto = **2.9% of its
  dynamic range**, against cross-docking specificity's floor at roughly 40% of its
  own signal scale. Two independent routes agree (computed MDE 0.0072; largest
  observed seed-only arm delta 0.0076).
- A correctly-sized molecule placed at the reference centroid already recovers
  **51%** of reference residue contacts and **31%** of typed interactions —
  absolute fingerprint scores are mostly positional.
- **Same-site filtering matters:** of 99 candidate true-binder ceiling pairs, 7
  bound a different site (centroid offsets up to 34.81 Å) and would have
  *deflated* the ceiling, inflating every Δ expressed as a fraction of range.
- Sanitising SDF readers silently discarded ~2.5% of generated molecules on
  chemistry grounds inside geometric measurements; `Chem.MolFromPDBFile` returned
  `None` for 1 of 44 receptors where direct ATOM parsing gives 44/44.

Recommendation for the Discussion: **evaluations of structure-based generative
models should report a measured detection limit and a same-scale null**, not only
a point estimate.

---

## Discipline the manuscript must keep

- Never state an effect without its control, and name the control.
- Never write "no effect" — write "no effect large enough for this design to
  detect", and quote the bound.
- Label anything unreplicated as suggestive. One seed is one seed.
- Report per-pocket / per-target units of analysis; molecules are nested within
  pockets and pooling them is pseudo-replication.
- Report the two retractions (the 22-target pose-scorer result, the −37.9% critic
  fall) in the main text, not a footnote. They are evidence of method, not
  embarrassments.
- Distinguish *label* from *conditioning input*: the reference ligand in a loss is
  supervision; the same ligand at sampling time would be leakage. The paper turns
  on this and it is easy to get wrong.

## Housekeeping

- Data/code availability: repository `ATOMICA-Diffusion-Antibiotic-design`,
  analysis plans committed before results (`results/*/ANALYSIS_PLAN.md`), raw
  per-pair scores in `results/specificity/` and `results/interface_fidelity/`.
- Include an **Artificial intelligence disclosure** section, as the reference
  manuscript does.
- Write `references.bib` in the same style as
  `/home/emilio/Documents/hairpin/mantis/paper/references.bib` (author/title/
  journal/volume/number/pages/year/doi). Verify every DOI you emit; if you cannot
  verify one, omit the field rather than inventing it.
- Do not invent numbers. Every quantity in the manuscript must come from this
  document. If something is needed that is not here, mark it `<!-- TODO: verify -->`
  and leave it for the author.
