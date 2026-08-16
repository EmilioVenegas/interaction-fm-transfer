# Research brief: find the next project

Written 2026-08-16, at the close of the ATOMICA direction. Hand this to a
research agent, or work it yourself. It is self-contained.

---

## What I need from you

**A ranked shortlist of 5–8 concrete research projects** in protein design,
drug design, or structural machine learning that (a) run on the hardware below,
(b) have a plausible path to a *positive* result, and (c) are not already
crowded. For each one, answer the template at the end. Do not return a survey of
a field; return projects I could start on Monday.

Bias hard toward **projects that can produce a positive result with a real
baseline**, not projects that are merely interesting.

---

## Hard constraints

**Compute.** One **8 GB** consumer GPU. 32 CPU cores. No cluster, no cloud
budget. For calibration: fine-tuning a ~1M-parameter SE(3)-equivariant diffusion
model on full-atom protein pockets peaked at ~7.6 GB at batch size 2 with
gradient checkpointing; batch 4 went OOM. So:

- Training a protein language model from scratch: **out**.
- Full fine-tuning of ESM-2 650M or larger, AlphaFold/Boltz training: **out**.
- LoRA/adapter fine-tuning of a mid-size pretrained model, inference with a
  large frozen model, small-model training from scratch, or classical
  ML on learned features: **in**.
- Anything whose headline experiment needs more than ~2 GPU-weeks: **out**.

**Time.** 2–4 months to a preprint.

**Person.** Comfortable with PyTorch, diffusion models, equivariant
architectures, structural bioinformatics, docking (smina/Vina/Uni-Dock), RDKit,
and rigorous experimental design. Working alone. **No wet lab** — anything
requiring experimental validation to be publishable is out, though projects where
validation is a bonus rather than a requirement are fine.

**Data.** Must be publicly downloadable and fit on a workstation disk. Assume
PDB, CrossDocked2020, ChEMBL, BindingDB, UniProt, PDBbind, and the standard
benchmark sets are available. Flag anything needing licensed or registration-
gated data (e.g. CASF-2016).

---

## Assets already in hand — reuse is a strong plus

- A working **DiffSBDD** installation (SE(3)-equivariant diffusion for
  pocket-conditioned ligand generation), fine-tunable, with LoRA implemented.
- Preprocessed **CrossDocked2020**: 164,814 complexes, 83,921 training complexes,
  and a properly target-disjoint held-out split (44 test targets, zero shared
  with train).
- A **docking and evaluation harness** on 32 cores: smina, cross-docking
  specificity with matched decoys, ProLIF interaction fingerprints in an isolated
  environment, QED/SA/Lipinski/diversity guardrails, and — unusually — measured
  detection limits for each metric.
- Demonstrated practice in **pre-registered analysis plans, matched controls, and
  power analysis**. This is the real asset. Design projects that exploit it.

---

## What just failed, and the lesson to carry

The previous direction used a pretrained molecular-interaction foundation model
(ATOMICA) to improve pocket-conditioned generation. It produced four controlled
negatives. The governing measurement:

| task | result |
|---|---|
| rank poses **within one complex** | AUROC 1.000 (probe fit per system) |
| rank poses **across unseen targets** | 63.9% docking power vs smina's 59.7%, p = 0.65 |

The representation discriminates sharply *inside* a system and does not transfer
*between* systems. Everything built on cross-system transfer failed; the one
regime that measured at ceiling was fitted per-target.

**Lessons that should shape what you propose:**

1. **Prefer within-system or per-target formulations** over ones needing a
   transferable absolute score across proteins.
2. **Every claim needs a control that could have detected the positive.** A
   buriedness baseline reached the 98.2nd percentile where the method under test
   scored 52.4 against a 52.2 floor. Propose the control alongside the claim.
3. **Check the detection limit before running.** In the previous project 94% of
   the variance in the headline effect was harness sampling noise, and every
   effect ever discussed was below the minimum detectable effect. Any project you
   propose must state what effect size it could resolve.
4. **Beware crowded evaluation niches.** PoseCheck, PoseBusters, GenBench3D,
   CBGBench and Delta Score already occupy "benchmark the generative models."
   A new benchmark is a hard sell.
5. **Beat a real baseline or don't bother.** smina is free, fast and from 2010,
   and it was not beaten. ECFP4 + random forest is the analogous baseline on the
   ligand side. Name the baseline that would kill each proposal.

---

## Directions worth investigating (not exhaustive — propose better ones)

Treat these as seeds, and say plainly if you think one is a dead end:

- **Per-target scoring / lead optimisation.** Fit a model per target using that
  target's known actives; hold out ligands *within* target. The regime that
  measured at ceiling above. The control that decides it: ligand-only features
  (ECFP4 + RF) — if those match, the structure adds nothing.
- **Small-molecule generation with a genuinely different conditioning signal** —
  something geometric and per-atom rather than a pooled global embedding.
- **Protein design at small scale**: binder design, loop/linker design, or
  peptide design, where 8 GB is enough and evaluation is *in silico* but
  well-established.
- **Antibiotic-relevant targets specifically**, if a niche exists where public
  actives are plentiful and the field is thinner than kinases.
- **Inverse folding / sequence design** on a constrained sub-problem.
- **Anything exploiting the measured-detection-limit practice** as a first-class
  contribution rather than an afterthought.
- **Data-centric work**: a dataset or split that fixes a known flaw others have
  been silently inheriting. (The previous project found a widely used split whose
  val and test index lists were *identical*.)

---

## Answer template — required for each proposal

1. **One-sentence claim.** What would the paper assert?
2. **The positive result.** What specific measurement, with what effect size,
   would constitute success?
3. **The baseline that would kill it.** The strongest cheap alternative that must
   be beaten, and your honest estimate of whether it can be.
4. **Feasibility on 8 GB.** Model size, batch size, training time, peak VRAM.
   Say if it needs quantisation, LoRA, or gradient checkpointing.
5. **Data.** Exact source, size, licence, and whether registration is needed.
6. **Prior art.** The 3–5 closest published works, with links, and one sentence
   on what is left unclaimed. **Search for these — do not answer from memory.**
   If the space is crowded, say so and rank the proposal down.
7. **Detection limit.** Roughly what effect size the proposed design could
   resolve, and whether the expected effect exceeds it.
8. **Time to preprint**, in weeks, assuming one person.
9. **Failure mode.** The most likely way this becomes another controlled
   negative — and whether that negative would still be publishable.

## Ranking

Rank by **P(publishable positive result within 4 months)**, not by novelty or
ambition. State that probability explicitly for each, and justify it. I would
rather have a modest defensible result than an ambitious one that produces a
fifth negative.

Finish with a single recommendation and the first week's concrete work.
