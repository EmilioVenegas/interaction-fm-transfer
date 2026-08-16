# interaction-fm-transfer

This repository contains the code, data, and evidence behind the study:
**"Within-system discrimination does not imply cross-system transfer: interaction foundation models in structure-based generative design."**

*Origin note:* This repository succeeds and replaces the class project on antibiotic design originally hosted at [ATOMICA-Diffusion-Antibiotic-design](https://github.com/EmilioVenegas/ATOMICA-Diffusion-Antibiotic-design). The framing and repository name have been updated to reflect the rigorous controlled study it became.

## Overview

Pretrained molecular-interaction foundation models are increasingly adapted for structure-based generative design. In this work, we demonstrate a sharp regime boundary in the transferability of these models: they discriminate interaction geometry perfectly *within* a single complex, but fail to transfer *between* complexes. We also identify a mechanistic consequence of this failure: optimising the model's interface distance as a critic trades against pocket fit.

We evaluate this using established metrics like PoseCheck (interaction recovery) and Delta Score (cross-docking specificity), finding that they cannot resolve the effects they are often used to claim.

## Claim-to-Evidence Map

Every quantitative claim in the preprint is backed by pre-registered analysis plans and raw data in this repository. 
Analysis plans were committed *before* results were generated (see `results/specificity/ANALYSIS_PLAN.md` and `results/interface_fidelity/ANALYSIS_PLAN.md`).

| Claim | Evidence Directory | Script / Analysis | Raw Data |
|---|---|---|---|
| **R1. Within-system discrimination is near-perfect** (AUROC 1.000 / 0.926) | `results/phase0/`, `results/critic_gate/` | `scripts/hotspot_validate.py` | `results/critic_gate/gate_cache.json` |
| **R2. Cross-system transfer fails** (63.9% vs smina 59.7%) | `results/pose_scorer/` | `scripts/train_pose_scorer.py` | `results/pose_scorer/pose_scorer_report.json` |
| **R3. Interaction hotspot fields carry no signal** (52.4 percentile) | `results/hotspot/` | `scripts/hotspot_validate.py` | `results/hotspot/hotspot_1h1s_4SP.json` |
| **R4. Pocket-only encodings are degenerate** (cosine 1.0000) | `results/featurization_probe/` | `scripts/featurization_probe.py` | `results/featurization_probe/featurization_probe.json` |
| **R5. The critic objective moves, but molecules dock worse** (0.232 kcal/mol worse) | `results/critic_arms/`, `results/specificity/` | `scripts/cross_dock_specificity.py` | `results/specificity/specificity_lambda20_paired.csv` |
| **R6. Interface fidelity and dose-response** | `results/interface_fidelity/` | `scripts/interface_fidelity.py` | `results/interface_fidelity/primary_fpa_tanimoto.csv`, `primary_result.json` |
| **R7. Detection limits (94% variance from noise)** | `results/specificity/` | `scripts/analyse_specificity_seeds.py` | `results/specificity/specificity_seeds.csv` |

*(See `docs/PREPRINT_PROMPT.md` and `docs/experiment-plan.md` for full technical constraints and detailed quantitative findings).*

## Environment Setup

The pipeline requires isolated conda environments to avoid silent build replacements:

1. **ATOMICA & DiffSBDD (Training & Generation)**
   ```bash
   conda env create -f environment-atomica.yml
   conda activate atomica-interface
   ```
   *Note:* Uses `torch==2.0.1` and CUDA 11.8. **Warning:** Do not run `pip install` on anything depending on `torch` without pinning it, as it will silently replace the CUDA build with a CPU wheel.

2. **Docking (smina)**
   ```bash
   conda create -n smina -c conda-forge smina=2020.12.10
   ```
   Point `SMINA_BIN` to `~/.conda/envs/smina/bin/smina`.

3. **Interaction Fingerprints (ProLIF/MDAnalysis/PLIP)**
   Requires a separate environment (`ifp`) due to dependency constraints with the main environment.
