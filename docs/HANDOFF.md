# Handoff prompt

Paste the block below into a fresh session. It assumes no memory of prior work
and points at the documents that carry the detail.

---

Continue work in `/home/emilio/Documents/atomica-diff-antibiotic/ATOMICA-Diffusion-Antibiotic-design`.

**Read `docs/experiment-plan.md` first** — it is a handoff document with full
state, numbers, and a gotchas section. Then `results/specificity/README.md` and
`results/critic_arms/README.md`, which hold the current result.

## Where the project is

We are testing ATOMICA as a **training-time critic** for a DiffSBDD diffusion
model:

```
L = L_diffusion + λ(t) · d( ATOMICA(pocket, x̂₀), ATOMICA(pocket, x_true) )
```

ATOMICA is frozen and appears only in the loss, so sampling is plain DiffSBDD.
Three earlier directions (cross-system pose scoring, hotspot fields,
training-free energy) are resolved negative; the plan explains why this one only
ever asks ATOMICA to compare two states of the *same* complex, which is the
regime that measured well.

**It is implemented, trained and evaluated.** The result so far:

- The critic optimises its own objective: `critic_distance/val` −37.9% over
  3,000 steps, monotone.
- It does **not** improve pocket specificity: −0.158 ± 0.111 kcal/mol against a
  matched control, 18/44 pockets, Wilcoxon p = 0.299.
- The decomposition is the interesting part: the critic arm docks *better* both
  into its own pocket and into other targets' pockets, and the gain against
  other pockets is larger — so specificity falls. Reporting mean docking score
  would have made it look mildly good.
- Ligand-only guardrails (QED, SA, MW, diversity) are indistinguishable.

## What is running / what to do next

Four seed replicates were launched because one seed per arm cannot resolve a
−0.158 difference. Check with `ls my_logs/` and
`python scripts/watch_training.py`.

1. **Confirm r2 finished** (`critic_graph_cosine_r2`, `critic_control_r2`, 3,000
   steps each). Configs: `DiffSBDD/configs/crossdock_fullatom_critic{,_control}_r{1,2}.yml`.
2. **Generate for r1 and r2**, both arms, exactly as r0:
   ```
   python scripts/run_baseline.py \
       --checkpoint my_logs/<run>/checkpoints/last.ckpt \
       --test_dir data/processed_expert_atomica/test \
       --outdir results/gen_<run> \
       --pocket_list data/receptor_pdbs_test_v2/pocket_targets.json \
       --n_samples 100 --batch_size 20 --timesteps 100 --no_atomica
   ```
   ~1.8 h per arm.
3. **Cross-dock**, all arms in one run if convenient:
   ```
   export SMINA_BIN=~/.conda/envs/smina/bin/smina
   python scripts/cross_dock_specificity.py \
       --arms critic_r0=results/gen_critic_graph_cosine_r0 control_r0=... \
       --pdb_dir data/receptor_pdbs_test_v2 \
       --out results/specificity/specificity_seeds.csv \
       --n_decoy_pockets 3 --max_mols_per_pocket 20 --exhaustiveness 8 --n_jobs 16
   ```
   **Keep every parameter identical to r0** — an inconsistency confounds the
   comparison the seeds exist to make.
4. **Analyse across seeds.** Per pocket, paired, then per seed. Report the
   fraction of pockets improved and the between-seed spread, not just the mean.
   `results/specificity/ANALYSIS_PLAN.md` has the pre-registered decision rules;
   honour them rather than re-deciding after seeing the numbers.

**Preliminary and important:** the +0.6% diffusion-loss penalty reported at r0
does not reproduce at r1 (critic 0.46177 vs control 0.46186 — sign flipped). The
seeds are already showing that single-seed differences at this scale are noise.
Expect the specificity lean may do the same.

## Environment

`conda activate ~/.conda/envs/atomica-interface` — it now runs both ATOMICA and
DiffSBDD training (torch 2.0.1 / CUDA 11.8, pytorch-lightning 2.3.3).
**Do not `pip install` anything that depends on torch without a constraints file
pinning `torch==2.0.1`** — it silently replaces the CUDA build. smina is in its
own env at `~/.conda/envs/smina/bin/smina`.

Single 8 GB GPU. The critic arm peaks at ~7.6 GB with `batch_size 2` and
gradient checkpointing; batch 4 OOMs inside the critic's encoder pass. Run
things sequentially.

## Constraints

- Commits: **Emilio as sole author, no `Co-Authored-By` trailer.**
- `WANDB_MODE=offline` (no credentials on this machine; `train.py` respects the
  env var).
- Don't delete anything under `data/` without checking — the cleanup already
  removed the superseded sets, and what remains is in use.

## Traps that have already cost time this session

Nearly every bug here was **silent** — code that ran and produced plausible
numbers. Assume the next one is too.

- **A control has to be shown to be a floor, not assumed to be one.** The
  critic gate first used `pocket_pool` as its negative control on the stated
  assumption that pocket blocks do not vary across poses of a target. They do
  (message passing from the ligand), and it *outscored* the real metric at 0.949
  vs 0.926. A permuted-weight control settled it.
- **`conda run` buffers stdout**, so a driver log stays empty for hours. Watch
  `metrics.csv` via `scripts/watch_training.py` instead.
- **`pgrep -f "<pattern>"` matches the launching shell** when the pattern
  appears in a heredoc in that shell's command line. This stalled a chained job
  for 35 minutes. Match exactly, or check for the artefact rather than the
  process.
- **Gradient calibration depends on what is trainable.** λ measured against the
  ATOMICA adapter gave 53.6; against LoRA on the EGNN it is 0.688. Recompute
  after any change to the trainable set (`scripts/calibrate_critic_weight.py`).
- **The train split has 83,921 complexes — an odd number** — so the last batch
  of every epoch is a singleton at batch size 2. That crashed a 3,000-step run
  at the epoch boundary; fixed, with a regression test.
- **Check array axes cover the data.** The size histogram clamped 72% of
  complexes into one bin for a year without erroring.
- `pocket_atomica_embeddings` in the cache derives from a two-segment encoding
  containing the **reference ligand**. It is a valid critic target and an
  invalid sampling-time input; `run_baseline.py` has a guard that refuses it.
