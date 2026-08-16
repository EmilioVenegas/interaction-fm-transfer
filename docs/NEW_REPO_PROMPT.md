# Prompt: create and refactor the successor repository

Hand this to an agent with shell access and `gh` authenticated. It is
self-contained.

---

## Context

`~/Documents/atomica-diff-antibiotic/ATOMICA-Diffusion-Antibiotic-design` began
as a class project on *de novo* antibiotic design and became, over 153 commits
from 2025-10-08 to 2026-08-16, a controlled study of whether a pretrained
molecular-interaction foundation model (ATOMICA) transfers to structure-based
generative design. It does not. The project produced four controlled negatives
plus a mechanism, and that is what is being written up
(`docs/PREPRINT_PROMPT.md`).

The repository name and framing no longer match the work. **The existing repo and
its GitHub remote must be left completely untouched** — it is referenced
elsewhere as the original class project. You are creating a *successor*.

## The single most important constraint

**The git history is the scientific evidence and must survive intact.**

The preprint claims that each analysis plan was committed *before* the results it
governs existed (`results/specificity/ANALYSIS_PLAN.md`,
`results/interface_fidelity/ANALYSIS_PLAN.md`). That claim is verifiable only
through commit timestamps. It also reports two retractions, which are claims
about a sequence of events recorded in the log.

Therefore:

- **Do not** create a fresh repository and copy files into it.
- **Do not** run `git filter-repo`, `git rebase`, squash, amend, or anything else
  that rewrites existing commit hashes or dates.
- **Do** push the full existing history to the new remote, then make all
  refactoring changes as *new commits on top*.

Removing a file in a new commit is correct and keeps it recoverable in history.
Rewriting the past is not.

If any instruction below appears to conflict with this constraint, this
constraint wins — stop and report rather than improvising.

---

## Step 1 — safety

1. `cd` to the repo. Confirm `git status` is clean; if not, stop and report.
2. Record the current HEAD sha and `git rev-list --count HEAD` (expect 153).
3. Make a local backup bundle before touching anything:
   `git bundle create ~/atomica-repo-backup-$(date +%F).bundle --all`
4. Confirm the existing remote:
   `origin  https://github.com/EmilioVenegas/ATOMICA-Diffusion-Antibiotic-design.git`
   **Never push to `origin` at any point in this task.**

## Step 2 — create the new remote and push history

Repository name: **`interaction-fm-transfer`** (use this unless the operator has
specified otherwise).

```bash
gh repo create interaction-fm-transfer --private \
  --description "Does a pretrained molecular-interaction foundation model transfer to structure-based generative design? Four controlled negatives and a mechanism."
git remote add newrepo https://github.com/<user>/interaction-fm-transfer.git
git push newrepo --all
git push newrepo --tags
```

Create it **private**. The operator will make it public when the preprint posts.

Verify the push: `git ls-remote newrepo | head` and confirm the commit count on
the new remote matches 153. Report both.

## Step 3 — switch working branch

Do all refactoring on a branch, not directly on `main`:

```bash
git checkout -b refactor/successor-layout
```

Push the branch to `newrepo` only. Open a PR on the new repo at the end so the
operator can review the diff before merging.

---

## Step 4 — the refactor

Goal: a repository whose structure matches the paper, where a reader can move
from a claim to the code and data that produced it. Work in **small, separately
reviewable commits**, each with a message explaining *why* — match the existing
commit style in this repo (read `git log` first: sentence-case subject under ~72
chars, a blank line, then prose paragraphs explaining reasoning and consequence;
no bullet-point-only messages, no `Co-Authored-By` trailers, Emilio Venegas as
sole author).

### 4a. Remove what the pipeline no longer needs

Delete in a normal commit (recoverable from history):

- `rl_loop/` — a post-generation ADMET/REOS/Tanimoto filter from the original
  class-project pipeline. Nothing in the paper's results depends on it. Confirm
  by grepping for imports before deleting.
- `docking/` — superseded by `scripts/cross_dock_specificity.py` and
  `scripts/dock.py`. Verify `docking/scores.csv` is not referenced by any
  surviving script or result README before removing.
- `CLAUDE.md` — tooling instructions for an assistant, not part of the work.
- Stale planning docs that the paper supersedes: `docs/OVERNIGHT_2026-08-15.md`,
  `docs/HANDOFF.md`, and `docs/EXPLORATION_PROMPT.md`. **Check each for content
  not preserved elsewhere before deleting**; anything still true and useful
  should be folded into `docs/experiment-plan.md` rather than lost.
- `results/deleted_runs_manifest.json` if it refers only to runs no longer
  present.

**Do not delete** without checking first, and report anything you chose to keep
and why. `results/` in particular is evidence — keep every `README.md`,
`ANALYSIS_PLAN.md`, and raw CSV under it. When unsure, keep it.

### 4b. Deal with the vendored forks and large binaries

`ATOMICA/` (70 files) and `DiffSBDD/` (64 files) are vendored third-party
codebases with local modifications, documented in `MODIFICATIONS.md` and
`THIRD_PARTY_NOTICES.md`. They carry 68 MB of committed model weights:

```
33 MB  ATOMICA/pretrain/pretrain_model_weights.pt
17 MB  DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt
16 MB  DiffSBDD/checkpoints/last_ckpt.ckpt
```

**Keep the vendored source** — the modifications are what make the work
reproducible, and `MODIFICATIONS.md` documents them. **Propose, but do not
execute without asking**, moving the three checkpoint files out of the repo and
into a release asset or Zenodo archive with a download script. Removing them in a
new commit does not shrink the clone (history retains them), so this is a
question of tidiness, not size — say so plainly in your report rather than
implying a saving that will not materialise.

Also remove obviously unused vendored example data if and only if nothing
references it: `ATOMICA/data/example/example_data/*.cif` (2.7 MB, 1.6 MB,
1.3 MB) and `DiffSBDD/img/overview.png`.

### 4c. Reorganise around the paper

Target layout — adapt if the existing structure already serves better, and
explain any deviation:

```
README.md                  paper-facing map: claim -> files
docs/
  experiment-plan.md       full technical state (keep)
  method.md, results.md    (keep; reconcile with the paper)
  PREPRINT_PROMPT.md       (keep)
  NEXT_PROJECT_RESEARCH_PROMPT.md
paper/                     new; the .qmd and references.bib land here
atomica_interface/         our ATOMICA wrappers (keep as-is)
scripts/                   analysis and pipeline entry points (keep)
results/                   evidence: READMEs, analysis plans, raw CSVs
tests/                     keep
ATOMICA/, DiffSBDD/        vendored, modifications documented
```

Use `git mv` so history follows the files.

### 4d. Rewrite the top-level README

This is the most important deliverable of the refactor. It must:

- State what the repository now is: the code and evidence behind a study of
  interaction-foundation-model transfer, originating as a class project on
  antibiotic design. Say that lineage plainly — it is honest and it explains the
  directory names.
- Give a **claim → evidence table**: each of the paper's findings mapped to the
  `results/` subdirectory, script, and raw data that produce it. Source the
  claims from `docs/PREPRINT_PROMPT.md`, which carries every number.
- Note the pre-registration practice and point at the analysis plans by path.
- Give exact environment setup: `environment-atomica.yml` for
  `~/.conda/envs/atomica-interface` (torch 2.0.1 / CUDA 11.8), a separate
  `~/.conda/envs/ifp` for ProLIF/MDAnalysis/PLIP interaction fingerprints, and
  `~/.conda/envs/smina` for docking. Include the warning that `pip install` of
  anything depending on torch silently replaces the CUDA build with a CPU wheel.
- Not overstate the contribution. The evaluation metrics are **not** novel —
  PoseCheck (ProLIF interaction fingerprints, "interaction recovery") and Delta
  Score (binding specificity) are prior art and must be cited as the framework
  used. What is new is the transfer boundary and the critic mechanism.

Also update `CITATION.cff` to the new repository name and the study's framing,
and add a `docs/` note or `CHANGELOG.md` entry recording that this repository
succeeds the class project, with a link to the original.

---

## Step 5 — verify nothing is broken

- `pytest tests -q` must pass (these are ablation-result regression tests and
  need no GPU or dataset).
- `python -c "import atomica_interface"` in the `atomica-interface` env.
- Grep the surviving `results/*/README.md` and `docs/*.md` for links to any path
  you moved or deleted, and fix them. Broken internal links in the evidence
  directories are the most likely damage from this refactor — check
  systematically, not by spot check.
- Confirm `git log --oneline | wc -l` on the branch is 153 + your new commits,
  and that `git log --format='%ad' --date=short | tail -1` still reads
  `2025-10-08`. If either changed, history was rewritten: stop and restore from
  the bundle.

## Step 6 — report

State: the new repo URL, commit count pushed, what you deleted and what you kept
against instruction 4a and why, whether tests pass, any broken links you fixed,
and the checkpoint-relocation proposal for the operator to decide. Open the PR
and give its URL.

**Do not make the repository public, do not touch `origin`, and do not rewrite
history.**
