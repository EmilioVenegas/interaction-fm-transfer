# Validity gates for the interface-fidelity analysis — all pass

`scripts/interface_fp.py`, `scripts/interface_fidelity_gates.py` (FP-A),
`scripts/interface_fidelity_fpb.py` (FP-B), run 2026-08-16 against the rules
fixed in `ANALYSIS_PLAN.md` and its two amendments.

**The primary comparison is not in this document and was not computed.** These
are the harness checks that decide which decision tree the comparison walks
into. `critic_lambda20_r0` appears only in the size table (gate 4), the
pocket-set check (gate 5) and the chemical-validity counts (gate 6) — never in a
fingerprint or recovery value.

## Verdicts

| gate | result |
|---|---|
| 1 — headroom | **pass**, both fingerprints |
| 2 — dynamic range | **pass**, monotone in both |
| 3 — true-binder ceiling | **pass**, 92 same-site pairs over 20/44 pockets, 0 unusable |
| 4 — size matching | **pass**, 20.30–20.60 heavy atoms across all seven arms |
| 5 — same pockets | **pass**, 44/44 in every arm |
| 6 — protonation | **pass**, no branch V |
| 7 — no silent drops | **pass**, 44/44 receptors and references |
| FP-B viability screen | **pass — FP-B keeps co-primary status** |

## The scale, measured

| | FP-A Tanimoto | FP-A recall | FP-B Tanimoto | FP-B recall |
|---|---|---|---|---|
| placement null (right size, right place, no pocket chemistry) | 0.5107 | 0.6261 | 0.3115 | 0.4696 |
| real arms, pooled (λ = 0.7 arms only) | 0.6277 | 0.7786 | 0.3764 | 0.5769 |
| true-binder ceiling (median, 92 same-site pairs) | 0.7596 | 0.8333 | 0.5251 | 0.7500 |
| **usable range (ceiling − null)** | **0.2489** | 0.2073 | **0.2136** | 0.2804 |
| where the generated molecules sit in that range | 47% | 74% | **30%** | 38% |

Reference fingerprints: FP-A median 11 bits (min 6, max 20); FP-B median 13
(min 3, max 21).

**Gate 1 passes but bounds the claim.** A molecule of the right size dropped in
the right place already recovers 51% of the reference residue contacts and 31% of
the typed interactions. Most of the absolute score is positional. Every Δ is
therefore reported against the *range*, never against 1.0.

**Gate 2 makes the numbers physical.** Rigidly displacing the reference ligand:

| displacement | FP-A Tanimoto | FP-B Tanimoto |
|---|---|---|
| 0.5 Å | 0.8733 | 0.7012 |
| 1.0 Å | 0.7610 | 0.5366 |
| 2.0 Å | 0.5542 | 0.3642 |
| 3.0 Å | 0.4499 | 0.2971 |

Interpolating FP-A onto that curve: the placement null sits at ≈2.4 Å of
displacement, the generated molecules at ≈1.64 Å, a genuine second binder at
≈1.01 Å. **The whole span being hunted — generated molecule to alternative true
binder — is about 0.6 Å of equivalent rigid displacement.**

## Power

Seed-only noise from the six within-arm comparisons (`control_r{0,1,2}` among
themselves, `graph_cosine_r{0,1,2}` among themselves), at both scales:

| | s_arm (mean) | s_pocket (mean) | MDE = 2.8·s_pocket/√44 | as % of range |
|---|---|---|---|---|
| FP-A Tanimoto | 0.0076 | 0.0170 | **0.0072** | **2.9%** |
| FP-A recall | 0.0087 | 0.0209 | 0.0088 | 4.3% |
| FP-B Tanimoto | 0.0033 | 0.0198 | 0.0083 | 3.9% |
| FP-B recall | 0.0054 | 0.0296 | 0.0125 | 4.5% |

The two independent routes to a floor agree: the computed MDE (0.0072) and the
largest observed seed-only arm delta (0.0076) land in the same place for FP-A.

For contrast, cross-docking specificity has an MDE of 0.131 kcal/mol against
absolute specificities of 0.23–0.39 — a floor at roughly 40% of its own signal
scale. **This readout resolves about an order of magnitude finer relative to its
dynamic range.** That is the substantive argument for having run direction B at
all, and it is now measured rather than assumed.

## FP-B survived, against my prediction

The pre-registered worry was that FP-B would be too sparse to clear its own noise
floor, forcing the downgrade branch in which FP-A alone cannot exclude chemical
collapse. **That did not happen.** ProLIF encodes bits as (residue × interaction
type), so a single residue contributes several — median 13 reference bits rather
than the 3–8 projected from polar-contact counts. FP-B's MDE is 3.9–4.5% of its
range against FP-A's 2.9–4.3%: comparable, not swallowed.

Viability screen, both ways of computing it, against the 50% downgrade trigger:

| | pooled ceiling | per-pocket ceilings (n = 20) |
|---|---|---|
| Tanimoto | 0/44 fail | 5/20 fail |
| recall | 2/44 fail | 7/20 fail |

Under the trigger by either route, so **the intersection rule stands and chemical
collapse remains detectable.** The 2 pockets failing on recall (pooled null above
pooled ceiling) are excluded from FP-B's denominator, as pre-registered.

## Gate 6 in full

Union of hard failures, `AddHs` failures and valence repairs — the amended
threshold, which catches degradation that is merely *repairable*:

| arm | n | hard fail | repaired | union |
|---|---|---|---|---|
| control r0 / r1 / r2 | 4294 / 4317 / 4298 | 110 / 103 / 99 | 0 | 2.56% / 2.39% / 2.30% |
| critic λ=0.7 r0 / r1 / r2 | 4303 / 4323 / 4320 | 127 / 99 / 109 | 0 | 2.95% / 2.29% / 2.52% |
| **critic λ=20 r0** | 4308 | 101 | 0 | **2.34%** |

λ = 20 differs from the control mean by **0.07 percentage points** against a 2.0
pp threshold, and every arm is far under the 10% absolute bar. **No branch V**:
the critic is not buying its lower ATOMICA distance by breaking valences. The
repair path never fired on any arm, so the amendment did not bind — it was still
right to fix the loophole before the numbers existed rather than after.

## Two implementation findings worth carrying

- **FP-A must not sanitize.** It reads heavy-atom coordinates only, so valence
  models are irrelevant to it, and sanitising silently drops molecules on
  chemistry grounds inside a geometric measurement — including `complex_000148`'s
  own reference ligand (an N of explicit valence 4 with no charge block, the
  standard CrossDocked artefact). Dropping sanitisation raised per-arm counts from
  ~4,180 to ~4,300, i.e. **~2.5% of molecules had been silently discarded**. The
  guardrail table in `results/specificity/README.md` (n = 4,176 / 4,184) was
  computed with a sanitising reader; the loss is small and symmetric across arms,
  so nothing published is threatened, but the note belongs there.
- **`Chem.MolFromPDBFile` returns None for 1 of the 44 receptors.** Parsing ATOM
  records directly gives 44/44. Gate 7 exists because this project already lost
  target `9WT9` to a silent tokenizer `KeyError`.

## What happens next

All gates pass and FP-B keeps co-primary status, so the full pre-registered
decision table in `ANALYSIS_PLAN.md` is live: branches P (positive), C (chemical
collapse), N (negative) and A (ambiguous), with FP-A Tanimoto and FP-B recall as
the decision statistics and `s_arm` as the bar.

Nothing about any of this raises the reliability of a single λ = 20 seed. Branch
P still terminates in two further replicates before any claim.
