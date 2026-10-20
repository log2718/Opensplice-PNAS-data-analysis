# B1 randomization pilot — middle-exon selection and experimental design

Planning and data-inspection only. No AlphaGenome calls, no library generated, no model trained, no existing file modified, nothing committed.

Companion to `UPSTREAM_EXON_B1_DESIGN_REVIEW.md` (architecture review).

Labels used throughout: **[VERIFIED]** computed from existing data or read from code in this session · **[INTERPRETATION]** my reading of those numbers · **[RECOMMENDATION]** a design proposal · **[OPEN]** unresolved.

---

## 0. The finding that should reshape the pilot

**[VERIFIED] Your existing data already shows that a B1-only swap moves plateau exons — and moves them *upward, off the plateau*.**

From `outputs/synthetic_context_8block/chimera8_ag_results.csv`, comparing `PPPPPPPP` (native PNAS) against `FPPPPPPP` (B1 replaced by the FAS/TP53 upstream exon body, all other blocks native):

| Pre-tuner regime | n exons | native median | B1-swapped median | median Δ | median \|Δ\| | max \|Δ\| |
|---|---|---|---|---|---|---|
| **Plateau, `PNAS_common` ≥ 5** | 26 | **+2.645** | **+3.789** | **+1.107** | 1.107 | 1.390 |
| Plateau, `PNAS_common` ≥ 2.5 | 62 | +2.622 | +3.623 | +0.786 | 0.907 | 1.390 |
| Rising, −5 ≤ x < 2.5 | 282 | +0.380 | −0.962 | **−0.914** | 0.918 | 2.385 |
| Low, x < −10 | 300 | −4.209 | −4.740 | −0.428 | 0.428 | 1.224 |

Three things follow.

1. **[VERIFIED] The plateau is not a ceiling.** A single 113-nt block substitution lifts plateau exons from ~+2.6 to ~+3.8 — well past the plateau level of +2.5 — using only one alternative B1 sequence. The y-axis has headroom.
2. **[VERIFIED] The sign of the B1 effect flips with regime.** B1-swap is **positive** (+1.1) on the plateau and **negative** (−0.9) in the rising region. That is a strong B1 × middle-exon-strength interaction, and it is exactly the kind of thing the planned model would need to capture.
3. **[INTERPRETATION] The pilot's question should therefore be sharpened.** "Can B1 move a plateau exon at all?" is already answered: yes, by ≈1.1 logit units, for one specific alternative sequence. The open questions are (a) what the *distribution* of effects looks like over many random B1s, (b) whether random B1s behave like the FAS B1 or are mostly neutral, and (c) whether the upward direction is general or specific to the FAS sequence. Design the pilot around those.

A useful consistency check that fell out of the same table: **[VERIFIED]** the `TP53_e6` and `TP53_e7` arms give byte-identical Δ values, which is correct — both share `FAS_E5`, so their B1 blocks are the same sequence. The pipeline is self-consistent.

---

## 1. The figure, grounded in its data

### 1.1 Provenance **[VERIFIED]**

The attached panel is the `PNAS_original_1309` panel of `outputs/synthetic_context/plots/original_context/original_context_three_panel_mean_logit.png`, produced by `analysis_script/analyze_synthetic_mean_logit.py`:

- `X = "PNAS_common"` (`analyze_synthetic_mean_logit.py:59`), labelled "PNAS pre-tuner score (common) (energy_seq_struct)" (`:61`)
- `Y = "AG_mean_logit"`
- panel builder `_hist_panel` (`:104-122`), 2-D histogram with a log count scale
- data: `outputs/synthetic_context/synthetic_endpoint_ag_results.csv` filtered to `context == "PNAS"` (`context_label == "PPPPPP"`), joined to `outputs/synthetic_context/synthetic_exon_library.csv`
- n = 10,000, matching the panel's subtitle exactly

**[VERIFIED] What `PNAS_common` is.** `run_synthetic_endpoint_experiment.py:66-71` computes it once per exon via `pnas_prediction_hybrid_input.score_pnas_pretuner_batch` on the 90-nt input `PNAS_INTRON1[-7:] + (GTT + syn70 + CAG) + PNAS_INTRON2[:7]`. That is the `SumDiff` scalar `w·(Σ incl − Σ skip) + b`, i.e. `energy_seq_struct`, the pre-tuner — it includes **both** the sequence and the structure branches, so it is not identical to `compute_sr_balance`, which is sequence-only. The script's own docstring (`:14-17`) states the intent: one x per exon, reused under every context, so any y shift is attributable to context.

Column names verified directly, not assumed:

```
synthetic_endpoint_ag_results.csv : row_id, sequence_id, context, context_label, alpha,
    PNAS_common, exon_start, exon_len, AG_acceptor_probability, AG_donor_probability,
    AG_mean_prob, AG_mean_logit, AG_prod_logit, core_length
synthetic_exon_library.csv        : sequence_id, alpha, target_pA..target_pT,
    count_A..count_T, synthetic_70mer, full_middle_exon, PNAS_common
```

### 1.2 Is the flattening a plateau? **[VERIFIED], with an important qualification**

Binned medians of `AG_mean_logit` against `PNAS_common` (PNAS context, all 10,000):

| `PNAS_common` bin | n | median y | IQR |
|---|---|---|---|
| (−7.5, −5.0] | 984 | −2.597 | 1.41 |
| (−5.0, −2.5] | 1161 | −1.161 | 2.09 |
| (−2.5, 0.0] | 1239 | +0.799 | 2.22 |
| (0.0, 2.5] | 1131 | +2.002 | 1.33 |
| **(2.5, 5.0]** | 679 | **+2.423** | 0.90 |
| **(5.0, 7.5]** | 299 | **+2.556** | 0.69 |
| **(7.5, 10.0]** | 99 | **+2.583** | 0.99 |
| **(10.0, 12.5]** | 20 | **+2.469** | 0.83 |
| **(12.5, 15.0]** | 3 | **+2.613** | 0.30 |

OLS slope of y on x with 2,000-sample bootstrap CIs:

| Window | n | slope | 95 % CI | median y |
|---|---|---|---|---|
| rising, −7.5 ≤ x < 0 | 3,384 | **+0.6038** | [+0.5801, +0.6274] | −1.094 |
| transition, 0 ≤ x < 2.5 | 1,131 | +0.2785 | [+0.1564, +0.4035] | +2.002 |
| x ≥ 2.5 | 1,100 | +0.0630 | **[+0.0344, +0.0941]** | +2.479 |
| x ≥ 3 | 935 | +0.0636 | [+0.0311, +0.0977] | +2.512 |
| **x ≥ 5** | 421 | **+0.0264** | **[−0.0240, +0.0776]** | +2.556 |

**[VERIFIED] The distinction you asked for.** The flattening is real and large — the slope drops ~9.6-fold from the rising region to x ≥ 2.5. But over x ≥ 2.5 the slope is **still significantly positive** (CI excludes zero), so that region is *strongly attenuated*, not statistically flat. Only at **x ≥ 5** does the CI include zero, making that the region where response is statistically indistinguishable from flat. Calling anything here a "verified plateau" would overstate it: the honest statement is *strong attenuation from x ≈ 2.5, statistically indistinguishable from flat beyond x ≈ 5, with n falling off sharply past x ≈ 10*.

**[VERIFIED] Your visual estimate was close but shifted left.** You read the plateau as x ∈ [0, +10], y ∈ [+2, +3]. The data put the onset at x ≈ +2.5: the (0, 2.5] bin still rises by +1.2 median over the bin below it, so 0–2.5 belongs to the transition, not the plateau. The y range +2 to +3 is accurate — plateau median +2.48, q90 +2.98.

**[VERIFIED] The flattening is of the *median*, not of individuals.** Within x ≥ 2.5 the IQR is 0.84 and the 10th percentile is **+0.748** — far below the plateau level, with a minimum of −3.92. A meaningful minority of high-pre-tuner exons are not high-AG at all. This is why selecting on x alone would be wrong, and confirms your instruction not to take the highest pre-tuner scores regardless of their AG values.

### 1.3 A quantization effect you should know about **[VERIFIED]**

AlphaGenome's `SPLICE_SITES` probabilities arrive on a discrete grid. The six largest unique acceptor values in the PNAS context are

```
0.95312, 0.95703, 0.96094, 0.96484, 0.96875, 0.97266
```

— successive differences of exactly **1/256 = 0.003906**. Observed maximum is 0.97266 for both acceptor and donor (reached by 3 and 1 exons respectively).

**[INTERPRETATION] Consequence for a plateau study.** Quantization in probability becomes coarse in logit space as p → 1. At the plateau level (p ≈ 0.92) one quantum is ≈ 0.046 logit units; at p ≈ 0.97 it is ≈ 0.14. The plateau's IQR of 0.84 is ~18 quanta wide, so the flattening is **not** a quantization artifact — but single-variant differences smaller than ≈ 0.05 logit units are at the resolution floor and should not be interpreted. **[RECOMMENDATION]** Set the pilot's minimum interpretable effect size at 0.1 logit units (≈ 2 quanta) and report the quantum alongside any effect.

Note this is a *grid* effect, not clipping: nothing is pinned at the top value, and the `_logit` eps clip at ±16.1 is never approached. Consistent with your instruction that we are not looking for probability clipping near 1.

### 1.4 So what is the plateau, mechanistically? **[INTERPRETATION]**

Three readings are consistent with the data, and they are not mutually exclusive:

1. **AlphaGenome's own response is compressive at high splice-site strength.** Once acceptor and donor are each ≈0.92–0.97, further intrinsic exon strength has little left to add in AlphaGenome's representation.
2. **The pre-tuner and AlphaGenome stop agreeing at the top.** `PNAS_common` keeps increasing (to +14.8) because `SumDiff` is an unbounded sum over filters and positions; AlphaGenome need not track it. The flattening may be as much about the x-axis continuing to grow as about y stopping.
3. **The limiting factor shifts from the exon to its context.** This is the reading the chimera data most supports: once the middle exon is intrinsically strong, what remains variable is the surrounding construct — which is precisely why a B1 perturbation can still move it (§0).

**[VERIFIED] Reading 3 has direct support.** For the five plateau candidates below, a *full* context swap (PNAS → FAS/TP53) moves `AG_mean_logit` from ≈ +2.9 down to between +0.4 and −3.2 — swings of 2.5 to 6.0 logit units. The plateau level is not an upper bound imposed by the exon; it is where the native PNAS context happens to place a strong exon.

**[OPEN]** Which reading dominates cannot be settled from these data. It matters for interpretation but not for the pilot design, since all three permit B1 to have an effect.

---

## 2. Candidate middle exons

### 2.1 Selection procedure **[VERIFIED]**

1. Restrict to the PNAS context (`context == "PNAS"`), n = 10,000.
2. Require **both** axes: `PNAS_common ≥ 3.0` **and** `AG_mean_logit ≥ 2.2`. The joint requirement is the point — it excludes the low-y tail inside the plateau x-range. → 633 exons.
3. Composition vetting, to avoid the Dirichlet library's deliberate extremes: longest homopolymer ≤ 5, max single-base fraction ≤ 0.45, GC ∈ [0.35, 0.70], ≥ 45 distinct 3-mers (of 64). → **13 exons**.

For scale, the library as a whole is composition-extreme by construction: median longest homopolymer **6**, 90th percentile **24**; median max-base fraction 0.529; median distinct 3-mers only 24 of 64. The vetting filter is doing real work. A slightly relaxed variant (homopolymer ≤ 7, max-base ≤ 0.50, ≥ 38 distinct 3-mers) was used only to reach the far-right end of the plateau, where the clean pool runs out.

### 2.2 Recommended candidates **[RECOMMENDATION]**

Five exons spanning the attenuated region from onset (x ≈ 3.2) to its far end (x ≈ 10.3), across three Dirichlet α regimes.

| # | `sequence_id` | α | pre-tuner | AG mean-logit | acceptor | donor | GC | max homo | distinct 3-mers | A/C/G/T |
|---|---|---|---|---|---|---|---|---|---|---|
| **1** | `syn_a5.0_00482` | 5.0 | **+10.252** | +2.931 | 0.9531 | 0.9453 | 0.600 | 3 | 38 | 15/28/14/13 |
| **2** | `syn_a5.0_00535` | 5.0 | **+8.790** | +2.906 | 0.9375 | 0.9570 | 0.500 | 3 | 39 | 24/13/22/11 |
| **3** | `syn_a0.5_00475` | 0.5 | **+6.344** | +2.632 | 0.9180 | 0.9453 | 0.557 | 3 | 46 | 18/20/19/13 |
| **4** | `syn_a1.0_00422` | 1.0 | **+4.251** | **+3.065** | 0.9492 | 0.9609 | 0.557 | 3 | 45 | 16/27/12/15 |
| **5** | `syn_a5.0_01784` | 5.0 | **+3.194** | +2.736 | 0.9062 | 0.9609 | 0.514 | 3 | 45 | 16/18/18/18 |

Sequences (`synthetic_70mer`, then `full_middle_exon` = `GTT` + 70-mer + `CAG`):

**1. `syn_a5.0_00482`** — far right end of the attenuated region, highest clean pre-tuner available.
```
ATCACGCCTCTTACTGAGACGCCGAATGCCACCTGTCTACGCCCGTCCAGAAGCCCGACGTCGCCAACTT
GTTATCACGCCTCTTACTGAGACGCCGAATGCCACCTGTCTACGCCCGTCCAGAAGCCCGACGTCGCCAACTTCAG
```
**2. `syn_a5.0_00535`** — deep in the statistically-flat zone (x ≥ 5), purine-rich (A+G = 46/70) and so compositionally distinct from #1, which is C-rich.
```
GACGGTCTGAATTACCTGCGCCTAAAGCGAAGACGAATGCGGATATGGAAACGATGGAGCAGTGACAGAA
GTTGACGGTCTGAATTACCTGCGCCTAAAGCGAAGACGAATGCGGATATGGAAACGATGGAGCAGTGACAGAACAG
```
**3. `syn_a0.5_00475`** — still inside x ≥ 5, drawn from a *different* Dirichlet regime (α = 0.5) yet with clean composition; guards against an α = 5.0-specific result.
```
CACCGATGTGCGGTCAATTTCGAAAGCTATTACCGACGGAAGCGCACGACCGGATCCCATCGTAGACTGG
GTTCACCGATGTGCGGTCAATTTCGAAAGCTATTACCGACGGAAGCGCACGACCGGATCCCATCGTAGACTGGCAG
```
**4. `syn_a1.0_00422`** — **highest AG mean-logit of the five** (+3.065) at a modest pre-tuner; the hardest case to move further up, and so the most stringent test of headroom.
```
CAACCTAATCGCACTCACCAAAGGACCATCCTCCCGATTCGGTGGCCATCGCTTACTCCACGCGTGTCTA
GTTCAACCTAATCGCACTCACCAAAGGACCATCCTCCCGATTCGGTGGCCATCGCTTACTCCACGCGTGTCTACAG
```
**5. `syn_a5.0_01784`** — plateau *onset*; near-perfectly balanced composition (16/18/18/18), the most "neutral" sequence in the set.
```
TGGCACGACGACCAGGGCGACGTGATTGCGGCCATAAATCGTACTCTACTTCCTGAAATGCGTTGTTACT
GTTTGGCACGACGACCAGGGCGACGTGATTGCGGCCATAAATCGTACTCTACTTCCTGAAATGCGTTGTTACTCAG
```

Why these five and not the top five by pre-tuner: they span x = 3.2 → 10.3 rather than clustering; they cover α = 0.5, 1.0 and 5.0; #1 is C-rich and #2 purine-rich, so a composition-driven artifact would show as a split between them; #4 sits highest on y and #5 at the onset, bracketing the plateau vertically as well as horizontally. **[VERIFIED]** None of the five is in the 400-exon `chimera8` subset, so none has a pre-existing B1-swap measurement — the pilot will not be re-deriving a known answer.

### 2.3 Rising-region comparison controls **[RECOMMENDATION]**

Two exons from the steep part of the curve (local slope ≈ +0.60), composition-vetted by the same filter:

| `sequence_id` | α | pre-tuner | AG mean-logit | acceptor | donor | GC | max homo |
|---|---|---|---|---|---|---|---|
| `syn_a1.0_02380` | 1.0 | −0.604 | +2.022 | 0.8750 | 0.8906 | 0.543 | 3 |
| `syn_a1.0_00359` | 1.0 | −0.480 | +2.098 | 0.8945 | 0.8867 | 0.571 | 4 |

```
syn_a1.0_02380  TCGAGTGAATCTTCGCGTTCCGGATGCGATAGCTTCACCCAGTCCACGTCCGTAACCTTTATCCGTGGGT
syn_a1.0_00359  AGGCTCTGATTGTGTACCCCAACTCGCGAGTCAGGAATGACCTGCACTAGGCGGAAGCCCTATCTCCTCG
```

These are **secondary**. Their role is to test whether the sign flip seen in §0 (B1-swap negative in the rising region, positive on the plateau) reproduces across many random B1s — which would be a far more interesting result than either regime alone.

### 2.4 Final-exon selection criteria for the scale-up **[RECOMMENDATION]**

After the pilot, choose the single fixed middle exon for the large training run by, in priority order:

1. **Largest spread of `AG_mean_logit` across B1 variants** (SD, and p95 − p5). The model needs variance in its target; an exon whose response is flat across B1s yields an untrainable dataset regardless of how interesting it is.
2. **Spread not dominated by a few outliers** — check that the distribution is not one tight mode plus three escapees. A reasonable criterion: interquartile range ≥ 0.3 logit units, well above the 0.046 quantum.
3. **Effect direction consistent with the FAS-B1 reference.** If the FAS B1 lands inside the random-B1 distribution rather than far outside it, the random panel is exploring the relevant space.
4. **Composition neutrality**, as a tiebreak — prefer #5 or #2 over #1 if the first three criteria are close, so that results are not confounded with an unusual middle-exon composition.
5. **Keep a second exon as a replication arm** in the scale-up if budget allows (§7).

---

## 3. B1-only randomization design

### 3.1 What is held fixed **[VERIFIED]**

From `analysis_script/splice_site_partition.py`, PNAS context:

| Block | Role | Length | Status in this experiment |
|---|---|---|---|
| **B1** | upstream exon body | **113** | **THE ONLY VARIABLE** |
| B2 | upstream exon 5′SS (3 exonic + 6 intronic) | 9 | fixed — `CAGGTTGGT` |
| B3 | upstream intron body | 101 | fixed |
| B4 | middle exon 3′SS (20 intronic + `GTT`) | 23 | fixed — `TGTCTTTCTCTGCCATCCAGGTT` |
| — | `synthetic70` | 70 | fixed — the selected candidate |
| B5 | middle exon 5′SS (`CAG` + 6 intronic) | 9 | fixed — `CAGGTCTGA` |
| B6 | downstream intron body | 823 | fixed |
| B7 | downstream exon 3′SS | 23 | fixed — `CTCTTATCTTCCTCCCACAGCTC` |
| B8 | downstream exon body | 138 | fixed |

`B1 = PNAS_EXON1[:-3]` — the final three exonic nucleotides of the upstream exon belong to B2 and are preserved, exactly as you specified. Total construct remains **1,309 nt**.

**[VERIFIED] The assembly machinery already does this correctly.** `assemble8(label, synthetic70, pb, fb)` returns `(core, exon_start)` with `exon_start` computed from the realized left side, so the middle exon's AlphaGenome read positions stay correct. Substituting B1 is a one-line dict edit on `pnas_blocks8()`:

```
pb = SP.pnas_blocks8()
pb["B1"] = <113-nt variant>
core, exon_start = SP.assemble8("PPPPPPPP", syn70, pb, pb)
```

Passing `pb` as both the P and F block source guarantees every other block is native regardless of the label — a useful safety property worth keeping.

**[RECOMMENDATION] Required per-construct assertions** before any call is spent, mirroring the discipline in `build_chimera8_manifest.py`:

- `len(core) == 1309`
- `core[exon_start : exon_start + 76] == "GTT" + syn70 + "CAG"`
- `core.count(syn70) == 1`
- blocks B2–B8, re-extracted from the assembled core by offset, are byte-identical to `pnas_blocks8()`
- `len(b1_variant) == 113` and alphabet ⊆ {A, C, G, T}
- `sha256(core)` recorded in the manifest and re-verified inside the scoring loop

### 3.2 How to generate B1 **[RECOMMENDATION]**

| Method | Advantages | Disadvantages | Role in pilot |
|---|---|---|---|
| **Uniform random (25 % each)** | maximum entropy, every position independent, cleanest identifiability for position-bias parameters later, trivially reproducible from a seed | far out of distribution for real exonic sequence — GC ≈ 0.50, no codon structure, no ESE density; AlphaGenome may treat it as non-exonic and the effect may say more about "degraded exon" than "different exon" | **primary arm** |
| **Composition-matched to native B1** | removes GC/composition as a confound; isolates *motif arrangement* from *base content* | still no higher-order structure; needs native B1 composition computed first | **primary arm** |
| **Mononucleotide shuffle of native B1** | exactly preserves composition by construction; the canonical null | destroys all motif and codon structure | control |
| **Dinucleotide-preserving shuffle (Altschul–Erikson)** | preserves composition *and* dinucleotide frequencies — the strongest sequence-level null; difference from native is attributable to higher-order structure | implementation care needed for correctness | **control, high value** |
| **Natural upstream exon bodies** (113-nt windows from real internal exons) | in-distribution; the only arm that tests biologically realistic variation; directly interpretable | requires sourcing and trimming real exons to exactly 113 nt; introduces length-trimming artifacts; genomic context is lost | **secondary arm, high value** |
| **Native PNAS B1** | the anchor — reproduces the known `PPPPPPPP` value | n = 1 | **mandatory reference** |
| **FAS/TP53 B1** | **a known +1.1 logit reference point** (§0); ties the pilot to completed work | n = 1 | **mandatory reference** |

**[RECOMMENDATION] Pilot composition.** Uniform random and composition-matched as the two primary arms, dinucleotide shuffles as the null, plus the two single-sequence references. Natural exons are the most scientifically informative arm but need sourcing work; include them if that is cheap, otherwise defer to the scale-up. Markov generation is correctly deprioritized — the dinucleotide shuffle already covers first-order structure.

**[INTERPRETATION] The out-of-distribution risk is the main scientific threat to this design.** If uniform-random B1 produces a large effect but composition-matched and natural B1s produce none, the honest conclusion is "destroying the upstream exon changes the prediction", which is much weaker than "upstream exon sequence modulates middle-exon recognition". Having all three arms in the pilot is what lets you tell those apart, and it is why I would not run a uniform-only pilot even though it is the simplest.

B1 length stays fixed at **113 nt** throughout, as you specified.

---

## 4. Pilot plan

### 4.1 Design **[RECOMMENDATION]**

| Arm | Variants per exon | Applied to | Purpose |
|---|---|---|---|
| Native PNAS B1 | 1 | all 7 exons | anchor; must reproduce the known `PPPPPPPP` value |
| FAS/TP53 B1 | 1 | all 7 exons | known reference (+1.1 on plateau exons) |
| Uniform random | 16 | all 7 exons | primary; maximum-entropy probe |
| Composition-matched | 12 | all 7 exons | primary; GC/content controlled |
| Dinucleotide shuffle of native B1 | 8 | all 7 exons | null; higher-order structure removed |
| **Total per exon** | **38** | | |

Applied to **5 plateau candidates + 2 rising-region controls = 7 exons**.

**Total unique AlphaGenome calls: 7 × 38 = 266.**

For scale: the completed 8-block run spent ~155,000 calls, so 266 is negligible. Budget ~300 to allow a few re-runs.

**[RECOMMENDATION] One addition worth its cost.** Score the native-B1 construct for each exon **twice**, in separate calls, to measure AlphaGenome's call-to-call reproducibility directly (+7 calls, total 273). Every effect size in the pilot is interpreted against that noise floor, and right now we are assuming determinism rather than having measured it. **[OPEN]** If AlphaGenome is strictly deterministic for identical input, these duplicates are free confirmation; if not, the duplicate spread is the single most important number in the pilot.

### 4.2 Pipeline

Reuse without modification, per the architecture review:

- `splice_site_partition.pnas_blocks8`, `assemble8` — construct assembly
- `synthetic_context_lib.score_core` — the `AG_mean_logit` definition, unchanged, so numbers are directly comparable to the endpoint and chimera runs
- `synthetic_context_lib.run_scoring` — resumable, `row_id`-keyed, single-stream, 90 s watchdog
- `alphagenome_local_prediction_pipeline_minigene` — client, `center_pad`, track extraction

New code is limited to B1 generation, manifest building and analysis. All outputs under a new namespace, e.g. `outputs/upstream_exon_b1/pilot/`, writing nothing into `outputs/synthetic_context/` or `outputs/synthetic_context_8block/`.

### 4.3 Outputs

**Tables.** `b1_pilot_manifest.csv` (row_id, exon id, arm, `b1_sha256`, `core_sha256`, exon_start) · `b1_pilot_ag_results.csv` (manifest fields + acceptor, donor, `AG_mean_logit`, `AG_prod_logit`) · `b1_pilot_summary.csv` (per exon × arm: n, median, IQR, SD, min, max, Δ vs native, effect-size-to-noise ratio).

**Plots.**
1. Per-exon strip/violin of `AG_mean_logit` by arm, with the native and FAS references as horizontal lines and the plateau level (+2.48) and quantum (0.046) marked.
2. Δ-from-native distributions, all exons overlaid, split plateau vs rising — the direct test of the §0 sign flip.
3. The §1.2 binned-median curve with each pilot exon's full B1 range drawn as a **vertical bar at its fixed x** — the single most communicative figure, since it shows directly whether B1 variation spans more of the y-axis than the plateau's x-variation does.
4. Effect size against middle-exon pre-tuner, testing whether B1 sensitivity depends on exon strength.
5. Acceptor vs donor Δ scatter — do both sites move together, or is one carrying the effect?

### 4.4 Decision criteria **[RECOMMENDATION]**

Let *σ_rep* be the replicate spread from §4.1 (if AlphaGenome is deterministic, use the 0.046 quantum as the floor).

| Outcome for a given exon | Reading | Action |
|---|---|---|
| Random-B1 SD ≥ 0.3 **and** p95−p5 ≥ 0.8, with composition-matched arm showing a comparable spread | B1 genuinely modulates a plateau exon, not via composition alone | **proceed to scale-up with this exon** |
| Uniform arm spreads but composition-matched and dinucleotide arms do not | effect is driven by base composition / exon degradation | reframe: the variable is composition, and the model should be told so; reconsider the randomization scheme before scaling |
| Spread ≤ 2 σ_rep across all arms | B1 does not move this exon | drop it; escalate to the next candidate |
| Plateau exons move but rising-region controls move more | B1 sensitivity is highest where the exon is weak | scientifically interesting; the scale-up should probably not use a plateau exon |
| Sign flip from §0 reproduces (plateau positive, rising negative) | strong B1 × exon-strength interaction | high-value result; design the scale-up to span both regimes |

**[RECOMMENDATION] Overall go/no-go:** proceed to the large run if **at least two** plateau candidates meet the first row's criteria. One alone risks building a training set on an exon-specific quirk.

### 4.5 What the pilot does and does not establish **[INTERPRETATION]**

It *can* establish: that `AG_mean_logit` for a fixed, intrinsically strong middle exon varies with upstream-exon sequence; the magnitude and direction of that variation; whether it survives composition control; and whether the target has enough variance to train on.

It *cannot* establish: that the model has "escaped saturation" in any mechanistic sense. A y-shift at fixed x says AlphaGenome's prediction changed — not that a saturation mechanism was relieved. The x-axis is a PNAS pre-tuner score computed on a 90-nt window that does not include B1, so by construction it cannot move; "escaping the plateau" is therefore a statement about a 2-D plot, not about a mechanism. **[RECOMMENDATION] Phrase results as "upstream-exon sequence modulates AlphaGenome's middle-exon recognition independently of the middle exon's own PNAS pre-tuner score"** and keep mechanistic claims out of it. A genuine saturation claim would need an independent handle on middle-exon strength, varied while B1 is held fixed.

---

## 5. Connection to the model

Brief, as requested — no implementation.

- **Input:** the 113-nt B1 sequence, one-hot `(N, 4, 113)`. Fixed length by design, so no position-bias resampling. Structure channels deferred (see the architecture review, §C.6).
- **Target:** `AG_mean_logit` from `score_core`, unchanged — a single unbounded scalar per construct.
- **Pilot data as training examples:** directly usable, joined on `b1_sha256`. 266 rows is far too few to train on (vs 2,160 position-bias parameters alone) but exactly right for validating the end-to-end join, the encoding, and the target's distribution — and the per-arm spreads set the noise floor against which any later model's R² must be judged.
- **One exon or several?** **[RECOMMENDATION] Model a single fixed exon first.** With one exon the target is a pure function of B1 and the model is directly interpretable. Pooling exons requires either a per-exon intercept or conditioning the model on the middle exon, and if the §0 sign flip is real then the B1 effect *interacts* with exon strength — meaning a pooled model without an interaction term would average two opposite effects toward zero. Pool later, deliberately, as a second experiment.
- **Before scaling up, establish:** (i) that spread exceeds the noise floor for the chosen exon; (ii) that the effect survives composition control; (iii) the call-to-call reproducibility; (iv) the target's distribution and whether it is multimodal; (v) the number of variants needed, estimated from the pilot's observed effect size.

---

## 6. Deliverable summary

1. **Saturation region explained** — §1.2–1.4: onset x ≈ +2.5, statistically flat only beyond x ≈ +5, level ≈ +2.48 median, slope attenuated 9.6× but not zero until x ≥ 5; flattening is of the median, with large residual spread; not a probability-clipping artifact, though a 1/256 quantization grid sets a 0.046-logit resolution floor.
2. **Candidate table** — §2.2, five plateau exons with real IDs, sequences and measured scores; §2.3, two rising-region controls.
3. **Final-exon criteria** — §2.4, led by spread of response across B1 variants.
4. **B1-only design** — §3, 113 nt, B2–B8 byte-identical, six per-construct assertions, SHA-256 manifest.
5. **Pilot plan** — §4, 7 exons × 38 variants + 7 replicates = **273 calls**, five arms including two reference sequences.
6. **Analysis plan** — §4.3, five figures and three tables; the key figure is vertical B1-range bars overlaid on the binned-median curve.
7. **Open questions** — §7.

---

## 7. Open questions to settle before implementation

1. **[OPEN] Is AlphaGenome deterministic for identical input?** Never measured here. Addressed by the 7 duplicate calls in §4.1. Everything else is interpreted against this number.
2. **[OPEN] Does the acceptor/donor peak stay at the expected index when B1 is randomized?** `score_core` reads one fixed position per site. Randomizing 113 nt is a larger perturbation than previous experiments applied. **[RECOMMENDATION]** Record a ±5 nt window around both positions in the pilot and confirm the argmax does not move; keep reporting the exact-index value regardless, for comparability.
3. **[OPEN] Could a random B1 create a cryptic splice site?** A 113-nt random sequence can contain `AG`/`GT` dinucleotides in plausible contexts, and a new competing site would change the middle exon's prediction through an uninteresting mechanism. **[RECOMMENDATION]** Scan the whole padded construct's splice-site tracks for novel peaks above a threshold, not just the two target positions, and flag affected variants. This is cheap — the tracks are already returned — and it is the most likely confound in the whole design.
4. **[OPEN] Is uniform-random B1 so out-of-distribution that the result is uninterpretable?** Mitigated but not resolved by the composition-matched and dinucleotide arms (§3.2). The natural-exon arm is what would settle it.
5. **[OPEN] How many variants will the scale-up need?** Estimable only from the pilot's effect size relative to noise. The architecture review's 4,000–8,000 is a placeholder.
6. **[OPEN] Should the x-axis be reconsidered?** `PNAS_common` is fixed by construction across B1 variants, which makes the "escape the plateau" framing geometrically odd (§4.5). **[RECOMMENDATION]** Discuss whether a pre-tuner computed over a window that *includes* B1 would be a more meaningful x-axis for this specific experiment — it would change the plot's meaning substantially and should be a deliberate decision, not a default.
7. **[OPEN] Why does the left branch of the figure turn non-monotone below x ≈ −35?** Median rises again toward −2.1 at the extreme left. `analysis_script/diagnose_endpoint_spikes.py` exists and appears to address exactly this. Not relevant to the plateau work, but it bears on how much to trust the pre-tuner axis globally.
