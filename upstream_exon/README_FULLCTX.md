# Full-context libraries and the logit-average metric

Three things change here relative to v1–v3. All three are deliberate, and
none of them touches anything already on disk.

| | before (v1–v3) | now |
|---|---|---|
| input to AlphaGenome | 1,309-nt cassette rebuilt from `pnas_blocks8()`, N-padded to 16,384 | the **entire 8,612-nt ExonSkip7_Lib2 context**, N-padded to 16,384 |
| output requested | `SPLICE_SITES` | `SPLICE_SITE_USAGE` |
| target | `AG_mean_logit = (logit(p_acc) + logit(p_don)) / 2` | `logit_average = logit((u_acc + u_don) / 2)` |

**From here on, logit-average is the only target.**

## The metric, precisely

```
logit_average = logit( (usage_acceptor + usage_donor) / 2 )
```

Average on the **probability** scale, take the logit **once**, at the end.

This is not a cosmetic rewrite of `AG_mean_logit`. Taking the mean of two
logits is not the logit of the mean: logit is concave below 0.5 and convex
above it, so by Jensen's inequality the two statistics disagree whenever the
acceptor and the donor differ, and they disagree most in the tails. On
`(0.9, 0.5)` the old statistic gives 1.0986 and the new one 0.8473. **The two
numbers must never be pooled, compared, or plotted on a shared axis.**

`SPLICE_SITE_USAGE` also answers a different question from `SPLICE_SITES`.
`SPLICE_SITES` says a position *looks* like a splice site; `SPLICE_SITE_USAGE`
says what fraction of transcripts actually *use* it, which is what the PNAS
assay measures.

## Geometry

Derived at import by locating the repo's own PNAS blocks inside the context,
then asserted against recorded constants, so a changed context file fails
immediately rather than scoring the wrong positions.

| feature | 0-based span in the 8,612-nt context |
|---|---|
| B1 — upstream exon body (113 nt) | `[5885, 5998)` |
| middle exon body (70 nt, the `N` run) | `[6131, 6201)` |
| middle exon, with `GTT` / `CAG` | `[6128, 6204)` |
| **acceptor** | `6128` → padded `10014` |
| **donor** | `6203` → padded `10089` |
| library barcode (20 nt, left as `N`) | `[7195, 7215)` |

Padding: 8,612 is not an accepted AlphaGenome input length, so the context is
centre-padded with `N` to 16,384 (`left_pad = 3886`).

Two things to know about the context file:

- **One extra `T` at context position 6408**, inside intron 2, relative to the
  repo's reconstructed 1,309-nt cassette. Everything else is identical. The
  context file is treated as ground truth; the cassette is never rebuilt from
  blocks here.
- **The 20-nt barcode is left as `N`.** It is a per-clone tag, unknown for any
  given library member, so `N` is the honest encoding. `build_construct(...,
  barcode=...)` can fill it, but that changes the input and therefore the
  labels.

The RNA is circular. The cut point is the one in the text file; it is taken as
given and not re-derived.

## The two libraries

Both are **exact-composition**: every sequence is a permutation of one fixed
multiset, so composition cannot vary and any response variation is
attributable to nucleotide **order**. This is the v3 design, carried over.

| | `upstream` | `middle` |
|---|---|---|
| varies | B1, 113 nt | middle exon body, 70 nt |
| composition | 29 A, 28 C, 28 G, 28 T | 18 A, 18 T, 17 C, 17 G |
| held fixed | middle exon = `syn_a5.0_00482` | B1 = native PNAS |
| sequences | 1,000,000 | 500,000 |
| train / val / test | 800,000 / 40,000 / 160,000 | 400,000 / 20,000 / 80,000 |
| seed / split seed | 20261115 / 20261116 | 20261117 / 20261118 |
| output dir | `outputs/upstream_exon/fullctx_upstream_1m/` | `outputs/upstream_exon/fullctx_middle_500k/` |

Why those compositions: 113 is not divisible by 4, so one base takes the extra
copy — A takes it, leaving G = C = 28, so GC content is a single number for
the library and there is no G/C asymmetry to read. 70 leaves a remainder of 2;
giving both extras to A and T keeps G = C = 17 **and** A = T = 18, balancing
the GC and purine/pyrimidine axes at once.

Why `middle` is the sanity check: B1 sits 130 nt upstream of the acceptor and
is not part of the exon being measured, so its effect is the surprising
result. Permuting the exon body itself is the positive control — sequence
*between* the two sites being read should move the readout. If it does not
move it at least as much as B1 does, the upstream result is not what it
appears to be.

## Backends, and what they cost

Everything downstream of the predictor — geometry, assembly, the metric, the
resumable loop — reaches it through exactly one method, so the backend is a
single swappable function, `fullctx_lib.make_model`:

```python
model.predict_sequence(sequence=..., organism=...,
                       requested_outputs=[OutputType.SPLICE_SITE_USAGE, ...],
                       ontology_terms=[...])
```

Select with `--backend` on `score_fullctx.py`, or `BACKEND=` on the sbatch.

### `api` (default) — the DeepMind gRPC client

This is what `alphagenome/src` in this repo is: **client code only**, a
`grpc.secure_channel` to Google authenticated from `.env.alphagenome`. Putting
it on `PYTHONPATH` makes the *source* local; the inference is not.

1,000,000 + 500,000 = **1,500,000 calls**. At this repo's measured 0.49–0.66
s/call on a single stream that is **204–275 hours — 8.5 to 11.5 days** of
continuous wall-clock scoring. v3's 250,000 calls took 35–46 h; this is six
times that. The SDK's own README warns it is "likely not suitable" for
analyses beyond about a million predictions.

Every previous run here was deliberately serialized (`%1`) because the quota
is not known to permit concurrent streams, and nothing in this repo has ever
tested that. If two streams are allowed, the libraries can run side by side
and the total halves.

### `local` — weights held on the cluster

**Not wired up yet**; `make_model("local", ...)` raises with the contract an
adapter has to satisfy. With local weights there is no quota and no
single-stream rule, which changes three things in the plan:

- the `%1` serialization and the 25,000-row shards become needlessly
  conservative — size them to GPU throughput instead;
- the `.scoring.lock` is still worth keeping (it guards the append-mode CSV,
  not the quota);
- `run_scoring` calls one sequence at a time because the API forced that. A
  local GPU wants batching, which is the one piece of real work the swap
  needs beyond the adapter itself.

The pilot exists either way, so the go/no-go is made on data rather than on
an estimate.

## Running it

Everything below is for you to run; nothing here submits jobs or makes calls
on its own.

```bash
# ---- 0. push from the laptop -------------------------------------------
git add -A && git commit && git push
# on Torch:
cd /scratch/yl12316/Opensplice-PNAS-data-analysis && git pull

# ---- 1. resolve the readout (3 calls) ----------------------------------
# MUST pass before anything else. Confirms splice_site_usage has one
# positive-strand track, that usages lie in [0,1], and that the designed
# sites are where the usage actually peaks.
python upstream_exon/probe_splice_site_usage.py --dry-run   # geometry, no calls
python upstream_exon/probe_splice_site_usage.py             # 3 calls

# ---- 2. build the manifests (no API calls) -----------------------------
python upstream_exon/build_fullctx_library.py --library upstream --validate-only
python upstream_exon/build_fullctx_library.py --library upstream   # ~120 MB
python upstream_exon/build_fullctx_library.py --library middle     # ~45 MB
# or: LIBRARY=upstream sbatch upstream_exon/slurm/build_fullctx.sbatch

# ---- 3. pilot (2,000 calls, ~25 min) -----------------------------------
mkdir -p logs
LIBRARY=upstream PILOT=2000 sbatch --array=0-0 --time=02:00:00 \
    upstream_exon/slurm/score_fullctx.sbatch

# ---- 4. full runs, only after looking at the pilot ---------------------
LIBRARY=upstream sbatch --array=0-39%1 upstream_exon/slurm/score_fullctx.sbatch
LIBRARY=middle   sbatch --array=0-19%1 upstream_exon/slurm/score_fullctx.sbatch
```

`--array=0-39%1` is 40 shards of 25,000, strictly one at a time. 25,000 ×
0.66 s/call = 4.6 h, inside the 6 h walltime with margin. A shard that hits
the wall is not a loss — the next resumes where it stopped, losing at most
`--flush-every` (200) calls.

Do **not** run steps 3 and 4 concurrently: they write different CSVs so the
lock will not stop them, but they share one quota.

## Safety rails

- **Resumable on `row_id`.** Re-running a shard never re-spends a call.
- **Exclusive lock** (`.scoring.lock`) — a second scorer on the same library
  refuses to start rather than interleaving rows mid-line.
- **Append-mode writes.** `synthetic_context_lib.run_scoring` rewrites the
  whole CSV at every flush, which is O(n²) over a run; at 250,000 rows that
  was wasteful, at 1,000,000 it would dominate the wall clock. The scorer here
  appends.
- **Refuses to overwrite a manifest**, because labels are keyed to manifest
  `row_id`s and a regenerated manifest would orphan them.
- **Connectivity and geometry preflights** in the sbatch, so a broken setup
  fails in seconds rather than 20,000 calls in.
- **`--also-splice-sites`** records the legacy `SPLICE_SITES` metric from the
  *same* call, at no extra call cost, if you want the old and new metrics on
  identical inputs for a comparison figure.

## Files

| file | role |
|---|---|
| `ExonSkip7_Lib2_text.txt` (repo root) | the 8,612-nt context |
| `fullctx_lib.py` | context loading, geometry, assembly, the metric, the resumable scorer |
| `probe_splice_site_usage.py` | resolves the usage layout — run first |
| `build_fullctx_library.py` | builds either manifest, no API calls |
| `score_fullctx.py` | the labeling driver |
| `slurm/build_fullctx.sbatch` | manifest build on a compute node |
| `slurm/score_fullctx.sbatch` | the labeling array |

Manifests and label CSVs are gitignored; `manifest_fingerprint.json` is
committed and carries a SHA-256 over the ordered sequences plus the context
hash, so a manifest rebuilt anywhere can be proven identical to the one that
was labeled.
