"""
Experiment driver: randomly sampled triple exonic substitutions for WT TP53_e6
and WT TP53_e7, scored with the same PNAS pretuner + AlphaGenome minigene
definitions as ``run_hybrid_pnas_alphagenome.py`` / ``run_tp53_double_substitutions.py``.

Biological goal
----------------
The pairwise (double) exonic substitution experiment
(``run_tp53_double_substitutions.py``) tested whether perturbing two exonic
positions at once pushes the PNAS pretuner / SR-balance range further than
single-SNV saturation. This experiment asks the same question one step
further out: do *triple* substitutions widen the PNAS range even more?

Full triple-mutant space (``C(n, 3) * 27``):

    TP53_e7 (length 110): C(110, 3) * 27 = 5,827,140
    TP53_e6 (length 113): C(113, 3) * 27 = 6,321,672

That is too large to score exhaustively with AlphaGenome, so this script does
**deterministic random sampling** instead of exhaustive enumeration: a fixed
number of unique triple mutants (``--sample-size``, default 50,000) is drawn
per exon, fully reproducibly from ``--seed``.

Reuse, not reimplementation
----------------------------
This script does **not** redefine any scoring logic. It reuses, verbatim:

* ``run_hybrid_pnas_alphagenome.load_opensplice_exon`` to build the WT
  ``HybridExon`` (exon_seq / upstream_flank / downstream_flank) for
  TP53_e6 / TP53_e7 from ``arush_data/wt_exons.csv``.
* ``run_hybrid_pnas_alphagenome.score_hybrid_variants`` for both PNAS
  (batched) and AlphaGenome (per-record) scoring, including the exact
  ``alphagenome_prod_logit = logit(P_acceptor * P_donor)`` and
  ``alphagenome_mean_logit`` definitions.

Only the *triple*-substitution sampling and chunk/merge/CLI plumbing are new
(and mirror ``run_tp53_double_substitutions.py``'s structure). Neither the
single-SNV pipeline nor the double-substitution script is modified.

Deterministic sampling
-----------------------
For a given ``(--exon, --seed)``, ``sample_triple_keys`` draws unique
``(i, j, k, alt_i, alt_j, alt_k)`` keys (``i < j < k``, each alt != the WT
base at that position) via rejection sampling from a ``random.Random`` seeded
by a SHA-256 digest of ``f"{seed}:{exon_id}"`` (not Python's built-in
``hash()``, which is salted per-process for strings and would break
reproducibility across runs). The **full** ``sample_size``-length sample is
always generated first and canonicalized by sorting, and *only then* sliced
into chunks -- so the sample itself never depends on ``--chunk-index`` /
``--num-chunks``, and every chunk is a disjoint slice of one consistent
global sample (mirroring the double-substitution script's index-based
chunking, reused here via ``chunk_bounds``).

HPC / chunking
---------------
Identical Slurm-array pattern to ``run_tp53_double_substitutions.py``:

    # one array task per chunk, e.g. --array=0-19
    python analysis_script/run_tp53_triple_substitutions.py \\
        --exon TP53_e7 --chunk-index $SLURM_ARRAY_TASK_ID --num-chunks 20 \\
        --sample-size 50000 --seed 42 \\
        --alphagenome --env-path .env.alphagenome

    # after all chunks finish:
    python analysis_script/run_tp53_triple_substitutions.py \\
        --exon TP53_e7 --merge --sample-size 50000 --seed 42 --num-chunks 20
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Any, Sequence

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import pandas as pd  # noqa: E402

import run_hybrid_pnas_alphagenome as hybrid  # noqa: E402

DEFAULT_OPENSPLICE_CSV = hybrid.DEFAULT_OPENSPLICE_CSV
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "triple_substitutions"

_BASES = ("A", "C", "G", "T")
_EXON_CHOICES = ("TP53_e6", "TP53_e7")

TripleKey = tuple[int, int, int, str, str, str]

__all__ = [
    "total_triple_variants",
    "sample_triple_keys",
    "chunk_bounds",
    "build_variant_records",
]


# ==========================================================================
# Deterministic triple-substitution sampling
# ==========================================================================

def _alts(ref: str, alphabet: Sequence[str] = _BASES) -> tuple[str, ...]:
    return tuple(b for b in alphabet if b != ref)


def total_triple_variants(n: int) -> int:
    """``C(n, 3) * 27`` -- the full triple-substitution count for an exon of length n."""
    return math.comb(n, 3) * 27


def _derive_rng_seed(seed: int, exon_id: str) -> int:
    """Deterministic per-(seed, exon) integer seed.

    Deliberately uses SHA-256 rather than Python's built-in ``hash()``: string
    hashing is salted per-process (``PYTHONHASHSEED``) unless explicitly
    disabled, which would silently break reproducibility of the sample across
    runs/machines.
    """
    digest = hashlib.sha256(f"{seed}:{exon_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def sample_triple_keys(
    exon_seq: str, sample_size: int, seed: int, exon_id: str
) -> list[TripleKey]:
    """Deterministically sample ``sample_size`` unique triple-mutant keys.

    Each key is ``(i, j, k, alt_i, alt_j, alt_k)`` with ``0 <= i < j < k <
    len(exon_seq)`` and each ``alt_*`` drawn from the 3 non-reference bases at
    that position. Sampling is by rejection: draw 3 distinct positions (via
    ``random.Random.sample``, so positions are always distinct) and 3 alt
    bases, and keep going until ``sample_size`` *unique* keys have been
    collected (duplicate draws are simply absorbed by the ``set`` and
    re-drawn). The RNG stream -- and therefore the resulting set -- is a pure
    function of ``(seed, exon_id)``.

    The returned list is sorted (canonical order), so the global sample does
    not depend on the order keys happened to be drawn in -- only on the set
    of unique keys itself. This is what ``--chunk-index`` slices are then
    disjoint parts of.
    """
    n = len(exon_seq)
    total = total_triple_variants(n)
    if sample_size < 0:
        raise ValueError(f"sample_size must be >= 0, got {sample_size}.")
    if sample_size > total:
        raise ValueError(
            f"sample_size {sample_size} exceeds the full triple-mutant space "
            f"{total} for exon length {n} ({exon_id})."
        )

    rng = random.Random(_derive_rng_seed(seed, exon_id))
    seen: set[TripleKey] = set()
    while len(seen) < sample_size:
        i, j, k = sorted(rng.sample(range(n), 3))
        alt_i = rng.choice(_alts(exon_seq[i]))
        alt_j = rng.choice(_alts(exon_seq[j]))
        alt_k = rng.choice(_alts(exon_seq[k]))
        seen.add((i, j, k, alt_i, alt_j, alt_k))

    return sorted(seen)


def chunk_bounds(total: int, num_chunks: int, chunk_index: int) -> tuple[int, int]:
    """Balanced, deterministic, disjoint ``[start, end)`` slice for one chunk.

    Chunk sizes differ by at most 1; every index in ``range(total)`` belongs to
    exactly one chunk, for any ``num_chunks >= 1``. Identical logic to
    ``run_tp53_double_substitutions.chunk_bounds``, duplicated here so this
    script has no import-time dependency on the double-substitution module.
    """
    if num_chunks < 1:
        raise ValueError(f"num_chunks must be >= 1, got {num_chunks}.")
    if not 0 <= chunk_index < num_chunks:
        raise ValueError(f"chunk_index must be in [0, {num_chunks}), got {chunk_index}.")

    base, rem = divmod(total, num_chunks)
    start = chunk_index * base + min(chunk_index, rem)
    size = base + (1 if chunk_index < rem else 0)
    return start, start + size


def build_variant_records(
    wt: "hybrid.HybridExon", exon_id: str, keys: Sequence[TripleKey]
) -> list[dict[str, Any]]:
    """Materialize variant records for a list of ``(i, j, k, alt_i, alt_j, alt_k)`` keys."""
    seq = wt.exon_seq
    records: list[dict[str, Any]] = []
    for i, j, k, alt_i, alt_j, alt_k in keys:
        ref_i, ref_j, ref_k = seq[i], seq[j], seq[k]
        mutated = (
            seq[:i] + alt_i + seq[i + 1:j] + alt_j + seq[j + 1:k] + alt_k + seq[k + 1:]
        )
        records.append({
            "exon_id": exon_id,
            "is_reference": False,
            "mutation_region": "triple_exon",
            "pos1": i,
            "pos2": j,
            "pos3": k,
            "position1": i + 1,
            "position2": j + 1,
            "position3": k + 1,
            "ref1": ref_i,
            "alt1": alt_i,
            "ref2": ref_j,
            "alt2": alt_j,
            "ref3": ref_k,
            "alt3": alt_k,
            "exon_length": len(seq),
            "exon_seq": mutated,
            "upstream_flank": wt.upstream_flank,
            "downstream_flank": wt.downstream_flank,
        })
    return records


# ==========================================================================
# Scoring (PNAS batched first, then AlphaGenome per-record -- both delegated
# to run_hybrid_pnas_alphagenome.score_hybrid_variants; nothing re-derived)
# ==========================================================================

_SCORE_COLS = (
    "alphagenome_acceptor",
    "alphagenome_donor",
    "alphagenome_mean_prob",
    "alphagenome_mean_logit",
    "alphagenome_prod_logit",
)

OUTPUT_COLUMNS = [
    "exon_id", "is_reference", "mutation_region",
    "pos1", "pos2", "pos3", "position1", "position2", "position3",
    "ref1", "alt1", "ref2", "alt2", "ref3", "alt3",
    "exon_length", "exon_seq", "upstream_flank", "downstream_flank",
    "pnas_pretuner", "pnas_pretuner_delta",
    "alphagenome_acceptor", "alphagenome_acceptor_delta",
    "alphagenome_donor", "alphagenome_donor_delta",
    "alphagenome_mean_prob", "alphagenome_mean_prob_delta",
    "alphagenome_mean_logit", "alphagenome_mean_logit_delta",
    "alphagenome_prod_logit", "alphagenome_prod_logit_delta",
]


def _order_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [c for c in OUTPUT_COLUMNS if c in df.columns]
    rest = [c for c in df.columns if c not in preferred]
    return df[preferred + rest]


def score_reference(
    wt: "hybrid.HybridExon", dna_model: Any, run_alphagenome: bool, pnas_kwargs: dict | None,
) -> pd.Series:
    """Score the unmutated WT exon once, reusing ``score_hybrid_variants``."""
    record = dict(wt.base_record())
    record["is_reference"] = True
    df = hybrid.score_hybrid_variants(
        [record], dna_model,
        run_pnas=True, run_alphagenome=run_alphagenome,
        add_deltas=False, pnas_kwargs=pnas_kwargs,
    )
    return df.iloc[0]


def _reference_output_row(ref_row: pd.Series, exon_id: str, run_alphagenome: bool) -> pd.DataFrame:
    d: dict[str, Any] = {
        "exon_id": exon_id,
        "is_reference": True,
        "mutation_region": "reference",
        "pos1": None, "pos2": None, "pos3": None,
        "position1": None, "position2": None, "position3": None,
        "ref1": None, "alt1": None,
        "ref2": None, "alt2": None,
        "ref3": None, "alt3": None,
        "exon_length": ref_row.get("exon_length"),
        "exon_seq": ref_row.get("exon_seq"),
        "upstream_flank": ref_row.get("upstream_flank"),
        "downstream_flank": ref_row.get("downstream_flank"),
        "pnas_pretuner": ref_row.get("pnas_pretuner"),
        "pnas_pretuner_delta": 0.0,
    }
    if run_alphagenome:
        for col in _SCORE_COLS:
            if col in ref_row.index:
                d[col] = ref_row[col]
                d[f"{col}_delta"] = 0.0
    return pd.DataFrame([d])


def _add_deltas(df: pd.DataFrame, ref_row: pd.Series, run_alphagenome: bool) -> None:
    df["pnas_pretuner_delta"] = df["pnas_pretuner"] - float(ref_row["pnas_pretuner"])
    if run_alphagenome:
        for col in _SCORE_COLS:
            if col in df.columns:
                df[f"{col}_delta"] = df[col] - float(ref_row[col])


# ==========================================================================
# CLI: score one chunk
# ==========================================================================

def _resolve_out_dir(out_dir_arg: Path | None) -> Path:
    return out_dir_arg if out_dir_arg is not None else DEFAULT_OUTPUT_DIR


def run_score(args: argparse.Namespace) -> int:
    wt = hybrid.load_opensplice_exon(args.exon, args.csv)
    n = wt.exon_length
    full_total = total_triple_variants(n)
    if args.sample_size > full_total:
        print(
            f"ERROR: --sample-size {args.sample_size} exceeds the full triple-mutant "
            f"space {full_total} for {args.exon} (length {n}).",
            file=sys.stderr,
        )
        return 1

    out_dir = _resolve_out_dir(args.out_dir)
    out = out_dir / f"{args.exon}_chunk_{args.chunk_index:03d}.csv"
    if out.exists() and not args.force:
        print(f"[skip] {out} already exists (use --force to overwrite)")
        return 0

    dna_model = None
    if args.alphagenome:
        import alphagenome_local_prediction_pipeline_minigene as ag  # noqa: E402

        dna_model = ag.create_model(env_path=args.env_path)

    pnas_kwargs: dict[str, Any] = {"num_threads": args.pnas_threads}
    if args.device:
        pnas_kwargs["device"] = args.device

    ref_row = score_reference(wt, dna_model, args.alphagenome, pnas_kwargs)

    keys = sample_triple_keys(wt.exon_seq, args.sample_size, args.seed, args.exon)
    start, end = chunk_bounds(len(keys), args.num_chunks, args.chunk_index)
    chunk_keys = keys[start:end]

    print(
        f"{args.exon}: sampled {len(keys)}/{full_total} triple mutants (seed={args.seed}); "
        f"chunk {args.chunk_index}/{args.num_chunks} -> {len(chunk_keys)} variants "
        f"(global sample index [{start}, {end}))"
    )

    records = build_variant_records(wt, args.exon, chunk_keys)

    frames: list[pd.DataFrame] = []
    if args.chunk_index == 0:
        frames.append(_reference_output_row(ref_row, args.exon, args.alphagenome))

    if records:
        df = hybrid.score_hybrid_variants(
            records, dna_model,
            run_pnas=True, run_alphagenome=args.alphagenome,
            add_deltas=False, pnas_kwargs=pnas_kwargs,
        )
        _add_deltas(df, ref_row, run_alphagenome=args.alphagenome)
        frames.append(df)

    out_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS)
    out_df = _order_columns(out_df)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"wrote {len(out_df)} rows -> {out}")
    return 0


# ==========================================================================
# CLI: merge + verify
# ==========================================================================

def run_merge(args: argparse.Namespace) -> int:
    wt = hybrid.load_opensplice_exon(args.exon, args.csv)
    n = wt.exon_length
    full_total = total_triple_variants(n)
    if args.sample_size > full_total:
        print(
            f"ERROR: --sample-size {args.sample_size} exceeds the full triple-mutant "
            f"space {full_total} for {args.exon} (length {n}).",
            file=sys.stderr,
        )
        return 1
    expected_total = args.sample_size

    out_dir = _resolve_out_dir(args.out_dir)
    files = sorted(out_dir.glob(f"{args.exon}_chunk_*.csv"))
    if not files:
        print(f"No chunk files found matching {out_dir / (args.exon + '_chunk_*.csv')}", file=sys.stderr)
        return 1

    if args.num_chunks:
        expected_names = {f"{args.exon}_chunk_{i:03d}.csv" for i in range(args.num_chunks)}
        found_names = {f.name for f in files}
        missing_files = sorted(expected_names - found_names)
        if missing_files:
            print(
                f"ERROR: missing {len(missing_files)}/{args.num_chunks} chunk file(s): "
                f"{missing_files[:10]}{' ...' if len(missing_files) > 10 else ''}",
                file=sys.stderr,
            )
            return 1

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    is_ref = df["is_reference"].fillna(False).astype(bool)
    ref_rows = df[is_ref]
    variant_rows = df[~is_ref].copy()

    ok = True

    if len(ref_rows) == 0:
        print("ERROR: no reference row found across chunks (was chunk 0 scored?)", file=sys.stderr)
        ok = False
    elif len(ref_rows) > 1:
        print(f"ERROR: {len(ref_rows)} reference rows found, expected exactly 1 (duplicate?)", file=sys.stderr)
        ok = False

    if len(variant_rows) != expected_total:
        print(
            f"ERROR: {len(variant_rows)} variant rows found, expected {expected_total} "
            f"(sample_size={args.sample_size}, seed={args.seed}).",
            file=sys.stderr,
        )
        ok = False

    for c in ("pos1", "pos2", "pos3"):
        variant_rows[c] = variant_rows[c].astype(int)

    key_cols = ["pos1", "pos2", "pos3", "alt1", "alt2", "alt3"]
    dup_mask = variant_rows.duplicated(subset=key_cols, keep=False)
    if dup_mask.any():
        n_dup = int(dup_mask.sum())
        examples = variant_rows.loc[dup_mask, key_cols].head(5).values.tolist()
        print(f"ERROR: {n_dup} duplicate {tuple(key_cols)} rows found, e.g. {examples}", file=sys.stderr)
        ok = False

    expected_keys = set(sample_triple_keys(wt.exon_seq, args.sample_size, args.seed, args.exon))
    actual_keys = set(
        zip(
            variant_rows["pos1"], variant_rows["pos2"], variant_rows["pos3"],
            variant_rows["alt1"], variant_rows["alt2"], variant_rows["alt3"],
        )
    )
    missing_keys = expected_keys - actual_keys
    extra_keys = actual_keys - expected_keys
    if missing_keys:
        print(
            f"ERROR: {len(missing_keys)} missing {tuple(key_cols)} combinations, "
            f"e.g. {list(missing_keys)[:5]}",
            file=sys.stderr,
        )
        ok = False
    if extra_keys:
        print(
            f"ERROR: {len(extra_keys)} unexpected {tuple(key_cols)} combinations "
            f"(seed/sample-size mismatch with the chunks that were scored?), "
            f"e.g. {list(extra_keys)[:5]}",
            file=sys.stderr,
        )
        ok = False

    if not ok:
        print("Merge validation FAILED; not writing merged CSV.", file=sys.stderr)
        return 1

    merged = pd.concat(
        [ref_rows, variant_rows.sort_values(key_cols)], ignore_index=True
    )
    merged = _order_columns(merged)

    out = out_dir / f"{args.exon}_merged.csv"
    if out.exists() and not args.force:
        print(f"[skip] {out} already exists (use --force to overwrite)")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(
        f"Merge OK: {len(ref_rows)} reference row(s) + {len(variant_rows)} variant rows "
        f"(expected {expected_total}) -> {out}"
    )
    return 0


# ==========================================================================
# CLI
# ==========================================================================

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deterministically sampled triple exonic substitution scoring for TP53_e6 / TP53_e7."
    )
    parser.add_argument("--exon", required=True, choices=_EXON_CHOICES)
    parser.add_argument("--csv", type=Path, default=DEFAULT_OPENSPLICE_CSV)
    parser.add_argument("--sample-size", type=int, default=50_000,
                        help="number of unique triple mutants to sample per exon (default 50000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed; the sample is a pure, reproducible function of (seed, exon)")
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=20)
    parser.add_argument("--alphagenome", action="store_true",
                        help="also run AlphaGenome (needs an API key via --env-path)")
    parser.add_argument("--env-path", default=".env.alphagenome")
    parser.add_argument("--device", default=None, help="PNAS torch device override")
    parser.add_argument("--pnas-threads", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output directory (default outputs/triple_substitutions/)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing (presumed-complete) chunk/merge output")
    parser.add_argument("--merge", action="store_true",
                        help="merge + verify chunk outputs instead of scoring")
    args = parser.parse_args(argv)

    if args.merge:
        return run_merge(args)

    if not 0 <= args.chunk_index < args.num_chunks:
        parser.error(f"--chunk-index must be in [0, {args.num_chunks}), got {args.chunk_index}")

    return run_score(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
