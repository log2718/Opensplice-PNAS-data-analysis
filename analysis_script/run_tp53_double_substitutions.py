"""
Experiment driver: pairwise (double) exonic substitutions for WT TP53_e6 and
WT TP53_e7, scored with the same PNAS pretuner + AlphaGenome minigene
definitions as ``run_hybrid_pnas_alphagenome.py``.

Biological goal
----------------
Single-nucleotide saturation may not perturb the PNAS pretuner / SR-balance
strongly enough to move AlphaGenome's read of TP53_e7. This experiment tests
every unordered pair of distinct exonic positions ``i < j``, each combined
with all 3 x 3 = 9 alternate-base combinations, for both TP53_e6 (responsive)
and TP53_e7 (stubborn) -- i.e. full pairwise (2-nt) saturation mutagenesis.

Expected full counts (``C(n, 2) * 9``):

    TP53_e7 (length 110): C(110, 2) * 9 = 53,955
    TP53_e6 (length 113): C(113, 2) * 9 = 56,952

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

Only the *double*-substitution variant generation and chunk/merge/CLI
plumbing are new. The existing single-SNV pipeline in
``run_hybrid_pnas_alphagenome.py`` is untouched.

HPC / chunking
---------------
~111k variants total (both exons), too many for one AlphaGenome run. Variants
are enumerated in a fixed deterministic order (ascending ``i``, ascending
``j``, then the 3x3 alt-base grid in a fixed order) and sliced into
``--num-chunks`` disjoint, contiguous, index-addressable chunks -- no chunk
overlaps and no variant is skipped, independent of ``--num-chunks``.

The single WT/reference row for an exon is emitted **once**, attached only to
chunk 0's output file, to avoid duplicating it per pair/chunk. Every chunk
still computes its own ``*_delta`` columns against that same WT score (scored
once per invocation), so deltas are correct even for chunks that don't carry
the reference row.

Typical Slurm usage
--------------------
    # one array task per chunk, e.g. --array=0-19
    python analysis_script/run_tp53_double_substitutions.py \\
        --exon TP53_e7 --chunk-index $SLURM_ARRAY_TASK_ID --num-chunks 20 \\
        --alphagenome --env-path .env.alphagenome

    # after all chunks finish:
    python analysis_script/run_tp53_double_substitutions.py \\
        --exon TP53_e7 --merge --num-chunks 20
"""

from __future__ import annotations

import argparse
import itertools
import math
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
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "double_substitutions"

_BASES = ("A", "C", "G", "T")
_EXON_CHOICES = ("TP53_e6", "TP53_e7")

__all__ = [
    "total_double_variants",
    "iter_double_variants",
    "chunk_bounds",
    "build_chunk_records",
]


# ==========================================================================
# Deterministic double-substitution enumeration
# ==========================================================================

def _alts(ref: str, alphabet: Sequence[str] = _BASES) -> tuple[str, ...]:
    return tuple(b for b in alphabet if b != ref)


def _pair_list(n: int) -> list[tuple[int, int]]:
    """All ``(i, j)`` with ``0 <= i < j < n``, in ascending lexicographic order."""
    return list(itertools.combinations(range(n), 2))


def total_double_variants(n: int) -> int:
    """``C(n, 2) * 9`` -- the full pairwise-substitution count for an exon of length n."""
    return math.comb(n, 2) * 9


def variant_at_index(
    exon_seq: str, pairs: Sequence[tuple[int, int]], k: int
) -> tuple[int, int, str, str, str, str]:
    """Unrank global variant index ``k`` -> ``(i, j, ref_i, alt_i, ref_j, alt_j)``.

    Ordering: pairs ascending (from ``pairs``), then 3 alt bases at ``i``
    (ascending, excluding ``ref_i``) x 3 alt bases at ``j`` (ascending,
    excluding ``ref_j``), row-major -- i.e. ``k = pair_index * 9 + a * 3 + b``.
    """
    m, rem = divmod(k, 9)
    i, j = pairs[m]
    a, b = divmod(rem, 3)
    ref_i, ref_j = exon_seq[i], exon_seq[j]
    alt_i = _alts(ref_i)[a]
    alt_j = _alts(ref_j)[b]
    return i, j, ref_i, alt_i, ref_j, alt_j


def iter_double_variants(exon_seq: str):
    """Yield every ``(i, j, ref_i, alt_i, ref_j, alt_j)`` in deterministic order."""
    pairs = _pair_list(len(exon_seq))
    for k in range(total_double_variants(len(exon_seq))):
        yield variant_at_index(exon_seq, pairs, k)


def chunk_bounds(total: int, num_chunks: int, chunk_index: int) -> tuple[int, int]:
    """Balanced, deterministic, disjoint ``[start, end)`` slice for one chunk.

    Chunk sizes differ by at most 1; every index in ``range(total)`` belongs to
    exactly one chunk, for any ``num_chunks >= 1``.
    """
    if num_chunks < 1:
        raise ValueError(f"num_chunks must be >= 1, got {num_chunks}.")
    if not 0 <= chunk_index < num_chunks:
        raise ValueError(f"chunk_index must be in [0, {num_chunks}), got {chunk_index}.")

    base, rem = divmod(total, num_chunks)
    start = chunk_index * base + min(chunk_index, rem)
    size = base + (1 if chunk_index < rem else 0)
    return start, start + size


def build_chunk_records(
    wt: "hybrid.HybridExon", exon_id: str, pairs: Sequence[tuple[int, int]], start: int, end: int
) -> list[dict[str, Any]]:
    """Materialize the variant records for global index range ``[start, end)``."""
    seq = wt.exon_seq
    records: list[dict[str, Any]] = []
    for k in range(start, end):
        i, j, ref_i, alt_i, ref_j, alt_j = variant_at_index(seq, pairs, k)
        mutated = seq[:i] + alt_i + seq[i + 1:j] + alt_j + seq[j + 1:]
        records.append({
            "exon_id": exon_id,
            "is_reference": False,
            "mutation_region": "double_exon",
            "pos1": i,
            "pos2": j,
            "position1": i + 1,
            "position2": j + 1,
            "ref1": ref_i,
            "alt1": alt_i,
            "ref2": ref_j,
            "alt2": alt_j,
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
    "pos1", "pos2", "position1", "position2",
    "ref1", "alt1", "ref2", "alt2",
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
        "pos1": None, "pos2": None,
        "position1": None, "position2": None,
        "ref1": None, "alt1": None,
        "ref2": None, "alt2": None,
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

def _resolve_out(out_arg: Path | None, default_name: str) -> Path:
    out = out_arg if out_arg is not None else Path(default_name)
    if not out.is_absolute() and out.parent == Path("."):
        out = DEFAULT_OUTPUT_DIR / out
    return out


def run_score(args: argparse.Namespace) -> int:
    wt = hybrid.load_opensplice_exon(args.exon, args.csv)
    n = wt.exon_length
    total = total_double_variants(n)
    start, end = chunk_bounds(total, args.num_chunks, args.chunk_index)

    out = _resolve_out(args.out, f"{args.exon}_chunk_{args.chunk_index:03d}.csv")
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

    pairs = _pair_list(n)
    records = build_chunk_records(wt, args.exon, pairs, start, end)
    print(
        f"{args.exon}: chunk {args.chunk_index}/{args.num_chunks} -> "
        f"{len(records)} variants (global index [{start}, {end}) of {total})"
    )

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

    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"wrote {len(out_df)} rows -> {out}")
    return 0


# ==========================================================================
# CLI: merge + verify
# ==========================================================================

def run_merge(args: argparse.Namespace) -> int:
    wt = hybrid.load_opensplice_exon(args.exon, args.csv)
    n = wt.exon_length
    expected_total = total_double_variants(n)

    chunks_dir = args.chunks_dir if args.chunks_dir is not None else DEFAULT_OUTPUT_DIR
    files = sorted(chunks_dir.glob(f"{args.exon}_chunk_*.csv"))
    if not files:
        print(f"No chunk files found matching {chunks_dir / (args.exon + '_chunk_*.csv')}", file=sys.stderr)
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
            f"(exon length {n}).",
            file=sys.stderr,
        )
        ok = False

    variant_rows["pos1"] = variant_rows["pos1"].astype(int)
    variant_rows["pos2"] = variant_rows["pos2"].astype(int)

    key_cols = ["pos1", "pos2", "alt1", "alt2"]
    dup_mask = variant_rows.duplicated(subset=key_cols, keep=False)
    if dup_mask.any():
        n_dup = int(dup_mask.sum())
        examples = variant_rows.loc[dup_mask, key_cols].head(5).values.tolist()
        print(f"ERROR: {n_dup} duplicate (pos1,pos2,alt1,alt2) rows found, e.g. {examples}", file=sys.stderr)
        ok = False

    pairs = _pair_list(n)
    expected_keys = {
        (i, j, alt_i, alt_j)
        for i, j, _ref_i, alt_i, _ref_j, alt_j in (
            variant_at_index(wt.exon_seq, pairs, k) for k in range(expected_total)
        )
    }
    actual_keys = set(
        zip(variant_rows["pos1"], variant_rows["pos2"], variant_rows["alt1"], variant_rows["alt2"])
    )
    missing_keys = expected_keys - actual_keys
    extra_keys = actual_keys - expected_keys
    if missing_keys:
        print(
            f"ERROR: {len(missing_keys)} missing (pos1,pos2,alt1,alt2) combinations, "
            f"e.g. {list(missing_keys)[:5]}",
            file=sys.stderr,
        )
        ok = False
    if extra_keys:
        print(
            f"ERROR: {len(extra_keys)} unexpected (pos1,pos2,alt1,alt2) combinations, "
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

    out = _resolve_out(args.out, f"{args.exon}_merged.csv")
    if out.exists() and not args.force:
        print(f"[skip] {out} already exists (use --force to overwrite)")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
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
        description="Pairwise (double) exonic substitution scoring for TP53_e6 / TP53_e7."
    )
    parser.add_argument("--exon", required=True, choices=_EXON_CHOICES)
    parser.add_argument("--csv", type=Path, default=DEFAULT_OPENSPLICE_CSV)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--alphagenome", action="store_true",
                        help="also run AlphaGenome (needs an API key via --env-path)")
    parser.add_argument("--env-path", default="../.env")
    parser.add_argument("--device", default=None, help="PNAS torch device override")
    parser.add_argument("--pnas-threads", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None,
                        help="output CSV path (bare name -> outputs/double_substitutions/)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing (presumed-complete) chunk/merge output")
    parser.add_argument("--merge", action="store_true",
                        help="merge + verify chunk outputs instead of scoring")
    parser.add_argument("--chunks-dir", type=Path, default=None,
                        help="merge mode: directory holding {exon}_chunk_*.csv (default outputs/double_substitutions/)")
    args = parser.parse_args(argv)

    if args.merge:
        return run_merge(args)

    if not 0 <= args.chunk_index < args.num_chunks:
        parser.error(f"--chunk-index must be in [0, {args.num_chunks}), got {args.chunk_index}")

    return run_score(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
