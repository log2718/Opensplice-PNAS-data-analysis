"""
JOB 2b (merge step) -- combine new shard results with reused 6-block results.

Produces the complete 2 arms x 256 contexts x 400 exons = 204,800-row table,
every row carrying whether its AlphaGenome value was newly scored or reused,
and from which file.

Reads the shard CSVs plus the two 6-block result files (read-only) and writes
into ``outputs/synthetic_context_8block/`` only.

Usage::

    python analysis_script/merge_chimera8_results.py
    python analysis_script/merge_chimera8_results.py --allow-partial
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import splice_site_partition as SP  # noqa: E402

OLD = ROOT / "outputs" / "synthetic_context"
NEW = ROOT / "outputs" / "synthetic_context_8block"
CONTEXT_MAP_CSV = NEW / "chimera8_context_map.csv"
SHARD_DIR = NEW / "shards"
RESULTS_CSV = NEW / "chimera8_ag_results.csv"
SUMMARY_CSV = NEW / "chimera8_context_summary.csv"

OLD_ENDPOINT_CSV = OLD / "synthetic_endpoint_ag_results.csv"
OLD_CHIMERA_CSV = OLD / "chimera_ag_results.csv"

VALUE_COLS = ["AG_acceptor", "AG_donor", "AG_mean_logit"]
OUT_COLS = (["sequence_id", "comparison", "context_label_8"]
            + list(SP.BLOCK_ORDER_8)
            + ["n_F_blocks", "alpha", "PNAS_common"]
            + VALUE_COLS
            + ["reused_vs_new", "source_file", "source_label_6", "core_sha256"])


def load_new() -> pd.DataFrame:
    """All shard results, keyed on job_row_id."""
    files = sorted(SHARD_DIR.glob("chimera8_ag_results_shard*.csv"))
    if not files:
        return pd.DataFrame(columns=["row_id", "source_file"] + VALUE_COLS)
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["source_file"] = f.name
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    dup = d.row_id.duplicated().sum()
    if dup:
        raise SystemExit(f"[abort] {dup} duplicate row_id across shards -- "
                         "shards must be disjoint")
    return d.rename(columns={"AG_acceptor_probability": "AG_acceptor",
                             "AG_donor_probability": "AG_donor"})


def load_reused() -> pd.DataFrame:
    """The 6-block rows that the reusable contexts point at (read-only)."""
    frames = []
    if OLD_ENDPOINT_CSV.exists():
        ep = pd.read_csv(OLD_ENDPOINT_CSV)
        ep = ep.rename(columns={"AG_acceptor_probability": "AG_acceptor",
                                "AG_donor_probability": "AG_donor"})
        ep["source_file"] = OLD_ENDPOINT_CSV.name
        ep["lookup"] = ep.context + "|" + ep.sequence_id
        frames.append(ep[["lookup", "source_file"] + VALUE_COLS])
    if OLD_CHIMERA_CSV.exists():
        ch = pd.read_csv(OLD_CHIMERA_CSV)
        ch = ch.rename(columns={"AG_acceptor_probability": "AG_acceptor",
                                "AG_donor_probability": "AG_donor"})
        ch["source_file"] = OLD_CHIMERA_CSV.name
        ch["lookup"] = ch.comparison + "|" + ch.context_label + "|" + ch.sequence_id
        frames.append(ch[["lookup", "source_file"] + VALUE_COLS])
    return pd.concat(frames, ignore_index=True)


def reuse_lookup_key(row) -> str:
    """Rebuild the key a reused cell points at, from its recorded source."""
    if row.source_file == OLD_ENDPOINT_CSV.name:
        # PPPPPPPP -> the PNAS endpoint; FFFFFFFF -> this arm's FAS endpoint
        ctx = "PNAS" if row.context_label_8 == "PPPPPPPP" else row.comparison
        return f"{ctx}|{row.sequence_id}"
    return f"{row.comparison}|{row.source_label_6}|{row.sequence_id}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-partial", action="store_true",
                    help="write the merge even if some new calls are missing")
    args = ap.parse_args(argv)

    cells = pd.read_csv(CONTEXT_MAP_CSV, low_memory=False)
    new = load_new()
    reused_src = load_reused()
    print(f"cells          : {len(cells):,}")
    print(f"new results    : {len(new):,} rows from "
          f"{len(sorted(SHARD_DIR.glob('chimera8_ag_results_shard*.csv')))} shard files")

    # ---- new cells -------------------------------------------------------
    nc = cells[cells.reused_vs_new == "new"].merge(
        new[["row_id"] + VALUE_COLS + ["source_file"]],
        left_on="job_row_id", right_on="row_id", how="left")
    missing = int(nc.AG_mean_logit.isna().sum())

    # ---- reused cells ----------------------------------------------------
    rc = cells[cells.reused_vs_new == "reused"].copy()
    rc["lookup"] = rc.apply(reuse_lookup_key, axis=1)
    rc = rc.drop(columns=["source_file"]).merge(
        reused_src.drop_duplicates("lookup"), on="lookup", how="left")
    missing_reused = int(rc.AG_mean_logit.isna().sum())

    print(f"new cells      : {len(nc):,}  (missing values: {missing:,})")
    print(f"reused cells   : {len(rc):,}  (missing values: {missing_reused:,})")
    if missing_reused:
        raise SystemExit("[abort] a reused cell could not be matched in the "
                         "6-block results; the reuse mapping is wrong")
    if missing and not args.allow_partial:
        print(f"\n[stop] {missing:,} of {len(nc):,} new calls are not yet "
              "scored. Re-run after the shards finish, or pass --allow-partial.")
        return 1

    out = pd.concat([nc, rc], ignore_index=True)
    out["source_label_6"] = out.get("source_label_6", pd.NA)
    out = out[OUT_COLS].sort_values(
        ["comparison", "context_label_8", "sequence_id"]).reset_index(drop=True)
    out.to_csv(RESULTS_CSV, index=False)

    # ---- per-context summary, mean-logit only ----------------------------
    summ = (out.groupby(["comparison", "context_label_8"])
            .agg(n=("AG_mean_logit", "count"),
                 n_F_blocks=("n_F_blocks", "first"),
                 reused_vs_new=("reused_vs_new", "first"),
                 median_AG_mean_logit=("AG_mean_logit", "median"),
                 mean_AG_mean_logit=("AG_mean_logit", "mean"),
                 sd_AG_mean_logit=("AG_mean_logit", "std"),
                 p90_AG_mean_logit=("AG_mean_logit",
                                    lambda s: s.quantile(0.90)))
            .reset_index())
    for i, b in enumerate(SP.BLOCK_ORDER_8):
        summ[b] = summ.context_label_8.str[i]
    summ.to_csv(SUMMARY_CSV, index=False)

    print(f"\nwrote {RESULTS_CSV.relative_to(ROOT)}  ({len(out):,} rows)")
    print(f"wrote {SUMMARY_CSV.relative_to(ROOT)}  ({len(summ):,} contexts)")
    print(f"  contexts per arm: "
          f"{summ.groupby('comparison').context_label_8.nunique().to_dict()}")
    print(f"  rows by origin  : {out.reused_vs_new.value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
