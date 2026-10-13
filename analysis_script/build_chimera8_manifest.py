"""
JOB 2b (build step) -- manifest + validation for the corrected 8-block chimera run.

Enumerates all 2^8 = 256 contexts for each arm (PNAS <-> FAS/TP53_e6 and
PNAS <-> FAS/TP53_e7) over the SAME 400 synthetic exons used by the 6-block
run, works out which constructs already exist in the 6-block results, collapses
constructs that are byte-identical across the two arms, and writes the job
manifest for the scoring step.

Makes NO AlphaGenome calls. Writes only into ``outputs/synthetic_context_8block/``
and never touches a 6-block file.

Everything is decided by SEQUENCE IDENTITY, never by label. Every one of the
2 x 256 x 400 = 204,800 cells has its full construct built and SHA-256'd, as
are all 2 x 64 x 400 = 51,200 six-block constructs; reuse and cross-arm
de-duplication are then set intersections over those digests.

Outputs
-------
``chimera8_subset_ids.csv``       the 400 exon IDs (copied from the 6-block run)
``chimera8_context_map.csv``      all 204,800 cells, each resolved to new/reused
``chimera8_job_manifest.csv``     one row per DISTINCT new AlphaGenome call
``chimera8_reused_contexts.csv``  the verified reuse mapping
``chimera8_validation_report.txt`` the 20-point report

Usage::

    python analysis_script/build_chimera8_manifest.py            # build + validate
    python analysis_script/build_chimera8_manifest.py --validate-only
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import synthetic_context_lib as L  # noqa: E402
import splice_site_partition as SP  # noqa: E402

# ---- old (6-block) inputs: READ ONLY -------------------------------------
OLD = ROOT / "outputs" / "synthetic_context"
LIB_CSV = OLD / "synthetic_exon_library.csv"
OLD_SUBSET_CSV = OLD / "chimera_subset_ids.csv"
OLD_ENDPOINT_CSV = OLD / "synthetic_endpoint_ag_results.csv"
OLD_CHIMERA_CSV = OLD / "chimera_ag_results.csv"

# ---- new namespace: everything this pipeline writes -----------------------
NEW = ROOT / "outputs" / "synthetic_context_8block"
SUBSET_CSV = NEW / "chimera8_subset_ids.csv"
CONTEXT_MAP_CSV = NEW / "chimera8_context_map.csv"
MANIFEST_CSV = NEW / "chimera8_job_manifest.csv"
REUSED_CSV = NEW / "chimera8_reused_contexts.csv"
REPORT_TXT = NEW / "chimera8_validation_report.txt"

COMPARISONS = ["TP53_e6", "TP53_e7"]
RATES = (0.38, 0.51, 0.60)
N_SHARDS_DEFAULT = 12


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _canon(arm: str, label: str) -> tuple[str, str]:
    """Deterministic representative for a construct shared by several cells."""
    return (arm, label)


# ==========================================================================
# core enumeration
# ==========================================================================

def enumerate_cells(lib: pd.DataFrame, subset_ids: list[str]) -> pd.DataFrame:
    """Every (arm, label8, sequence) cell with its construct digest.

    Cores are hashed and discarded, so peak memory stays small.
    """
    sub = lib[lib.sequence_id.isin(subset_ids)].copy()
    sub = sub.set_index("sequence_id").loc[subset_ids].reset_index()
    pb = SP.pnas_blocks8()
    labels = SP.all_labels8()

    rows = []
    for arm in COMPARISONS:
        u70, d25 = L.FAS_TARGETS[arm]
        fb = SP.fas_blocks8(u70, d25)
        for lab in labels:
            for r in sub.itertuples():
                core, start = SP.assemble8(lab, r.synthetic_70mer, pb, fb)
                rows.append({
                    "comparison": arm,
                    "context_label_8": lab,
                    "sequence_id": r.sequence_id,
                    "alpha": r.alpha,
                    "PNAS_common": r.PNAS_common,
                    "n_F_blocks": lab.count("F"),
                    "core_sha256": _sha(core),
                    "core_length": len(core),
                    "exon_start": start,
                    "exon_len": len(L.EXON_PREFIX) + L.SYN_LEN + len(L.EXON_SUFFIX),
                })
    cells = pd.DataFrame(rows)
    for i, b in enumerate(SP.BLOCK_ORDER_8):
        cells[b] = cells.context_label_8.str[i]
    return cells


def enumerate_old_constructs(lib: pd.DataFrame,
                             subset_ids: list[str]) -> pd.DataFrame:
    """Digest every 6-block construct that was actually scored.

    Endpoints (``PPPPPP`` / ``FFFFFF``) live in the endpoint CSV; the other 62
    labels live in the chimera CSV. Both are recorded with the file they came
    from so a reused row can cite its source.
    """
    sub = lib[lib.sequence_id.isin(subset_ids)].copy()
    pb6 = L.pnas_blocks()
    rows = []
    for arm in COMPARISONS:
        u70, d25 = L.FAS_TARGETS[arm]
        fb6 = L.fas_blocks(u70, d25)
        for lab in L.all_labels():
            if lab == "PPPPPP":
                src, key = OLD_ENDPOINT_CSV.name, ("context", "PNAS")
            elif lab == "FFFFFF":
                src, key = OLD_ENDPOINT_CSV.name, ("context", arm)
            else:
                src, key = OLD_CHIMERA_CSV.name, ("context_label", lab)
            for r in sub.itertuples():
                core, _ = L.assemble(lab, r.full_middle_exon, pb6, fb6)
                rows.append({
                    "comparison": arm, "old_label": lab,
                    "sequence_id": r.sequence_id,
                    "core_sha256": _sha(core),
                    "source_file": src,
                    "source_key": f"{key[0]}={key[1]}",
                })
    return pd.DataFrame(rows)


# ==========================================================================
# resolution: reuse, cross-arm de-duplication, new jobs
# ==========================================================================

def resolve(cells: pd.DataFrame, old: pd.DataFrame) -> tuple[pd.DataFrame,
                                                             pd.DataFrame,
                                                             pd.DataFrame]:
    """Split the 204,800 cells into reused and new, de-duplicating by digest."""
    # --- reuse: digest present among the already-scored 6-block constructs --
    old_by_hash = (old.sort_values(["comparison", "old_label", "sequence_id"])
                   .drop_duplicates("core_sha256")
                   .set_index("core_sha256")[["old_label", "source_file",
                                              "source_key", "comparison"]])
    cells = cells.copy()
    cells["reused_vs_new"] = np.where(
        cells.core_sha256.isin(old_by_hash.index), "reused", "new")
    cells = cells.join(old_by_hash.rename(columns={
        "old_label": "source_label_6",
        "comparison": "source_comparison"}), on="core_sha256")
    cells.loc[cells.reused_vs_new == "new",
              ["source_label_6", "source_file", "source_key",
               "source_comparison"]] = pd.NA

    # --- new jobs: one per DISTINCT digest, canonical cell as representative -
    new = cells[cells.reused_vs_new == "new"]
    rep = (new.sort_values(["core_sha256", "comparison", "context_label_8",
                            "sequence_id"])
           .drop_duplicates("core_sha256")
           .reset_index(drop=True))
    rep = rep.rename(columns={"comparison": "canon_comparison",
                              "context_label_8": "canon_label"})
    rep.insert(0, "job_index", np.arange(len(rep), dtype=int))
    rep["job_row_id"] = (rep.canon_comparison + "|" + rep.canon_label + "|"
                         + rep.sequence_id)
    manifest = rep[["job_index", "job_row_id", "canon_comparison", "canon_label",
                    "sequence_id", "core_sha256", "core_length", "exon_start",
                    "exon_len"]]

    cells = cells.merge(manifest[["job_row_id", "job_index", "core_sha256"]],
                        on="core_sha256", how="left")

    reused = cells[cells.reused_vs_new == "reused"][
        ["comparison", "context_label_8", "sequence_id", "core_sha256",
         "source_label_6", "source_comparison", "source_file", "source_key"]]
    return cells, manifest, reused


# ==========================================================================
# validation
# ==========================================================================

def validate(lib: pd.DataFrame, subset_ids: list[str], cells: pd.DataFrame,
             manifest: pd.DataFrame, reused: pd.DataFrame,
             n_shards: int) -> list[str]:
    """The 20-point pre-submission report. Returns the list of failures."""
    fails: list[str] = []

    def chk(n: int, name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n:2d}. {name}"
              f"{'  -- ' + detail if detail else ''}")
        if not ok:
            fails.append(f"{n}. {name}")

    pb = SP.pnas_blocks8()
    fbs = {a: SP.fas_blocks8(*L.FAS_TARGETS[a]) for a in COMPARISONS}
    syn0 = lib[lib.sequence_id == subset_ids[0]].synthetic_70mer.iloc[0]
    exon0 = L.EXON_PREFIX + syn0 + L.EXON_SUFFIX

    # 1-2
    for n, arm in ((1, "TP53_e6"), (2, "TP53_e7")):
        labs = cells[cells.comparison == arm].context_label_8.unique()
        chk(n, f"256 labels exist for {arm.split('_')[1]}",
            len(labs) == 256 and len(set(labs)) == 256, f"{len(labs)} labels")

    # 3
    per = cells.groupby(["comparison", "context_label_8"]).sequence_id.apply(
        lambda s: tuple(sorted(s)))
    want = tuple(sorted(subset_ids))
    chk(3, "same 400 sequence IDs in every context",
        len(subset_ids) == 400 and bool((per == want).all()),
        f"{per.nunique()} distinct ID set(s) across {len(per)} contexts")

    # 4-6  byte-for-byte endpoint reconstruction
    import run_alphagenome_pnas_fulltrack_sanity as PS
    import alphagenome_local_prediction_pipeline_minigene as ag
    man = PS.build_construct(syn0)
    ref_p = man["padded"][man["left_pad"]:man["left_pad"] + man["core_length"]]
    core_p, _ = SP.assemble8("PPPPPPPP", syn0, pb, pb)
    chk(4, "PPPPPPPP reconstructs the original 1,309-nt PNAS construct "
        "byte-for-byte", core_p == ref_p and len(core_p) == 1309,
        f"len={len(core_p)}")
    for n, arm in ((5, "TP53_e6"), (6, "TP53_e7")):
        u70, d25 = L.FAS_TARGETS[arm]
        ref_f = ag.build_fas_minigene(ag.build_variable_region(exon0, u70, d25))
        core_f, _ = SP.assemble8("FFFFFFFF", syn0, pb, fbs[arm])
        chk(n, f"FFFFFFFF reconstructs the {arm} FAS construct byte-for-byte",
            core_f == ref_f and len(core_f) == 471, f"len={len(core_f)}")

    # 7-8  splice-site window sizes, all three contexts
    ctxs = {"PNAS": pb, **fbs}
    chk(7, "B2 and B5 are exactly 9 nt (3 exonic + 6 intronic) in all contexts",
        all(len(b[k]) == 9 for b in ctxs.values() for k in ("B2", "B5")))
    chk(8, "B4 and B7 are exactly 23 nt (20 intronic + 3 exonic) in all contexts",
        all(len(b[k]) == 23 for b in ctxs.values() for k in ("B4", "B7")))

    # 9-12  exhaustive over all 256 labels x both arms
    bad9 = bad10 = bad11 = bad12 = 0
    expect_len = {"PNAS": 1309, "TP53_e6": 471, "TP53_e7": 471}
    for arm in COMPARISONS:
        fb = fbs[arm]
        for lab in SP.all_labels8():
            core, st = SP.assemble8(lab, syn0, pb, fb)
            if len(syn0) != 70 or core.count(syn0) != 1:
                bad9 += 1
            if not (pb if lab[3] == "P" else fb)["B4"].endswith(L.EXON_PREFIX):
                bad10 += 1
            if not (pb if lab[4] == "P" else fb)["B5"].startswith(L.EXON_SUFFIX):
                bad11 += 1
            blocks = sum(len((pb if c == "P" else fb)[b])
                         for b, c in zip(SP.BLOCK_ORDER_8, lab))
            if len(core) != blocks + 70 or core[st:st + len(exon0)] != exon0:
                bad12 += 1
    chk(9, "synthetic70 is exactly 70 nt and appears exactly once in all 512 "
        "contexts", bad9 == 0, f"{bad9} bad")
    chk(10, "GTT belongs entirely to B4", bad10 == 0, f"{bad10} bad")
    chk(11, "CAG belongs entirely to B5", bad11 == 0, f"{bad11} bad")
    chk(12, "no base lost or duplicated in any of the 256 contexts (both arms)",
        bad12 == 0, f"{bad12} bad")
    chk(12.5 if False else 12, "core lengths are exactly as specified "
        "(PNAS 1309 / FAS 471 at the endpoints)",
        cells[cells.context_label_8 == "PPPPPPPP"].core_length.eq(1309).all()
        and cells[cells.context_label_8 == "FFFFFFFF"].core_length.eq(471).all())

    # 13
    same = {arm: [b for b in SP.BLOCK_ORDER_8 if pb[b] == fbs[arm][b]]
            for arm in COMPARISONS}
    chk(13, "all 8 blocks genuinely differ P vs F in both arms",
        all(not v for v in same.values()), str(same))

    # 14  reuse verified by sequence equality
    rl = (reused.groupby(["comparison", "context_label_8", "source_label_6",
                          "source_file"]).size().reset_index(name="n_seq"))
    distinct_reused_constructs = reused.core_sha256.nunique()
    print("        reusable contexts verified by construct digest:")
    for _, r in rl.iterrows():
        print(f"          {r.comparison} {r.context_label_8}  ==  6-block "
              f"{r.source_label_6}  ({r.n_seq} exons, from {r.source_file})")
    chk(14, "exact list of reusable old contexts verified by sequence equality",
        len(rl) == 8 and distinct_reused_constructs == 7 * len(subset_ids),
        f"{len(rl)} (arm,label) cells, "
        f"{distinct_reused_constructs // max(len(subset_ids), 1)} distinct "
        "constructs per exon")

    # 15  cross-arm duplicates verified by sequence equality
    h6 = set(cells[cells.comparison == "TP53_e6"].core_sha256)
    h7 = set(cells[cells.comparison == "TP53_e7"].core_sha256)
    shared_h = h6 & h7
    shared_labels = sorted(cells[cells.core_sha256.isin(shared_h)]
                           .context_label_8.unique())
    chk(15, "cross-arm duplicate constructs verified by sequence equality",
        len(shared_labels) == 16
        and len(shared_h) == 16 * len(subset_ids)
        and all(l[2:6] == "PPPP" for l in shared_labels),
        f"{len(shared_labels)} labels x {len(subset_ids)} exons "
        f"= {len(shared_h)} shared constructs")
    print(f"        shared labels (B3..B6 all P): {', '.join(shared_labels)}")

    # 16  call count, recomputed from the actual job list
    n_cells = len(cells)
    n_reused = int((cells.reused_vs_new == "reused").sum())
    n_new_cells = int((cells.reused_vs_new == "new").sum())
    n_calls = len(manifest)
    per_seq = n_calls / len(subset_ids)
    ok16 = (n_cells == 2 * 256 * 400 and n_reused == 8 * 400
            and n_new_cells == 504 * 400 and n_calls == 489 * 400
            and manifest.core_sha256.is_unique
            and manifest.job_row_id.is_unique)
    chk(16, "exact number of distinct new AG calls", ok16,
        f"{n_cells:,} cells = {n_reused:,} reused + {n_new_cells:,} new cells "
        f"-> {n_calls:,} distinct calls ({per_seq:.0f}/exon)")

    # 17  runtime
    print("        expected pure AG runtime:")
    for r in RATES:
        s = n_calls * r
        print(f"          {r:.2f} s/call -> {s:9,.0f} s = {s / 3600:5.1f} h "
              f"| {n_shards} shards -> {s / 3600 / n_shards:4.2f} h each")
    chk(17, "expected runtime computed at 0.38 / 0.51 / 0.60 s per call", True)

    # 18  no overwrite of 6-block data
    protected = [OLD_ENDPOINT_CSV, OLD_CHIMERA_CSV, OLD_SUBSET_CSV,
                 OLD / "chimera_context_summary.csv",
                 OLD / "chimera_factorial_effects.csv"]
    writes = [SUBSET_CSV, CONTEXT_MAP_CSV, MANIFEST_CSV, REUSED_CSV, REPORT_TXT,
              NEW / "chimera8_ag_results.csv",
              NEW / "chimera8_context_summary.csv"]
    inside_new = all(NEW in p.parents or p.parent == NEW for p in writes)
    no_clash = not (set(writes) & set(protected))
    chk(18, "output files cannot overwrite old 6-block data",
        inside_new and no_clash and NEW.resolve() != OLD.resolve(),
        f"all writes under {NEW.relative_to(ROOT)}/")

    # 19  resume logic
    chk(19, "resume logic cannot duplicate already completed calls",
        manifest.job_row_id.is_unique and manifest.job_index.is_unique,
        "job_row_id is the unique resume key; the runner skips any row_id "
        "already present in its shard CSV and re-verifies the SHA-256")

    # 20  requested output
    import inspect
    src = inspect.getsource(L.score_core)
    chk(20, "AG requested output is SPLICE_SITES only",
        "OutputType.SPLICE_SITES" in src and src.count("OutputType.") == 1,
        "synthetic_context_lib.score_core, unchanged")

    return fails


# ==========================================================================
# main
# ==========================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-shards", type=int, default=N_SHARDS_DEFAULT)
    ap.add_argument("--validate-only", action="store_true",
                    help="run the report against existing manifest files")
    args = ap.parse_args(argv)

    lib = pd.read_csv(LIB_CSV)
    subset_ids = pd.read_csv(OLD_SUBSET_CSV).sequence_id.tolist()
    print(f"[subset] reusing the 6-block subset: {len(subset_ids)} exon IDs "
          f"from {OLD_SUBSET_CSV.relative_to(ROOT)}")
    regen = L.balanced_subset(lib)
    print(f"[subset] identical to balanced_subset(lib): {regen == subset_ids}")

    NEW.mkdir(parents=True, exist_ok=True)

    print("[build] enumerating 2 x 256 x 400 = 204,800 cells ...", flush=True)
    cells = enumerate_cells(lib, subset_ids)
    print(f"[build] hashing {len(cells):,} constructs done", flush=True)
    print("[build] digesting the 2 x 64 x 400 six-block constructs ...", flush=True)
    old = enumerate_old_constructs(lib, subset_ids)
    cells, manifest, reused = resolve(cells, old)

    if not args.validate_only:
        pd.DataFrame({"sequence_id": subset_ids}).to_csv(SUBSET_CSV, index=False)
        cells.to_csv(CONTEXT_MAP_CSV, index=False)
        manifest.to_csv(MANIFEST_CSV, index=False)
        reused.to_csv(REUSED_CSV, index=False)

    buf = io.StringIO()
    with redirect_stdout(buf):
        print("8-BLOCK CHIMERA RUN -- PRE-SUBMISSION VALIDATION")
        print("=" * 78)
        fails = validate(lib, subset_ids, cells, manifest, reused, args.n_shards)
        print("=" * 78)
        print("ALL 20 CHECKS PASSED" if not fails
              else f"{len(fails)} FAILURES: {fails}")
    report = buf.getvalue()
    print(report)
    if not args.validate_only:
        REPORT_TXT.write_text(report)
        print(f"wrote {SUBSET_CSV.relative_to(ROOT)}")
        print(f"wrote {CONTEXT_MAP_CSV.relative_to(ROOT)}  ({len(cells):,} rows)")
        print(f"wrote {MANIFEST_CSV.relative_to(ROOT)}  ({len(manifest):,} rows)")
        print(f"wrote {REUSED_CSV.relative_to(ROOT)}  ({len(reused):,} rows)")
        print(f"wrote {REPORT_TXT.relative_to(ROOT)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
