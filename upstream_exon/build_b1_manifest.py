"""
Build and validate the manifest for the upstream-exon (B1) pilot.

Makes NO AlphaGenome calls. Writes only into ``outputs/upstream_exon/`` and
never touches an existing results file.

Every construct is assembled, structurally validated and SHA-256'd here, so the
scoring step can verify each core before spending a call.

Usage::

    python upstream_exon/build_b1_manifest.py
    python upstream_exon/build_b1_manifest.py --validate-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(ROOT / "analysis_script") not in sys.path:
    sys.path.insert(0, str(ROOT / "analysis_script"))

import b1_lib as B  # noqa: E402
import synthetic_context_lib as L  # noqa: E402
import splice_site_partition as SP  # noqa: E402

OUT = ROOT / "outputs" / "upstream_exon"
MANIFEST_CSV = OUT / "b1_pilot_manifest.csv"
PANEL_CSV = OUT / "b1_variant_panel.csv"
EXONS_CSV = OUT / "b1_pilot_exons.csv"
REPORT_TXT = OUT / "b1_pilot_validation_report.txt"


def build_rows() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = B.build_variant_panel()
    exons = B.load_exons()

    rows = []
    for ex in exons:
        for v in panel:
            reps = (B.N_NATIVE_REPLICATES if v["arm"] == "native" else 1)
            core, start = B.build_construct(v["b1"], ex["synthetic_70mer"])
            for rep in range(1, reps + 1):
                rows.append({
                    "row_id": f"{ex['sequence_id']}|{v['b1_id']}|r{rep}",
                    "sequence_id": ex["sequence_id"],
                    "exon_class": ex["exon_class"],
                    "alpha": ex["alpha"],
                    "PNAS_common": ex["PNAS_common"],
                    "b1_id": v["b1_id"],
                    "arm": v["arm"],
                    "replicate": rep,
                    "fixed_length": v["fixed_length"],
                    "b1_len": v["b1_len"],
                    "b1_gc": v["b1_gc"],
                    "b1_sha256": v["b1_sha256"],
                    "exon_start": start,
                    "exon_len": len(ex["full_middle_exon"]),
                    "core_length": len(core),
                    "core_sha256": B.sha(core),
                })
    return pd.DataFrame(rows), pd.DataFrame(panel), pd.DataFrame(exons)


def validate(man: pd.DataFrame, panel: pd.DataFrame,
             exons: pd.DataFrame) -> list[str]:
    fails: list[str] = []
    lines: list[str] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        tag = "PASS" if ok else "FAIL"
        line = f"  [{tag}] {name}" + (f"  --  {detail}" if detail else "")
        print(line)
        lines.append(line)
        if not ok:
            fails.append(name)

    nat = B.native_b1()

    print("\n1. Panel composition")
    chk("1a. panel has 38 variants", len(panel) == 38, f"n={len(panel)}")
    chk("1b. arm counts are 1 native / 1 fas / 16 uniform / 12 compmatched / 8 dinuc",
        panel.arm.value_counts().to_dict() ==
        {"uniform": 16, "compmatched": 12, "dinuc": 8, "native": 1,
         "fas_reference": 1},
        str(panel.arm.value_counts().to_dict()))
    chk("1c. all B1 sequences are unique",
        panel.b1_sha256.nunique() == len(panel),
        f"{panel.b1_sha256.nunique()} distinct")
    fixed = panel[panel.fixed_length]
    chk("1d. every fixed-length variant is exactly 113 nt",
        (fixed.b1_len == B.B1_LEN).all(), f"n={len(fixed)}")
    chk("1e. the FAS reference is flagged NOT fixed-length",
        (~panel[panel.arm == "fas_reference"].fixed_length).all()
        and int(panel[panel.arm == "fas_reference"].b1_len.iloc[0]) == 59,
        f"fas b1_len={int(panel[panel.arm == 'fas_reference'].b1_len.iloc[0])}")
    chk("1f. the native variant is the unmodified PNAS B1",
        panel[panel.arm == "native"].b1_sha256.iloc[0] == B.sha(nat))

    print("\n2. Generator correctness")
    seqs = {r.b1_id: r.b1 for r in panel.itertuples()}
    dn = [seqs[i] for i in seqs if i.startswith("dinuc_")]
    nat_bc, nat_dc = B.base_counts(nat), B.dinuc_counts(nat)
    chk("2a. dinuc shuffles preserve length",
        all(len(s) == len(nat) for s in dn))
    chk("2b. dinuc shuffles preserve base counts exactly",
        all(B.base_counts(s) == nat_bc for s in dn))
    chk("2c. dinuc shuffles preserve dinucleotide counts exactly",
        all(B.dinuc_counts(s) == nat_dc for s in dn))
    chk("2d. dinuc shuffles are not merely the native sequence",
        all(s != nat for s in dn), f"{sum(s != nat for s in dn)}/{len(dn)} differ")
    uni = [seqs[i] for i in seqs if i.startswith("uniform_")]
    gcs = [(s.count("G") + s.count("C")) / len(s) for s in uni]
    chk("2e. uniform arm mean GC is near 0.50",
        0.44 <= sum(gcs) / len(gcs) <= 0.56, f"mean GC={sum(gcs)/len(gcs):.3f}")
    cm = [seqs[i] for i in seqs if i.startswith("compmatched_")]
    nat_gc = (nat.count("G") + nat.count("C")) / len(nat)
    cm_gc = sum((s.count("G") + s.count("C")) / len(s) for s in cm) / len(cm)
    chk("2f. composition-matched arm mean GC tracks native B1",
        abs(cm_gc - nat_gc) <= 0.06,
        f"native GC={nat_gc:.3f}, arm mean GC={cm_gc:.3f}")

    print("\n3. Determinism")
    again = pd.DataFrame(B.build_variant_panel())
    chk("3a. regenerating the panel reproduces every sequence",
        list(again.b1_sha256) == list(panel.b1_sha256))

    print("\n4. Construct structure (every row)")
    bad: list[str] = []
    for r in man.itertuples():
        b1 = seqs[r.b1_id]
        syn = exons.loc[exons.sequence_id == r.sequence_id,
                        "synthetic_70mer"].iloc[0]
        core, start = B.build_construct(b1, syn)
        f = B.validate_construct(
            core, start, b1, syn,
            require_length=B.B1_LEN if r.fixed_length else None)
        if start != r.exon_start or B.sha(core) != r.core_sha256:
            f.append("manifest exon_start/core_sha256 disagrees with rebuild")
        if f:
            bad.append(f"{r.row_id}: {f}")
    chk("4a. all constructs pass every structural assertion",
        not bad, f"{len(bad)} bad" + (f"; first: {bad[0]}" if bad else ""))
    chk("4b. fixed-length constructs are all 1,309 nt",
        set(man[man.fixed_length].core_length) == {1309},
        str(sorted(set(man[man.fixed_length].core_length))))
    chk("4c. the FAS-reference constructs are 1,255 nt (113-59 shorter)",
        set(man[~man.fixed_length].core_length) == {1309 - 113 + 59},
        str(sorted(set(man[~man.fixed_length].core_length))))

    print("\n5. Native construct reproduces the completed 8-block / endpoint work")
    nat_bad = []
    for ex in exons.itertuples():
        core, start = B.build_construct(nat, ex.synthetic_70mer)
        pb = SP.pnas_blocks8()
        ref8, start8 = SP.assemble8("PPPPPPPP", ex.synthetic_70mer, pb, pb)
        ref6, start6 = L.assemble("PPPPPP", ex.full_middle_exon,
                                  L.pnas_blocks(), L.pnas_blocks())
        if not (core == ref8 == ref6 and start == start8 == start6):
            nat_bad.append(ex.sequence_id)
    chk("5a. native-B1 construct is byte-identical to the 8-block PPPPPPPP "
        "and the 6-block PPPPPP construct", not nat_bad, f"{len(nat_bad)} bad")

    import run_alphagenome_pnas_fulltrack_sanity as PS
    man_ref = PS.build_construct(exons.synthetic_70mer.iloc[0])
    pnas_ref = man_ref["padded"][man_ref["left_pad"]:
                                 man_ref["left_pad"] + man_ref["core_length"]]
    core0, _ = B.build_construct(nat, exons.synthetic_70mer.iloc[0])
    chk("5b. and to the independent full-track sanity construct",
        core0 == pnas_ref and len(core0) == 1309, f"len={len(core0)}")

    print("\n6. Call accounting")
    chk("6a. 273 manifest rows", len(man) == 273, f"n={len(man)}")
    chk("6b. 266 distinct constructs (7 deliberate replicates)",
        man.core_sha256.nunique() == 266,
        f"{man.core_sha256.nunique()} distinct")
    chk("6c. row_id is unique", man.row_id.nunique() == len(man))
    dup = man[man.replicate == 2]
    chk("6d. the only duplicated constructs are the 7 native replicates",
        len(dup) == 7 and set(dup.arm) == {"native"}, f"n={len(dup)}")
    chk("6e. every exon has 39 rows (38 variants + 1 native replicate)",
        set(man.groupby("sequence_id").size()) == {39},
        str(sorted(set(man.groupby("sequence_id").size()))))
    chk("6f. 7 exons: 5 plateau + 2 rising controls",
        man.sequence_id.nunique() == 7
        and man.groupby("exon_class").sequence_id.nunique().to_dict() ==
        {"plateau": 5, "rising_control": 2},
        str(man.groupby("exon_class").sequence_id.nunique().to_dict()))

    print("\n7. Nothing existing is touched")
    chk("7a. output namespace is outputs/upstream_exon/",
        OUT.name == "upstream_exon" and OUT.parent.name == "outputs")

    return fails, lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validate-only", action="store_true",
                    help="run every check but write nothing")
    args = ap.parse_args()

    print("Building the B1 pilot manifest")
    print("=" * 72)
    man, panel, exons = build_rows()
    print(f"  {len(man)} rows, {man.core_sha256.nunique()} distinct constructs, "
          f"{len(panel)} B1 variants, {len(exons)} middle exons")

    fails, _ = validate(man, panel, exons)

    print("\n" + "=" * 72)
    if fails:
        print(f"{len(fails)} FAILURES: {fails}")
        return 1
    print("ALL CHECKS PASSED")

    if args.validate_only:
        print("--validate-only: nothing written")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    for path in (MANIFEST_CSV, PANEL_CSV, EXONS_CSV):
        if path.exists():
            print(f"REFUSING to overwrite existing {path}")
            return 1
    man.to_csv(MANIFEST_CSV, index=False)
    panel.to_csv(PANEL_CSV, index=False)
    exons.to_csv(EXONS_CSV, index=False)
    REPORT_TXT.write_text(
        "B1 pilot manifest validation\n" + "=" * 72 + "\n"
        f"rows={len(man)} distinct_constructs={man.core_sha256.nunique()} "
        f"variants={len(panel)} exons={len(exons)}\nALL CHECKS PASSED\n")
    print(f"\nwrote {MANIFEST_CSV}")
    print(f"wrote {PANEL_CSV}")
    print(f"wrote {EXONS_CSV}")
    print(f"wrote {REPORT_TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
