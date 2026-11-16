#!/usr/bin/env python
"""What does AlphaGenome return for SPLICE_JUNCTIONS on our minigene?

Nothing in this repository has ever requested a junction output: both call
sites (cryptic_lib.py:72, synthetic_context_lib.py:267) ask only for
SPLICE_SITES. So the response's type, fields, coordinate convention and
whether it is returned at all for a 16,384-nt padded synthetic sequence are
unknown, and no analysis script should be written against a guess.

This probe makes ONE construct and prints what comes back, introspecting the
object instead of assuming field names. It costs one AlphaGenome call per
requested output type (default: junctions only).

It deliberately reuses the existing machinery unchanged -- b1_lib.build_construct
for the construct, the same center_pad and the same ontology term -- so whatever
it learns transfers directly to a real scorer.

    python upstream_exon/probe_junction_output.py --dry-run   # no call
    python upstream_exon/probe_junction_output.py             # 1 call
    python upstream_exon/probe_junction_output.py --also-usage  # 2 calls
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
for _p in (str(_THIS_DIR), str(ROOT / "analysis_script")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import b1_lib as B  # noqa: E402
import cryptic_lib as C  # noqa: E402

OUT = ROOT / "outputs" / "upstream_exon"
PANEL_CSV = OUT / "b1_variant_panel.csv"
EXONS_CSV = OUT / "b1_pilot_exons.csv"
PRIMARY_EXON = "syn_a5.0_00482"
REPORT = OUT / "cryptic_check" / "junction_probe.txt"


def _default_env() -> Path:
    """The first key file this repo actually has."""
    for cand in (ROOT / ".env.alphagenome", ROOT / ".env"):
        if cand.exists():
            return cand
    return ROOT / ".env"


def describe(obj, name: str, depth: int = 0, lines: list[str] | None = None):
    """Print an object's type, public attributes and any array/frame shapes."""
    lines = [] if lines is None else lines
    pad = "  " * depth
    lines.append(f"{pad}{name}: {type(obj).__module__}.{type(obj).__name__}")

    # Leaf types are described at ANY depth: their shapes are the whole point,
    # and an early depth guard here silently swallowed them on the first probe.
    if isinstance(obj, (np.ndarray,)):
        lines.append(f"{pad}  ndarray shape={obj.shape} dtype={obj.dtype}")
        lines.append(f"{pad}  first values: {obj.flat[:8]}")
        return lines
    if isinstance(obj, pd.DataFrame):
        lines.append(f"{pad}  DataFrame shape={obj.shape}")
        lines.append(f"{pad}  columns: {list(obj.columns)}")
        lines.append(f"{pad}  head:\n{obj.head(8).to_string()}")
        return lines
    if isinstance(obj, (str, int, float, bool, type(None))):
        lines.append(f"{pad}  value: {obj!r}")
        return lines

    if depth >= 3:
        return lines

    attrs = [a for a in dir(obj) if not a.startswith("_")]
    lines.append(f"{pad}  public attrs: {attrs}")
    for a in attrs:
        try:
            v = getattr(obj, a)
        except Exception as exc:
            lines.append(f"{pad}  .{a} -> raised {type(exc).__name__}")
            continue
        if callable(v):
            continue
        describe(v, f".{a}", depth + 1, lines)
    return lines


def dump_junctions(jd, left_pad: int, core_len: int, dest: Path) -> pd.DataFrame:
    """Flatten JunctionData to a CSV in CORE coordinates.

    ``junctions`` holds interval-like objects whose exact class we do not want
    to depend on, so start/end/strand are pulled with getattr and anything
    unexpected is kept as its repr. Subtracting ``left_pad`` puts a junction
    endpoint on the same axis as known_sites(), where B1 is [0, 113) and the
    upstream donor is 115.
    """
    j = np.asarray(jd.junctions)
    vals = np.asarray(jd.values)
    rows = []
    for i, iv in enumerate(j):
        start = getattr(iv, "start", None)
        end = getattr(iv, "end", None)
        rows.append({
            "idx": i,
            "start_padded": start,
            "end_padded": end,
            "start_core": None if start is None else int(start) - left_pad,
            "end_core": None if end is None else int(end) - left_pad,
            "strand": str(getattr(iv, "strand", "")),
            "repr": repr(iv) if start is None else "",
        })
    df = pd.DataFrame(rows)
    # values is (n_junctions, n_tracks); keep every track column explicitly.
    if vals.ndim == 2 and len(vals) == len(df):
        for t in range(vals.shape[1]):
            df[f"value_t{t}"] = vals[:, t]
    elif vals.size == len(df):
        df["value"] = vals.reshape(-1)

    if {"start_core", "end_core"} <= set(df.columns):
        df["in_core"] = (df.start_core.between(0, core_len - 1) |
                         df.end_core.between(0, core_len - 1))
        df["touches_b1"] = (df.start_core.between(0, 112) |
                            df.end_core.between(0, 112))
    df.to_csv(dest, index=False)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1-id", default="native_PNAS")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and validate the construct, make no call")
    ap.add_argument("--also-usage", action="store_true",
                    help="additionally probe SPLICE_SITE_USAGE (one more call)")
    # create_model's own default is "../.env", which assumes cwd=analysis_script.
    # This repo keeps the key in .env.alphagenome at the root and is run from
    # the root, so resolve it here rather than relying on the relative default.
    ap.add_argument("--env-path", default=str(_default_env()))
    args = ap.parse_args()

    panel = pd.read_csv(PANEL_CSV)
    exons = pd.read_csv(EXONS_CSV)
    seqs = dict(zip(panel.b1_id, panel.b1))
    syn = dict(zip(exons.sequence_id, exons.synthetic_70mer))
    if args.b1_id not in seqs:
        sys.exit(f"FATAL: b1_id {args.b1_id!r} not in {PANEL_CSV}")

    b1 = seqs[args.b1_id]
    core, start = B.build_construct(b1, syn[PRIMARY_EXON])
    fails = B.validate_construct(core, start, b1, syn[PRIMARY_EXON],
                                 require_length=B.B1_LEN)
    if fails:
        sys.exit(f"FATAL: structural validation failed: {fails}")

    print(f"construct: {len(core)} nt, exon_start={start}, b1_id={args.b1_id}")
    print(f"known sites: {C.known_sites(len(b1), start, 76)}")
    if args.dry_run:
        print("--dry-run: no AlphaGenome call made")
        return 0

    import alphagenome_local_prediction_pipeline_minigene as ag
    from alphagenome.models import dna_client

    # Show what this build of the client even offers, before asking for it.
    names = [n for n in dir(dna_client.OutputType) if not n.startswith("_")]
    print(f"\nOutputType members: {names}")
    if not hasattr(dna_client.OutputType, "SPLICE_JUNCTIONS"):
        sys.exit("FATAL: this alphagenome build has no SPLICE_JUNCTIONS member; "
                 "nothing further to probe")

    model = ag.create_model(env_path=args.env_path)
    padded, left_pad, _ = ag.center_pad(ag.clean_seq(core))
    print(f"padded to {len(padded)}, left_pad={left_pad}")

    wanted = [("SPLICE_JUNCTIONS", dna_client.OutputType.SPLICE_JUNCTIONS)]
    if args.also_usage and hasattr(dna_client.OutputType, "SPLICE_SITE_USAGE"):
        wanted.append(("SPLICE_SITE_USAGE", dna_client.OutputType.SPLICE_SITE_USAGE))

    all_lines: list[str] = []
    for label, out_type in wanted:
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
        pred = model.predict_sequence(
            sequence=padded,
            organism=dna_client.Organism.HOMO_SAPIENS,
            requested_outputs=[out_type],
            ontology_terms=["CL:0002518"],
        )
        lines = describe(pred, "prediction")
        all_lines.append(f"===== {label} =====")
        all_lines.extend(lines)
        print("\n".join(lines))

        jd = getattr(pred, "splice_junctions", None)
        if label == "SPLICE_JUNCTIONS" and jd is not None:
            csv = REPORT.parent / "junction_probe_junctions.csv"
            df = dump_junctions(jd, left_pad, len(core), csv)
            print(f"\n{len(df)} junctions, {jd.num_tracks} tracks -> {csv}")
            print(f"ontology_terms: {jd.ontology_terms}")
            print(f"metadata columns: {list(jd.metadata.columns)}")
            print(jd.metadata.head(5).to_string())
            if "touches_b1" in df.columns:
                print(f"junctions with an endpoint inside B1 [0,113): "
                      f"{int(df.touches_b1.sum())}")
            cols = [c for c in ("start_core", "end_core", "strand") if c in df]
            vcols = [c for c in df.columns if c.startswith("value")]
            print(df[cols + vcols].head(25).to_string())

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(all_lines) + "\n")
    print(f"\nwrote {REPORT}")
    print(f"calls spent: {len(wanted)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
