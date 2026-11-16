#!/usr/bin/env python
"""Resolve the SPLICE_SITE_USAGE layout before spending a million calls.

Everything downstream assumes three things that have never been checked in
this repository, because every previous experiment read SPLICE_SITES:

  1. ``pred.splice_site_usage`` exists and, filtered to the positive strand
     with one ontology term, has exactly ONE track -- so the acceptor's usage
     and the donor's usage are two positions on the same curve, not two
     channels.
  2. The usage values are fractions in [0, 1], so the logit is meaningful and
     the eps-clip at 1e-7 is a sensible floor rather than a routine censor.
  3. The geometry derived in ``fullctx_lib`` points at the real sites: the
     designed acceptor and donor should be local maxima of the usage track,
     and should stand well above a random intronic position.

If any of those is wrong, the right time to find out is now, at a cost of a
handful of calls.

The probe also prints the LEGACY ``AG_mean_logit`` from the same response, so
the new logit-average can be put beside the old metric on an identical input
without spending a second call.

    python upstream_exon/probe_splice_site_usage.py --dry-run   # no calls
    python upstream_exon/probe_splice_site_usage.py             # 3 calls
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

import fullctx_lib as F  # noqa: E402

OUT = ROOT / "outputs" / "upstream_exon" / "fullctx_probe"


def describe(obj, name: str, depth: int = 0, max_depth: int = 3) -> None:
    """Print the shape of whatever the API actually returned."""
    pad = "  " * depth
    if isinstance(obj, np.ndarray):
        print(f"{pad}{name}: ndarray shape={obj.shape} dtype={obj.dtype} "
              f"min={obj.min():.6g} max={obj.max():.6g}")
        return
    if isinstance(obj, pd.DataFrame):
        print(f"{pad}{name}: DataFrame shape={obj.shape} "
              f"columns={list(obj.columns)[:12]}")
        return
    if isinstance(obj, (str, int, float, bool, type(None))):
        print(f"{pad}{name}: {type(obj).__name__} = {obj!r}")
        return
    if depth >= max_depth:
        print(f"{pad}{name}: {type(obj).__name__}")
        return
    if isinstance(obj, (list, tuple)):
        print(f"{pad}{name}: {type(obj).__name__} len={len(obj)}")
        for i, v in enumerate(obj[:4]):
            describe(v, f"[{i}]", depth + 1, max_depth)
        return
    print(f"{pad}{name}: {type(obj).__name__}")
    for attr in sorted(a for a in dir(obj) if not a.startswith("_")):
        try:
            val = getattr(obj, attr)
        except Exception:                                   # noqa: BLE001
            continue
        if callable(val):
            continue
        describe(val, attr, depth + 1, max_depth)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="build and validate the constructs, make no call")
    ap.add_argument("--env-path", default=None)
    ap.add_argument("--backend", default="api", choices=list(F.BACKENDS),
                    help="which predictor to probe; the layout this resolves "
                         "is a property of the backend, so probe the one you "
                         "will actually score with")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()

    g = F.GEOMETRY
    left_pad, acc_pos, don_pos = F.padded_positions()
    print("=" * 72)
    print("FULL-CONTEXT GEOMETRY")
    print("=" * 72)
    print(f"  context length      {F.CONTEXT_LEN:,}  -> padded to "
          f"{F.TARGET_LEN:,} (left_pad {left_pad:,})")
    print(f"  B1 slot             [{g['b1_start']}, {g['b1_end']})  "
          f"{F.B1_LEN} nt")
    print(f"  middle exon body    [{g['syn_start']}, {g['syn_end']})  "
          f"{F.SYN_LEN} nt")
    print(f"  acceptor            {g['acceptor']}  (padded {acc_pos:,})")
    print(f"  donor               {g['donor']}  (padded {don_pos:,})")
    print(f"  barcode (left as N) [{g['barcode_start']}, {g['barcode_end']})")

    exons = pd.read_csv(ROOT / "outputs" / "upstream_exon" / "b1_pilot_exons.csv")
    syn = dict(zip(exons.sequence_id, exons.synthetic_70mer))[F.PRIMARY_EXON]
    import splice_site_partition as SP
    native_b1 = SP.pnas_blocks8()["B1"]

    cases = {
        "native": (native_b1, syn),
        "permuted_b1": ("".join(np.random.default_rng(0).permutation(
            list("A" * 29 + "C" * 28 + "G" * 28 + "T" * 28))), syn),
        "permuted_exon": (native_b1, "".join(np.random.default_rng(1).permutation(
            list("A" * 18 + "T" * 18 + "C" * 17 + "G" * 17)))),
    }
    built = {}
    for name, (b1, s70) in cases.items():
        seq = F.build_construct(b1, s70)
        fails = F.validate_construct(seq, b1, s70)
        if fails:
            sys.exit(f"FATAL {name}: {fails}")
        built[name] = seq
    print(f"\n  {len(built)} constructs rebuilt and structurally validated")

    if args.dry_run:
        print("\n--dry-run: no AlphaGenome call made")
        return 0

    import alphagenome_local_prediction_pipeline_minigene as ag
    from alphagenome.models import dna_client

    env = args.env_path
    if env is None:
        for cand in (ROOT / ".env.alphagenome", ROOT / ".env"):
            if cand.exists():
                env = str(cand)
                break
    model = F.make_model(args.backend, env_path=env)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, seq in built.items():
        padded, lp, _ = ag.center_pad(ag.clean_seq(seq), target_len=F.TARGET_LEN)
        pred = model.predict_sequence(
            sequence=padded,
            organism=dna_client.Organism.HOMO_SAPIENS,
            requested_outputs=[dna_client.OutputType.SPLICE_SITE_USAGE,
                               dna_client.OutputType.SPLICE_SITES],
            ontology_terms=list(F.ONTOLOGY),
        )

        if name == "native":
            print("\n" + "=" * 72)
            print("RAW splice_site_usage STRUCTURE")
            print("=" * 72)
            describe(pred.splice_site_usage, "splice_site_usage")

        track = F.get_usage_track(pred)
        u_acc, u_don = float(track[acc_pos]), float(track[don_pos])
        d_t, a_t = ag.get_positive_strand_tracks(pred)
        p_acc, p_don = float(a_t[acc_pos]), float(d_t[don_pos])

        # Is the designed site actually where the usage peaks?
        win = 30
        a_lo, a_hi = acc_pos - win, acc_pos + win + 1
        d_lo, d_hi = don_pos - win, don_pos + win + 1
        a_arg = int(np.argmax(track[a_lo:a_hi])) + a_lo - acc_pos
        d_arg = int(np.argmax(track[d_lo:d_hi])) + d_lo - don_pos

        rows.append({
            "case": name,
            "usage_acceptor": u_acc, "usage_donor": u_don,
            "logit_average": F.logit_average(u_acc, u_don),
            "prob_acceptor": p_acc, "prob_donor": p_don,
            "legacy_mean_logit": (F.L._logit(p_acc) + F.L._logit(p_don)) / 2,
            "argmax_offset_acceptor": a_arg, "argmax_offset_donor": d_arg,
            "track_min": float(track.min()), "track_max": float(track.max()),
            "track_mean": float(track.mean()),
            "n_above_0.01": int((track > 0.01).sum()),
        })

    d = pd.DataFrame(rows)
    dest = args.out_dir / "usage_probe.csv"
    d.to_csv(dest, index=False)

    print("\n" + "=" * 72)
    print("READOUT")
    print("=" * 72)
    with pd.option_context("display.width", 200,
                           "display.max_columns", 40,
                           "display.float_format", lambda v: f"{v:.6g}"):
        print(d.to_string(index=False))

    print("\n" + "=" * 72)
    print("CHECKS")
    print("=" * 72)
    ok = True

    rng = (d.track_min.min(), d.track_max.max())
    in01 = rng[0] >= -1e-9 and rng[1] <= 1 + 1e-9
    print(f"  [{'PASS' if in01 else 'FAIL'}] usage values lie in [0, 1]  "
          f"(observed {rng[0]:.6g} .. {rng[1]:.6g})")
    ok &= in01

    peaked = bool((d.argmax_offset_acceptor == 0).all()
                  and (d.argmax_offset_donor == 0).all())
    print(f"  [{'PASS' if peaked else 'WARN'}] designed sites are the local "
          f"maxima within +-30 nt  "
          f"(acceptor offsets {list(d.argmax_offset_acceptor)}, "
          f"donor offsets {list(d.argmax_offset_donor)})")

    floored = int(((d.usage_acceptor <= 1e-7) | (d.usage_donor <= 1e-7)).sum())
    print(f"  [{'PASS' if not floored else 'WARN'}] no usage is at or below "
          f"the 1e-7 logit floor  ({floored} of {len(d)} cases affected)")

    spread = d.logit_average.max() - d.logit_average.min()
    print(f"  [INFO] logit-average spans {spread:.4f} across "
          f"{len(d)} very different inputs")
    print(f"  [INFO] legacy AG_mean_logit on the same calls: "
          f"{list(d.legacy_mean_logit.round(4))}")
    print(f"\nwrote {dest}")

    if not ok:
        print("\nAt least one hard check FAILED. Do not start the mass "
              "scoring run until the layout is understood.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
