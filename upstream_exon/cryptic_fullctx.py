#!/usr/bin/env python
"""Cryptic-splice-site scan of a full-context library, on local weights.

v3 asked this question in the 1,309-nt reconstructed cassette, over the API,
on 3,000 sequences: does permuting B1 create a cryptic splice site inside B1,
and is the measured effect on the middle exon just that artifact? The answer
there was no -- 32 % of sequences created a cryptic donor in B1, but dropping
every one of them retained 101 % of the target's variance.

The full 8,612-nt context changes the premise enough to be worth redoing. It
adds ~7,300 nt of real plasmid around the cassette, which can either host
competing sites of its own or suppress cryptic usage by giving the model a
realistic transcript to work with. Either outcome is informative, and
"unchanged" is the result that lets the v3 conclusion carry over.

Design notes:

* Pass 1 runs ``splice_sites`` + ``splice_site_usage`` together: 327 KB per
  sequence, and both the cryptic features and the target come from the SAME
  forward call, so they can never disagree about which sequence they
  describe. Three kinds of evidence come out of it -- a motif score, the
  usage the model assigns at that motif, and the designed sites' usage.
* Pass 2 runs ``splice_junctions``, which is the only head that speaks about
  COMPETITION: whether a cryptic site actually diverts flux from the
  designed pair, rather than merely existing. Its notorious 779 MB/sequence
  is not intrinsic -- that is the cost of its default top-512 candidate
  search. Supplying ``splice_site_positions`` (model.py:777) skips that and
  makes pred_counts (B, P, P, 734); at P = 8 it is 188 KB. The candidates
  are chosen from pass 1's output, which is why this cannot be folded into
  one call: the price is a second trunk evaluation, not a second job.
  Disable with --no-junctions.
* The sampled ``row_id``s come from the library manifest, so every row here
  joins straight onto ``ag_results.csv`` by ``row_id``. This is a VIEW of the
  main experiment, not a separate one.
* Channel indices are resolved from the track catalogue by name, never
  assumed -- the same discipline as the CL:0002518 usage track.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
for _p in (str(_THIS_DIR), str(ROOT / "analysis_script")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fullctx_lib as F          # noqa: E402
import splice_site_partition as SP  # noqa: E402
from build_fullctx_library import LIBRARIES  # noqa: E402

OUT = ROOT / "outputs" / "upstream_exon"

#: probability thresholds the per-window counts are reported at
THRESHOLDS = (0.01, 0.1, 0.5)

#: non-designed candidate sites handed to the junction head, per role
N_CANDIDATES = 7
#: minimum separation between candidates, nt (crude non-max suppression, so
#: the 7 picks are 7 distinct sites and not one peak's shoulders)
NMS_WINDOW = 20


# ==========================================================================
# channel resolution
# ==========================================================================

def resolve_splice_site_channels(catalog_path: str | Path) -> dict[str, int]:
    """Map ``donor_plus`` / ``acceptor_plus`` (and minus) to channel indices.

    The head is 5-wide and widely described as (donor+, acceptor+, donor-,
    acceptor-, none), but this reads the catalogue instead of trusting that,
    and raises with the actual names if they do not resolve.
    """
    df = pd.read_parquet(catalog_path)
    ss = df[df.output_type == "splice_sites"].reset_index(drop=True)
    if len(ss) != 5:
        raise ValueError(
            f"catalogue has {len(ss)} splice_sites rows, expected 5:\n"
            f"{ss[['track_index', 'track_name', 'strand']] if len(ss) else ''}")

    out: dict[str, int] = {}
    for r in ss.itertuples():
        name = str(r.track_name).lower()
        strand = str(getattr(r, "strand", "")).strip()
        kind = ("donor" if "donor" in name else
                "acceptor" if "acceptor" in name else None)
        if kind is None:
            continue
        side = "plus" if strand == "+" else "minus" if strand == "-" else None
        if side is None:                      # fall back to the name
            side = ("plus" if ("plus" in name or "forward" in name or
                               name.endswith("+")) else "minus")
        out[f"{kind}_{side}"] = int(r.track_index)

    need = ("donor_plus", "acceptor_plus")
    missing = [k for k in need if k not in out]
    if missing:
        raise ValueError(
            f"could not resolve {missing} from the splice_sites rows:\n"
            f"{ss[['track_index', 'track_name', 'strand']].to_string(index=False)}")
    return out


# ==========================================================================
# per-sequence feature extraction
# ==========================================================================

def resolve_junction_tissue(catalog_path: str | Path,
                            usage_index: int) -> tuple[int, str]:
    """Plus-strand tissue index for CL:0002518 on the junction axis.

    pred_counts' last axis is cat([pos_counts, neg_counts]), each 367 wide
    (model.py:277). If the catalogue carries splice_junctions rows we resolve
    the index from them; if it does not, we fall back to the usage index,
    because both heads are built from the same 367-tissue ordering -- but we
    say so out loud, because that is an inference, not a lookup.
    """
    df = pd.read_parquet(catalog_path)
    jj = df[df.output_type == "splice_junctions"]
    if len(jj):
        hit = jj[(jj.ontology_curie.astype(str).str.replace("_", ":")
                  == F.ONTOLOGY[0]) & (jj.strand == "+")]
        if len(hit) == 1:
            idx = int(hit.iloc[0].track_index)
            if idx >= 367:
                raise ValueError(
                    f"junction plus-strand index {idx} is >= 367; the axis "
                    f"layout is not what we assumed")
            return idx, "catalogue"
        raise ValueError(
            f"{len(hit)} splice_junctions rows matched {F.ONTOLOGY[0]} (+); "
            f"expected exactly 1")
    return usage_index, "inferred-from-usage-head"


def pick_candidates(track, fixed: list[int], left_pad: int,
                    n: int = N_CANDIDATES, nms: int = NMS_WINDOW):
    """``fixed`` positions first, then the top-n remaining peaks.

    Returns a LongTensor (B, len(fixed) + n) of PADDED indices. The fixed
    columns are the designed sites, so their cells in the junction matrix
    are always present regardless of what the motif head scores; the rest
    are competitors found greedily with +/- ``nms`` nt suppression, so they
    are distinct sites and not one peak's shoulders.
    """
    import torch
    B, _ = track.shape
    work = track.clone()
    work[:, :left_pad] = -1.0
    work[:, left_pad + F.CONTEXT_LEN:] = -1.0
    for f in fixed:
        work[:, max(0, f - nms): f + nms + 1] = -1.0
    cols = [torch.full((B,), f, dtype=torch.long, device=track.device)
            for f in fixed]
    for _ in range(n):
        val, idx = work.max(dim=1)
        idx = torch.where(val > 0, idx, torch.full_like(idx, -1))
        cols.append(idx)
        for bi in range(B):
            k = int(idx[bi])
            if k >= 0:
                work[bi, max(0, k - nms): k + nms + 1] = -1.0
    return torch.stack(cols, dim=1)


def junction_features(pred_counts, don_cols, acc_cols, tissue: int,
                      left_pad: int) -> list[dict]:
    """Inclusion / skipping / cryptic competition from (B, P, P, 2*367).

    Column order is fixed by ``pick_candidates``:
        donors    [0] upstream_donor  [1] middle_donor  [2:] cryptic
        acceptors [0] middle_acceptor [1] downstream_acceptor [2:] cryptic
    so the biologically named cells are

        [0, 0]  upstream donor -> middle acceptor   inclusion, 5' junction
        [1, 1]  middle donor   -> downstream acc.   inclusion, 3' junction
        [0, 1]  upstream donor -> downstream acc.   EXON SKIPPING
        [1, 0]  middle donor   -> middle acceptor   backwards; must be ~0

    That last cell is the sanity check: its acceptor lies 75 nt UPSTREAM of
    its donor, so a non-trivial value means the axes are transposed.

    Counts are softplus outputs -- unbounded, non-negative -- so only ratios
    mean anything. ``psi`` is the inclusion share and is the quantity the
    whole experiment is about.
    """
    import torch
    m = pred_counts[:, :, :, tissue].float()
    B, P, _ = m.shape
    valid = (don_cols >= 0)[:, :, None] & (acc_cols >= 0)[:, None, :]
    m = torch.where(valid, m, torch.zeros_like(m))

    incl5, incl3 = m[:, 0, 0], m[:, 1, 1]
    skip, backwards = m[:, 0, 1], m[:, 1, 0]
    incl = 0.5 * (incl5 + incl3)
    total = m.sum(dim=(1, 2))

    # any junction using a cryptic site at either end
    cryptic = total - (incl5 + incl3 + skip + backwards)

    off = m.clone()
    for r, c in ((0, 0), (1, 1), (0, 1), (1, 0)):
        off[:, r, c] = 0.0
    omax, oarg = off.reshape(B, -1).max(dim=1)
    od, oa = (oarg // P).cpu().numpy(), (oarg % P).cpu().numpy()

    # Host copies. The _h suffix is not decoration: every name above is a
    # device tensor and every name below is its numpy twin, and `incl` vs
    # `inc` differing by one letter is how that goes wrong silently.
    def _h(t):
        return t.cpu().numpy()

    incl5_h, incl3_h = _h(incl5), _h(incl3)
    skip_h, back_h = _h(skip), _h(backwards)
    incl_h, total_h = _h(incl), _h(total)
    cryptic_h, offmax_h = _h(cryptic), _h(omax)
    don_h, acc_h = _h(don_cols), _h(acc_cols)

    def ratio(num: float, den: float) -> float:
        """NaN, never 0, when the denominator carries no signal.

        An eps guard here would be worse than useless: it turns 0/0 into a
        confident 0.0, so a locus the model routes NOTHING through reports
        psi = 0 and is indistinguishable from a fully skipped exon. That
        would pile fake mass at PSI = 0. NaN propagates instead, so those
        rows drop out of means and correlations rather than biasing them.
        Soft cases -- tiny but non-zero flux -- are not caught here; filter
        them on junc_total, which is recorded for exactly that purpose.
        """
        return float(num) / float(den) if den > 0 else float("nan")

    recs = []
    for i in range(B):
        recs.append({
            "junc_incl_5p": float(incl5_h[i]),
            "junc_incl_3p": float(incl3_h[i]),
            "junc_skip": float(skip_h[i]),
            "junc_backwards": float(back_h[i]),   # sanity: expect ~0
            "junc_total": float(total_h[i]),
            # The headline. incl_h is the PER-TRANSCRIPT inclusion count,
            # i.e. the mean of the two inclusion junctions -- an included
            # transcript contributes a read to BOTH of them, a skipped one
            # contributes a single read to skip_h. Summing instead of
            # averaging would overstate inclusion by 2x.
            "psi": ratio(incl_h[i], incl_h[i] + skip_h[i]),
            # how asymmetric the two inclusion junctions are
            "junc_incl_asym": ratio(incl5_h[i] - incl3_h[i],
                                    incl5_h[i] + incl3_h[i]),
            # share of all flux routed through ANY cryptic site
            "junc_cryptic_frac": ratio(cryptic_h[i], total_h[i]),
            "junc_top_offpair_frac": ratio(offmax_h[i], total_h[i]),
            "junc_top_offpair_donor_ctx": int(don_h[i, od[i]] - left_pad),
            "junc_top_offpair_acceptor_ctx": int(acc_h[i, oa[i]] - left_pad),
        })
    return recs


def windows(slot: str) -> dict[str, tuple[int, int]]:
    """Half-open [start, end) windows in CONTEXT coordinates."""
    g = F.GEOMETRY
    w = {
        "b1": (g["b1_start"], g["b1_end"]),
        "syn70": (g["syn_start"], g["syn_end"]),
        # the two designed sites, each with a +/-5 nt guard so the window
        # "away from the designed sites" really is away from them
        "exon": (g["exon_start"], g["donor"] + 1),
    }
    w["variable"] = w[slot]
    return w


def extract(ss_probs, usage_col, ch: dict[str, int], slot: str,
            left_pad: int, acc_pos: int, don_pos: int) -> list[dict]:
    """Per-sequence cryptic features. ``ss_probs`` is (B, L, 5) on device."""
    import torch

    d = ss_probs[:, :, ch["donor_plus"]]        # (B, L)
    a = ss_probs[:, :, ch["acceptor_plus"]]
    B = d.shape[0]
    wins = windows(slot)
    recs: list[dict] = [{} for _ in range(B)]

    # designed sites, from the splice_sites head (reference, not the target)
    for i in range(B):
        recs[i]["ss_acceptor_designed"] = float(a[i, acc_pos])
        recs[i]["ss_donor_designed"] = float(d[i, don_pos])
        recs[i]["usage_acceptor"] = float(usage_col[i, acc_pos])
        recs[i]["usage_donor"] = float(usage_col[i, don_pos])
        recs[i]["logit_average"] = F.logit_average(
            float(usage_col[i, acc_pos]), float(usage_col[i, don_pos]))

    def summarize(track, lo: int, hi: int, tag: str) -> None:
        sub = track[:, left_pad + lo: left_pad + hi]        # (B, w)
        mx, am = sub.max(dim=1)
        mx = mx.float().cpu().numpy()
        am = am.cpu().numpy()
        counts = {t: (sub > t).sum(dim=1).cpu().numpy() for t in THRESHOLDS}
        for i in range(B):
            recs[i][f"{tag}_max"] = float(mx[i])
            # report the position in CONTEXT coordinates, which is what the
            # geometry table and any downstream plot speak
            recs[i][f"{tag}_argmax_ctx"] = int(lo + am[i])
            # USAGE at that same base. splice_sites says a motif is there;
            # usage says the model actually routes transcripts through it.
            # Free -- usage_col is the whole (B, L) vector, already computed.
            recs[i][f"{tag}_usage_at_argmax"] = float(
                usage_col[i, left_pad + lo + int(am[i])])
            for t in THRESHOLDS:
                recs[i][f"{tag}_n_gt_{t}"] = int(counts[t][i])

    lo, hi = wins["variable"]
    summarize(d, lo, hi, "cryptic_donor_inslot")
    summarize(a, lo, hi, "cryptic_acceptor_inslot")

    # Everything in the context EXCEPT the designed exon and the variable
    # slot: competing sites the plasmid itself contributes. Masked rather
    # than sliced, because the excluded regions are not contiguous.
    keep = torch.ones(F.CONTEXT_LEN, dtype=torch.bool, device=d.device)
    for nm in ("exon", "variable"):
        klo, khi = wins[nm]
        keep[max(0, klo - 5): khi + 5] = False
    ctx_d = d[:, left_pad: left_pad + F.CONTEXT_LEN]
    ctx_a = a[:, left_pad: left_pad + F.CONTEXT_LEN]
    idx = torch.nonzero(keep, as_tuple=True)[0]
    for track, tag in ((ctx_d, "plasmid_donor"), (ctx_a, "plasmid_acceptor")):
        sub = track.index_select(1, idx)
        mx, am = sub.max(dim=1)
        mx = mx.float().cpu().numpy()
        am = idx[am].cpu().numpy()
        for i in range(B):
            recs[i][f"{tag}_max"] = float(mx[i])
            recs[i][f"{tag}_argmax_ctx"] = int(am[i])
            recs[i][f"{tag}_usage_at_argmax"] = float(
                usage_col[i, left_pad + int(am[i])])
    return recs


# ==========================================================================
# driver
# ==========================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", choices=sorted(LIBRARIES), default="upstream")
    ap.add_argument("-n", "--n", type=int, default=10_000,
                    help="sequences to sample from the manifest")
    ap.add_argument("--seed", type=int, default=20261117,
                    help="sampling seed; the same seed gives the same subset, "
                         "so the scan is reproducible and resumable")
    ap.add_argument("--weights",
                    default="/scratch/yl12316/models/alphagenome/"
                            "model_all_folds.safetensors")
    ap.add_argument("--catalog", default=str(F.CATALOG_PATH))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--flush-every", type=int, default=500)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-junctions", action="store_true",
                    help="skip the splice_junctions pass (halves compute; "
                         "leaves only motif + usage evidence for cryptic "
                         "sites, with no competition measure)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import local_backend as LB

    cfg = LIBRARIES[args.library]
    dsdir = OUT / cfg["dirname"]
    manifest = dsdir / "manifest.csv"
    slot = cfg["slot"]
    out_csv = args.out or (dsdir / f"cryptic_fullctx_{args.n}.csv")

    if not manifest.exists():
        sys.exit(f"FATAL: {manifest} not found; build the library first")

    print("=" * 72)
    print(f"CRYPTIC SCAN  {args.library}  full {F.CONTEXT_LEN}-nt context")
    print("=" * 72)

    ch = resolve_splice_site_channels(args.catalog)
    print("  splice_sites channels:")
    for k in sorted(ch):
        print(f"    {k:16s} -> {ch[k]}")

    man = pd.read_csv(manifest, usecols=["row_id", slot, "split"])
    man["row_id"] = man.row_id.astype(str)
    samp = man.sample(n=min(args.n, len(man)), random_state=args.seed)
    samp = samp.sort_index()
    print(f"  manifest    {len(man):,} rows")
    print(f"  sampled     {len(samp):,} (seed {args.seed})")
    print(f"  splits      {samp.split.value_counts().to_dict()}")

    done: set[str] = set()
    if out_csv.exists():
        done = set(pd.read_csv(out_csv, usecols=["row_id"]).row_id.astype(str))
        print(f"  resume      {len(done):,} already scanned")
    todo = samp[~samp.row_id.isin(done)]
    print(f"  to scan     {len(todo):,}")
    print(f"  out         {out_csv}")

    if args.dry_run:
        print("\n--dry-run: no model loaded")
        return 0
    if not len(todo):
        print("\nnothing to do")
        return 0

    scorer = LB.LocalScorer(args.weights, device=args.device,
                            catalog_path=args.catalog,
                            batch_size=args.batch_size)
    pre = LB.preflight(scorer)

    exons = pd.read_csv(OUT / "b1_pilot_exons.csv")
    syn_primary = dict(zip(exons.sequence_id, exons.synthetic_70mer))[F.PRIMARY_EXON]
    native_b1 = SP.pnas_blocks8()["B1"]
    context = F.load_context()
    left_pad, acc_pos, don_pos = F.padded_positions()

    sites = F.designed_sites(left_pad)
    print("  designed sites (ctx): "
          + "  ".join(f"{k}={v - left_pad}" for k, v in sites.items()))
    tissue_idx = tissue_src = None
    if not args.no_junctions:
        tissue_idx, tissue_src = resolve_junction_tissue(
            args.catalog, scorer.track_index)
        print(f"  junction tissue index {tissue_idx} ({tissue_src}), "
              f"P = {1 + N_CANDIDATES} per role")
        if tissue_src != "catalogue":
            print("  NOTE: the catalogue has no splice_junctions rows, so "
                  "this index is inferred from the usage head's ordering.")

    LB.write_provenance(scorer, dsdir / f"cryptic_provenance_{args.n}.json",
                        extra={"library": args.library, "slot": slot,
                               "n_sampled": int(len(samp)),
                               "sample_seed": args.seed,
                               "splice_site_channels": ch,
                               "thresholds": list(THRESHOLDS),
                               "junctions": not args.no_junctions,
                               "junction_tissue_index": tissue_idx,
                               "junction_tissue_source": tissue_src,
                               "n_candidates": N_CANDIDATES,
                               "designed_sites_ctx": {k: v - left_pad for k, v in sites.items()},
                               "nms_window": NMS_WINDOW,
                               "preflight": pre})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    buf: list[dict] = []
    written = 0
    t0 = time.time()

    def flush() -> None:
        nonlocal written
        if not buf:
            return
        frame = pd.DataFrame(buf)
        if out_csv.exists():
            # Appending a DIFFERENT column set to an existing header silently
            # misaligns every later row. That is exactly what happens if the
            # feature set changed between runs (e.g. --no-junctions, or a new
            # column added), so refuse rather than corrupt.
            head = list(pd.read_csv(out_csv, nrows=0).columns)
            if head != list(frame.columns):
                raise SystemExit(
                    f"FATAL: {out_csv} was written with a different schema.\n"
                    f"  on disk ({len(head)}): {head}\n"
                    f"  now     ({len(frame.columns)}): "
                    f"{list(frame.columns)}\n"
                    f"  Move the old file aside or pass --out to a new path.")
            frame.to_csv(out_csv, mode="a", index=False, header=False)
        else:
            frame.to_csv(out_csv, mode="a", index=False, header=True)
        written += len(buf)
        buf.clear()

    rows = list(todo.itertuples())
    i = 0
    while i < len(rows):
        block = rows[i:i + scorer.batch_size]
        seqs, metas = [], []
        for r in block:
            var = getattr(r, slot)
            b1 = var if slot == "b1" else native_b1
            s70 = var if slot == "syn70" else syn_primary
            core = F.build_construct(b1, s70, context=context)
            padded, _, _ = F.center_pad(core)
            seqs.append(padded)
            metas.append({"row_id": r.row_id, slot: var, "split": r.split})

        out = scorer.predict_heads(seqs, ("splice_sites", "splice_site_usage"))
        ss = out["splice_sites"]["probs"]
        usage = out["splice_site_usage"]["predictions"][:, :, scorer.track_index]
        feats = extract(ss, usage, ch, slot, left_pad, acc_pos, don_pos)

        if tissue_idx is not None:
            # Second pass. The candidate list is chosen FROM pass 1, so it
            # cannot be supplied to pass 1 -- hence two forward calls per
            # batch. The junction head itself is nearly free at P=8; the
            # cost here is one extra trunk evaluation.
            import torch
            don_cols = pick_candidates(
                ss[:, :, ch["donor_plus"]],
                [sites["upstream_donor"], sites["middle_donor"]], left_pad)
            acc_cols = pick_candidates(
                ss[:, :, ch["acceptor_plus"]],
                [sites["middle_acceptor"], sites["downstream_acceptor"]],
                left_pad)
            neg = torch.full_like(don_cols, -1)
            positions = torch.stack([don_cols, acc_cols, neg, neg], dim=1)
            blk = scorer.predict_junctions(seqs, positions)
            jf = junction_features(blk["pred_counts"], don_cols, acc_cols,
                                   tissue_idx, left_pad)
            for f, j in zip(feats, jf):
                f.update(j)

        for meta, f in zip(metas, feats):
            meta.update(f)
            buf.append(meta)

        i += len(block)
        if len(buf) >= args.flush_every:
            flush()
        if i % max(args.flush_every, 1) < scorer.batch_size or i <= scorer.batch_size:
            el = time.time() - t0
            rate = i / max(el, 1e-9)
            print(f"  [{i:,}/{len(rows):,}] {rate:.2f} seq/s | "
                  f"ETA {(len(rows) - i) / max(rate, 1e-9) / 60:.1f} min",
                  flush=True)

    flush()
    el = time.time() - t0
    print(f"\n[done] {written:,} rows -> {out_csv} in {el / 60:.1f} min "
          f"({written / max(el, 1e-9):.2f} seq/s)")
    _summary(out_csv, slot)
    return 0


def _slot_bounds(slot: str) -> tuple[int, int]:
    lo, hi = windows(slot)["variable"]
    return lo, hi - 1


def _summary(out_csv: Path, slot: str) -> None:
    """The three numbers the v3 check turned on, printed immediately."""
    df = pd.read_csv(out_csv)
    print("\n" + "=" * 72)
    print(f"SUMMARY  n = {len(df):,}")
    print("=" * 72)
    for col in ("cryptic_donor_inslot_max", "cryptic_acceptor_inslot_max",
                "cryptic_donor_inslot_usage_at_argmax",
                "cryptic_acceptor_inslot_usage_at_argmax",
                "plasmid_donor_max", "plasmid_acceptor_max",
                "plasmid_donor_usage_at_argmax"):
        if col in df:
            v = df[col]
            print(f"  {col:34s} median {v.median():.4f}  "
                  f"p95 {v.quantile(0.95):.4f}  max {v.max():.4f}")
    for t in THRESHOLDS:
        c = f"cryptic_donor_inslot_n_gt_{t}"
        if c in df:
            frac = (df[c] > 0).mean()
            print(f"  sequences with a donor > {t:<4} in the slot: "
                  f"{100 * frac:5.1f} %")
    if "logit_average" in df and "cryptic_donor_inslot_max" in df:
        v = df.logit_average.var()
        for t in THRESHOLDS:
            clean = df[df[f"cryptic_donor_inslot_n_gt_{t}"] == 0]
            if len(clean) > 10:
                print(f"  dropping donor > {t:<4}: n {len(clean):,} "
                      f"({100 * len(clean) / len(df):.0f} %), variance "
                      f"retained {100 * clean.logit_average.var() / v:.0f} %")
        r = df.logit_average.corr(df.cryptic_donor_inslot_max)
        print(f"  corr(logit_average, cryptic_donor_max)   = {r:+.4f}")
        uc = "cryptic_donor_inslot_usage_at_argmax"
        if uc in df:
            print(f"  corr(logit_average, cryptic_donor_usage) = "
                  f"{df.logit_average.corr(df[uc]):+.4f}")
            print(f"  cryptic donor USAGE > 0.05 in slot: "
                  f"{100 * (df[uc] > 0.05).mean():5.1f} % of sequences")
    if "psi" in df:
        print("  -- junction flux (softplus counts; ratios only) --")
        for col in ("psi", "junc_incl_asym", "junc_cryptic_frac",
                    "junc_top_offpair_frac"):
            v = df[col]
            print(f"  {col:34s} median {v.median():.4f}  "
                  f"p05 {v.quantile(0.05):.4f}  p95 {v.quantile(0.95):.4f}")
        n_nan = int(df.psi.isna().sum())
        if n_nan:
            print(f"  psi is NaN (no predicted flux) in {n_nan:,} rows "
                  f"({100 * n_nan / len(df):.1f} %) -- excluded above")
        # the axes-transposed tripwire: this junction runs backwards
        bw = df.junc_backwards / (df.junc_total + 1e-12)
        print(f"  backwards junction share (expect ~0): max {bw.max():.2e}")
        if "logit_average" in df:
            print(f"  corr(logit_average, psi)                = "
                  f"{df.logit_average.corr(df.psi):+.4f}")
            print(f"  corr(logit_average, junc_cryptic_frac)  = "
                  f"{df.logit_average.corr(df.junc_cryptic_frac):+.4f}")
        lo, hi = _slot_bounds(slot)
        inslot = df.junc_top_offpair_donor_ctx.between(lo, hi)
        print(f"  top competing donor lies INSIDE the variable slot: "
              f"{100 * inslot.mean():5.1f} % of sequences")


if __name__ == "__main__":
    raise SystemExit(main())
