"""Full-context (ExonSkip7_Lib2) constructs and the logit-average metric.

Three things change relative to every earlier upstream-exon experiment, and
all three live here so that nothing already on disk has to be touched:

1. CONTEXT.  Earlier runs assembled a 1,309-nt core from
   ``splice_site_partition.pnas_blocks8()`` and centre-padded it with N to
   16,384. This module instead uses the ENTIRE 8,612-nt ExonSkip7_Lib2
   plasmid context as given, and only substitutes the two variable slots that
   the file itself marks with N. The RNA is circular; the cut point in the
   text file is the one we have decided to use, and it is not re-derived here.

2. READOUT.  ``OutputType.SPLICE_SITE_USAGE`` replaces ``SPLICE_SITES``.
   SPLICE_SITES says a position *looks* like a splice site. SPLICE_SITE_USAGE
   says what fraction of transcripts actually *use* it, which is the quantity
   the PNAS assay measures.

3. METRIC.  The target is the LOGIT-AVERAGE::

       logit_average = logit( (usage_acceptor + usage_donor) / 2 )

   Note the order: the two usages are averaged on the PROBABILITY scale and
   the logit is taken once, at the end. This is NOT the old ``AG_mean_logit``,
   which was ``(logit(acc) + logit(don)) / 2`` -- a logit of each, then the
   mean. The two disagree whenever the acceptor and donor differ, because
   logit is convex above 0.5 and concave below it; Jensen's inequality makes
   the old statistic the larger of the two in the tails. They are different
   numbers and must never be compared or pooled.

   From here on, logit-average is the only target.

GEOMETRY. The block coordinates are DERIVED by locating the repo's own PNAS
blocks inside the context at import time, then checked against the constants
recorded below. A context file that no longer matches raises immediately
rather than silently scoring the wrong positions.

One known difference between the plasmid and the repo's reconstructed
cassette: the context carries one extra ``T`` at context position 6408, inside
intron 2 (B6), 205 nt downstream of the middle exon. Everything else is
identical. The context file is treated as ground truth and the extra base is
kept -- this module never rebuilds the cassette from blocks.

Makes NO AlphaGenome calls at import.
"""

from __future__ import annotations

import hashlib
import math
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
for _p in (str(_THIS_DIR), str(ROOT / "analysis_script")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import splice_site_partition as SP  # noqa: E402
import synthetic_context_lib as L  # noqa: E402

#: the plasmid context, as text, at the repository root
CONTEXT_TXT = ROOT / "ExonSkip7_Lib2_text.txt"

#: length of the context after stripping whitespace and numbering
CONTEXT_LEN = 8612

#: AlphaGenome input length the context is centre-padded into. 8,612 is not an
#: accepted input size; 16,384 is the smallest accepted size that holds it.
TARGET_LEN = 16_384

#: expected 0-based geometry inside the 8,612-nt context. Derived at import
#: and asserted against these, so a changed context file fails loudly.
EXPECT = {
    "b1_start": 5885,       # upstream exon body, 113 nt
    "b1_end": 5998,
    "syn_start": 6131,      # the 70-N run: middle exon body
    "syn_end": 6201,
    "exon_start": 6128,     # GTT + syn70 + CAG, i.e. the acceptor base
    "acceptor": 6128,
    "donor": 6203,          # exon_start + 76 - 1
    "barcode_start": 7195,  # the 20-N run, downstream of B8
    "barcode_end": 7215,
}

B1_LEN = 113
SYN_LEN = 70
EXON_LEN = 76           # EXON_PREFIX(3) + SYN_LEN(70) + EXON_SUFFIX(3)
ALPHABET = "ACGT"
ONTOLOGY = ("CL:0002518",)

#: the middle exon held fixed while B1 varies, as in v1-v3
PRIMARY_EXON = "syn_a5.0_00482"


# ==========================================================================
# context loading and geometry
# ==========================================================================

def load_context(path: Path = CONTEXT_TXT) -> str:
    """The 8,612-nt context, upper-cased, non-letters stripped."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The full-context experiments need the "
            f"ExonSkip7_Lib2 text file at the repository root.")
    seq = re.sub(r"[^A-Za-z]", "", path.read_text()).upper()
    if len(seq) != CONTEXT_LEN:
        raise ValueError(
            f"context length {len(seq)} != {CONTEXT_LEN}; refusing to guess "
            f"at the geometry of a different file")
    bad = set(seq) - set("ACGTN")
    if bad:
        raise ValueError(f"context has unexpected characters: {sorted(bad)}")
    return seq


def derive_geometry(context: str | None = None) -> dict[str, int]:
    """Locate the variable slots in the context and check them.

    The two N runs give the variable slots directly. B1 is NOT N-masked in the
    file -- it carries the native PNAS upstream exon body -- so it is found by
    searching for that block, which must occur exactly once.
    """
    ctx = load_context() if context is None else context
    g: dict[str, int] = {}

    runs = [(m.start(), m.end()) for m in re.finditer(r"N+", ctx)]
    if len(runs) != 2:
        raise ValueError(f"expected exactly 2 N runs, found {len(runs)}: {runs}")
    (syn_start, syn_end), (bc_start, bc_end) = sorted(runs)
    if syn_end - syn_start != SYN_LEN:
        raise ValueError(
            f"first N run is {syn_end - syn_start} nt, expected {SYN_LEN} "
            f"for the middle exon body")
    g["syn_start"], g["syn_end"] = syn_start, syn_end
    g["barcode_start"], g["barcode_end"] = bc_start, bc_end

    native = SP.pnas_blocks8()["B1"]
    if ctx.count(native) != 1:
        raise ValueError(
            f"native B1 occurs {ctx.count(native)} times in the context; "
            f"expected exactly 1")
    g["b1_start"] = ctx.find(native)
    g["b1_end"] = g["b1_start"] + B1_LEN

    # The middle exon opens EXON_PREFIX ("GTT") before the 70-mer and closes
    # EXON_SUFFIX ("CAG") after it; those six bases are splice-site context and
    # are never randomized.
    g["exon_start"] = syn_start - len(L.EXON_PREFIX)
    g["acceptor"] = g["exon_start"]
    g["donor"] = g["exon_start"] + EXON_LEN - 1

    if ctx[g["exon_start"]:syn_start] != L.EXON_PREFIX:
        raise ValueError(
            f"expected {L.EXON_PREFIX!r} immediately before the 70-N run, "
            f"found {ctx[g['exon_start']:syn_start]!r}")
    if ctx[syn_end:syn_end + len(L.EXON_SUFFIX)] != L.EXON_SUFFIX:
        raise ValueError(
            f"expected {L.EXON_SUFFIX!r} immediately after the 70-N run, "
            f"found {ctx[syn_end:syn_end + len(L.EXON_SUFFIX)]!r}")

    mismatched = {k: (g[k], EXPECT[k]) for k in EXPECT if g[k] != EXPECT[k]}
    if mismatched:
        raise ValueError(
            f"derived geometry disagrees with the recorded constants "
            f"(key: derived vs expected): {mismatched}")
    return g


GEOMETRY = derive_geometry()


# ==========================================================================
# construct assembly
# ==========================================================================

def build_construct(b1: str, syn70: str, *, barcode: str | None = None,
                    context: str | None = None) -> str:
    """The full 8,612-nt context with the two variable slots filled.

    ``barcode`` is the 20-nt library barcode that the file leaves as N. It is
    a per-clone tag that is not part of the designed cassette and is unknown
    for any given library member, so the default keeps it as N -- an honest
    "unobserved" rather than an invented sequence. Supplying one is allowed
    but changes the input and so changes the labels.
    """
    ctx = load_context() if context is None else context
    g = GEOMETRY

    if len(b1) != B1_LEN:
        raise ValueError(f"b1 length {len(b1)} != {B1_LEN}")
    if len(syn70) != SYN_LEN:
        raise ValueError(f"syn70 length {len(syn70)} != {SYN_LEN}")
    for name, s in (("b1", b1), ("syn70", syn70)):
        if set(s) - set(ALPHABET):
            raise ValueError(
                f"{name} has non-ACGT characters: {sorted(set(s) - set(ALPHABET))}")

    out = (ctx[:g["b1_start"]] + b1 + ctx[g["b1_end"]:g["syn_start"]]
           + syn70 + ctx[g["syn_end"]:])

    if barcode is not None:
        if len(barcode) != g["barcode_end"] - g["barcode_start"]:
            raise ValueError(
                f"barcode length {len(barcode)} != "
                f"{g['barcode_end'] - g['barcode_start']}")
        out = (out[:g["barcode_start"]] + barcode.upper()
               + out[g["barcode_end"]:])

    if len(out) != CONTEXT_LEN:
        raise ValueError(f"assembled length {len(out)} != {CONTEXT_LEN}")
    return out


def validate_construct(seq: str, b1: str, syn70: str,
                       *, barcode: str | None = None) -> list[str]:
    """Every structural assertion for one full-context construct."""
    g = GEOMETRY
    ctx = load_context()
    fails: list[str] = []

    if len(seq) != CONTEXT_LEN:
        fails.append(f"length {len(seq)} != {CONTEXT_LEN}")
        return fails                      # offsets below would be meaningless

    if seq[g["b1_start"]:g["b1_end"]] != b1:
        fails.append("b1 not at the expected offset")
    if seq[g["syn_start"]:g["syn_end"]] != syn70:
        fails.append("syn70 not at the expected offset")
    if seq[g["exon_start"]:g["syn_start"]] != L.EXON_PREFIX:
        fails.append("EXON_PREFIX missing before the middle exon body")
    if seq[g["syn_end"]:g["syn_end"] + 3] != L.EXON_SUFFIX:
        fails.append("EXON_SUFFIX missing after the middle exon body")

    # everything outside the two slots (and the barcode) must be untouched
    frozen = [(0, g["b1_start"]), (g["b1_end"], g["syn_start"])]
    if barcode is None:
        frozen.append((g["syn_end"], CONTEXT_LEN))
    else:
        frozen += [(g["syn_end"], g["barcode_start"]),
                   (g["barcode_end"], CONTEXT_LEN)]
    for lo, hi in frozen:
        if seq[lo:hi] != ctx[lo:hi]:
            fails.append(f"context altered in [{lo}, {hi})")

    expected_n = 0 if barcode is not None else (g["barcode_end"]
                                                - g["barcode_start"])
    if seq.count("N") != expected_n:
        fails.append(f"{seq.count('N')} N bases, expected {expected_n}")
    return fails


def padded_positions(context_len: int = CONTEXT_LEN,
                     target_len: int = TARGET_LEN) -> tuple[int, int, int]:
    """``(left_pad, acceptor_index, donor_index)`` in the padded input."""
    left_pad = (target_len - context_len) // 2
    return (left_pad, left_pad + GEOMETRY["acceptor"],
            left_pad + GEOMETRY["donor"])


# ==========================================================================
# the metric
# ==========================================================================

def logit_average(usage_acceptor: float, usage_donor: float,
                  eps: float = 1e-7) -> float:
    """logit of the MEAN USAGE -- average first, logit once.

    Deliberately not ``(logit(a) + logit(d)) / 2``. See the module docstring.
    """
    return L._logit((float(usage_acceptor) + float(usage_donor)) / 2.0, eps=eps)


def get_usage_track(pred_out) -> np.ndarray:
    """The positive-strand splice-site-usage track, as one 1-D array.

    SPLICE_SITE_USAGE is a per-base quantity per (ontology, strand); with a
    single ontology term there is one positive-strand track, and the usage at
    an acceptor and at a donor are read from that same track at their two
    positions. Older/other layouts that name separate donor and acceptor
    channels are also accepted. Anything else raises rather than guessing
    which column is which.
    """
    usage = pred_out.splice_site_usage
    if hasattr(usage, "filter_to_positive_strand"):
        usage = usage.filter_to_positive_strand()

    values = np.asarray(usage.values, dtype=np.float64)
    if values.ndim == 1:
        return values
    if values.shape[1] == 1:
        return values[:, 0]

    names = [str(n).lower() for n in getattr(usage, "names", [])]
    if "donor" in names and "acceptor" in names:
        raise ValueError(
            "splice_site_usage exposes separate donor/acceptor channels "
            f"({names}); use get_usage_pair() instead of get_usage_track()")
    raise ValueError(
        f"splice_site_usage has {values.shape[1]} positive-strand tracks "
        f"(names={names}); expected exactly 1. Narrow ontology_terms, or "
        f"resolve the layout with probe_splice_site_usage.py before scoring.")


def get_usage_pair(pred_out) -> tuple[np.ndarray, np.ndarray]:
    """``(donor_track, acceptor_track)`` when the layout names both channels."""
    usage = pred_out.splice_site_usage
    if hasattr(usage, "filter_to_positive_strand"):
        usage = usage.filter_to_positive_strand()
    values = np.asarray(usage.values, dtype=np.float64)
    names = [str(n).lower() for n in usage.names]
    return values[:, names.index("donor")], values[:, names.index("acceptor")]


#: backends ``make_model`` knows how to build
BACKENDS = ("api", "local")


def make_model(backend: str = "api", *, env_path: str | None = None, **kw):
    """Build a predictor.

    THE CONTRACT. Everything downstream -- geometry, assembly, the metric,
    the resumable loop -- touches the predictor through exactly one method::

        model.predict_sequence(
            sequence=<str of length TARGET_LEN>,
            organism=...,
            requested_outputs=[OutputType.SPLICE_SITE_USAGE, ...],
            ontology_terms=[...],
        ) -> object with .splice_site_usage (and .splice_sites if requested)

    So swapping the API for locally held weights is a change to this function
    and nothing else, as long as the adapter returns per-base tracks that
    ``get_usage_track`` can read.

    ``api``   the DeepMind client in ``alphagenome/src``: a gRPC stream to
              Google, authenticated from ``.env.alphagenome``, subject to the
              published quota (the SDK's own README warns it is "likely not
              suitable" past ~1e6 predictions).
    ``local`` model weights held on the cluster. No quota and no single-stream
              restriction, so the caller should also batch and drop the
              serialization that the API path needs.
    """
    if backend == "api":
        import alphagenome_local_prediction_pipeline_minigene as ag
        return ag.create_model(env_path=env_path)
    if backend == "local":
        raise NotImplementedError(
            "The local backend is not wired up yet. Implement an adapter "
            "exposing predict_sequence(...) as described in make_model's "
            "docstring and return it here; nothing else in this module needs "
            "to change. Note that a local backend also makes batching "
            "worthwhile, which run_scoring does not currently do -- it calls "
            "one sequence at a time because the API quota forced that.")
    raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")


def score_construct(model, seq: str, *, ontology_terms=ONTOLOGY,
                    also_splice_sites: bool = False) -> dict[str, float]:
    """Centre-pad, call AlphaGenome, and read the two usages.

    With ``also_splice_sites`` the old SPLICE_SITES probabilities and the old
    ``AG_mean_logit`` are recorded alongside, from the SAME call, so the two
    metrics can be compared without spending a second call. That costs nothing
    in calls but makes each response larger.
    """
    import alphagenome_local_prediction_pipeline_minigene as ag
    from alphagenome.models import dna_client

    padded, left_pad, _ = ag.center_pad(ag.clean_seq(seq), target_len=TARGET_LEN)
    acc_pos = left_pad + GEOMETRY["acceptor"]
    don_pos = left_pad + GEOMETRY["donor"]

    outputs = [dna_client.OutputType.SPLICE_SITE_USAGE]
    if also_splice_sites:
        outputs.append(dna_client.OutputType.SPLICE_SITES)

    pred = model.predict_sequence(
        sequence=padded,
        organism=dna_client.Organism.HOMO_SAPIENS,
        requested_outputs=outputs,
        ontology_terms=list(ontology_terms) if ontology_terms else None,
    )

    try:
        track = get_usage_track(pred)
        u_acc, u_don = float(track[acc_pos]), float(track[don_pos])
    except ValueError:
        donor_t, acceptor_t = get_usage_pair(pred)
        u_acc, u_don = float(acceptor_t[acc_pos]), float(donor_t[don_pos])

    rec = {
        "AG_acceptor_usage": u_acc,
        "AG_donor_usage": u_don,
        "AG_mean_usage": (u_acc + u_don) / 2.0,
        "logit_average": logit_average(u_acc, u_don),
    }
    if also_splice_sites:
        d_t, a_t = ag.get_positive_strand_tracks(pred)
        p_acc, p_don = float(a_t[acc_pos]), float(d_t[don_pos])
        rec.update({
            "AG_acceptor_probability": p_acc,
            "AG_donor_probability": p_don,
            # the OLD metric, recorded only for the metric-comparison figure
            "legacy_mean_logit": (L._logit(p_acc) + L._logit(p_don)) / 2.0,
        })
    return rec


# ==========================================================================
# resumable scoring
# ==========================================================================

def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def run_scoring(jobs, out_csv: Path, env_path: str, *,
                flush_every: int = 200, call_timeout: float = 90.0,
                max_retries: int = 3, failed_csv: Path | None = None,
                also_splice_sites: bool = False,
                ontology_terms=ONTOLOGY, progress_every: int = 200,
                skip_ids: set[str] | None = None,
                done_before: int = 0, shard_total: int = 0,
                backend: str = "api") -> int:
    """Score ``jobs`` one at a time, APPENDING to ``out_csv``.

    Appending rather than rewriting is the difference that makes a million-row
    run feasible: ``synthetic_context_lib.run_scoring`` rewrites the whole CSV
    at every flush, which is O(n) per flush and O(n^2) over a run. At 250,000
    rows that was merely wasteful; at 1,000,000 it would dominate the wall
    clock.

    Each job is a dict with ``row_id`` plus whatever identifying columns should
    be carried through, and ``seq`` (dropped before writing). Resumes on
    ``row_id``. A single stream only: the AlphaGenome quota is not known to
    permit concurrent use.

    ``skip_ids`` supplies the already-scored ids directly. The caller usually
    has them already, and at a million rows re-reading the results CSV once
    per batch is the difference between a second and several minutes of pure
    bookkeeping per shard.
    """
    jobs = list(jobs)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if skip_ids is not None:
        done = skip_ids
    elif out_csv.exists():
        done = set(pd.read_csv(out_csv, usecols=["row_id"]).row_id.astype(str))
        print(f"[resume] {len(done):,} rows already scored in {out_csv.name}",
              flush=True)
    else:
        done = set()

    todo = [j for j in jobs if str(j["row_id"]) not in done]
    total = len(todo)
    # When the caller drives batches (shard_total set) it has already printed
    # the shard banner; repeating it once per batch is noise in the log.
    if not shard_total:
        print(f"[scoring] {total:,} of {len(jobs):,} rows to call", flush=True)
    if not total:
        return 0

    model = make_model(backend, env_path=env_path)
    pool = ThreadPoolExecutor(max_workers=1)
    buf: list[dict] = []
    failures: list[dict] = []
    written = 0
    t0 = time.time()

    def flush() -> None:
        nonlocal written
        if not buf:
            return
        pd.DataFrame(buf).to_csv(out_csv, mode="a", index=False,
                                 header=not out_csv.exists())
        written += len(buf)
        buf.clear()

    for i, job in enumerate(todo, 1):
        result = None
        for attempt in range(1, max_retries + 1):
            try:
                fut = pool.submit(score_construct, model, job["seq"],
                                  ontology_terms=ontology_terms,
                                  also_splice_sites=also_splice_sites)
                result = fut.result(timeout=call_timeout)
                break
            except FutureTimeout:
                print(f"  [timeout {attempt}/{max_retries}] {job['row_id']} "
                      f"(>{call_timeout}s) -- rebuilding client", flush=True)
                pool.shutdown(wait=False)
                pool = ThreadPoolExecutor(max_workers=1)
                model = make_model(backend, env_path=env_path)
            except Exception as exc:                      # noqa: BLE001
                print(f"  [error {attempt}/{max_retries}] {job['row_id']}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                time.sleep(min(2 ** attempt, 30))
        if result is None:
            rec = {k: v for k, v in job.items() if k != "seq"}
            failures.append(rec)
            if failed_csv:
                # Append: the caller drives this in batches, and rewriting
                # would drop every failure from the earlier batches.
                pd.DataFrame([rec]).to_csv(failed_csv, mode="a", index=False,
                                           header=not failed_csv.exists())
            continue

        rec = {k: v for k, v in job.items() if k != "seq"}
        rec.update(result)
        buf.append(rec)

        if len(buf) >= flush_every:
            flush()
        if progress_every and i % progress_every == 0:
            el = time.time() - t0
            rate = el / i
            # Report against the whole SHARD, not this batch: the caller
            # drives batches of a couple of thousand, and a per-batch ETA
            # would read as minutes when hours are left.
            if shard_total:
                did = done_before + i
                print(f"  [{did:,}/{shard_total:,} this shard] "
                      f"{rate:.3f}s/call | "
                      f"ETA {(shard_total - did) * rate / 3600:.2f}h",
                      flush=True)
            else:
                print(f"  [{i:,}/{total:,}] {rate:.3f}s/call | "
                      f"elapsed {el / 3600:.2f}h | "
                      f"ETA {(total - i) * rate / 3600:.2f}h", flush=True)

    flush()
    pool.shutdown(wait=False)
    el = time.time() - t0
    if not shard_total:
        print(f"\n[done] wrote {written:,} new rows -> {out_csv} "
              f"in {el / 3600:.2f}h ({el / max(1, total):.3f}s/call)",
              flush=True)
    if failures:
        print(f"[warn] {len(failures):,} failures in this batch "
              f"-> {failed_csv}", flush=True)
    return written
