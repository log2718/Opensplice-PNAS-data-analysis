"""Batched scoring of the full context against LOCAL alphagenome_pytorch weights.

Why this exists as its own module rather than another ``make_model`` branch:
the API forced one call at a time, so ``fullctx_lib.run_scoring`` is a
single-stream loop with timeouts, client rebuilds and retry backoff -- all of
which are gRPC concerns that mean nothing to a local forward pass. Locally the
throughput lever is the batch, so this module owns a batched loop that keeps
the same on-disk contract (append-mode CSV, resume on ``row_id``, a failures
sidecar) and nothing else.

Two facts from the 0.3.1 source make this efficient and correct:

* ``forward(..., heads=('splice_site_usage',))`` -- ``predict`` passes
  ``**kwargs`` through, and the usage head is gated on membership in that
  tuple. Without it every call also materializes ``splice_junctions``
  ``pred_counts`` at (B, 512, 512, 734) float32 = 779 MB *per sequence*, which
  is both the memory ceiling and most of the compute, for an output we discard.
* ``splice_site_usage['predictions']`` is already probabilities in [0, 1] at
  1 bp over all 16,384 input positions (uncropped), so the two positions we
  want can be sliced directly and no sigmoid is applied.

The CL:0002518 track is selected by INDEX (the local head takes no
``ontology_terms``); see ``fullctx_lib.resolve_usage_track``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import fullctx_lib as F  # noqa: E402

#: the only head we ask forward() to compute
HEAD = "splice_site_usage"

#: batch sizes tried in order; the first that does not OOM is kept
BATCH_LADDER = (32, 16, 8, 4, 2, 1)

#: tissues per strand in the splice_junctions head (model.py:277). The
#: pred_counts last axis is 2x this: [plus 367][minus 367].
N_JUNCTION_TISSUES = 367


# ==========================================================================
# the scorer
# ==========================================================================

class LocalScorer:
    """Holds the weights and turns construct strings into the two usages.

    One instance per process. ``batch_size`` is a ceiling, not a promise: the
    first batch probes downward through ``BATCH_LADDER`` on CUDA OOM and the
    surviving size is reused, so a job does not die an hour in because one
    allocation spiked.
    """

    def __init__(self, weights: str | Path, *, device: str = "cuda",
                 catalog_path: str | Path = F.CATALOG_PATH,
                 track_index: int | None = None,
                 batch_size: int = 32, verify_catalog: bool = True):
        import torch
        from alphagenome_pytorch import AlphaGenome

        self.torch = torch
        self.weights = str(weights)
        self.device = device

        # Resolve the track from the catalogue rather than trusting the
        # constant. Every label depends on this index being right.
        if track_index is None:
            if verify_catalog:
                self._check_catalog(catalog_path)
            track_index = F.resolve_usage_track(catalog_path, strand="+")
        self.track_index = int(track_index)
        self.track_index_minus = F.USAGE_TRACK_MINUS

        _, self.acc_pos, self.don_pos = F.padded_positions()

        print(f"[local] loading {self.weights} on {device}", flush=True)
        t0 = time.time()
        self.model = AlphaGenome.from_pretrained(self.weights, device=device)
        self.model.eval()
        print(f"[local] loaded in {time.time() - t0:.1f}s | "
              f"track {self.track_index} (+) | "
              f"acceptor {self.acc_pos} donor {self.don_pos}", flush=True)

        self._onehot_fn = self._find_onehot_fn()
        self.batch_size = int(batch_size)
        self._size_logged = False

    # ---------------------------------------------------------------- setup

    @staticmethod
    def _check_catalog(catalog_path: str | Path) -> None:
        import hashlib

        p = Path(catalog_path)
        if not p.exists():
            raise FileNotFoundError(
                f"track catalogue missing: {p}\nfetch it with:\n"
                f"  curl -fL --create-dirs -o {p} {F.CATALOG_URL}")
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != F.CATALOG_SHA256:
            raise ValueError(
                f"track catalogue sha256 mismatch for {p}\n"
                f"  expected {F.CATALOG_SHA256}\n  got      {got}\n"
                "A different catalogue may order the 734 tracks differently, "
                "which would silently change which cell type is scored.")

    @staticmethod
    def _find_onehot_fn():
        from alphagenome_pytorch import utils
        for name in ("sequence_to_onehot_tensor", "sequence_to_onehot",
                     "dna_to_onehot", "one_hot_encode"):
            fn = getattr(utils, name, None)
            if callable(fn):
                return fn
        raise AttributeError(
            "no one-hot helper found in alphagenome_pytorch.utils; "
            f"available: {[a for a in dir(utils) if not a.startswith('_')]}")

    def _onehot(self, seqs: list[str]):
        """``(B, TARGET_LEN, 4)`` float32 on the model's device."""
        torch = self.torch
        mats = []
        for s in seqs:
            t = self._onehot_fn(s)
            if not torch.is_tensor(t):
                t = torch.as_tensor(np.asarray(t))
            if t.ndim == 2:              # (L, 4) -> (1, L, 4)
                t = t.unsqueeze(0)
            elif t.ndim != 3:
                raise ValueError(f"one-hot has ndim {t.ndim}, expected 2 or 3")
            mats.append(t)
        x = torch.cat(mats, dim=0).to(device=self.device, dtype=torch.float32)
        if x.shape[1:] != (F.TARGET_LEN, 4):
            raise ValueError(
                f"one-hot is {tuple(x.shape)}, expected (B, {F.TARGET_LEN}, 4)")
        return x

    # -------------------------------------------------------------- forward

    def _forward(self, seqs: list[str], *, tracks: tuple[int, ...]) -> np.ndarray:
        """``(B, len(tracks), 2)`` -- usage at (acceptor, donor) per track."""
        torch = self.torch
        x = self._onehot(seqs)
        out = self.model.predict(x, 0, heads=(HEAD,))

        if HEAD not in out:
            raise KeyError(
                f"predict(heads=('{HEAD}',)) returned keys {sorted(out)} "
                f"without '{HEAD}'")
        block = out[HEAD]
        if not isinstance(block, dict) or "predictions" not in block:
            raise KeyError(
                f"'{HEAD}' is {type(block).__name__} with "
                f"{sorted(block) if isinstance(block, dict) else '?'}; "
                "expected a dict carrying 'predictions'")
        pred = block["predictions"]                     # (B, 16384, 734)

        if pred.shape[1] != F.TARGET_LEN:
            raise ValueError(
                f"predictions has {pred.shape[1]} positions, expected "
                f"{F.TARGET_LEN}; the output is cropped and the padded "
                "acceptor/donor indices would be wrong")
        if pred.shape[2] != F.N_USAGE_TRACKS:
            raise ValueError(
                f"predictions has {pred.shape[2]} tracks, expected "
                f"{F.N_USAGE_TRACKS}; the catalogue does not describe it")

        # Slice on the GPU: the full tensor is 48 MB per sequence and we want
        # 2 floats per track from it.
        idx_t = torch.as_tensor(tracks, device=pred.device, dtype=torch.long)
        idx_p = torch.as_tensor([self.acc_pos, self.don_pos],
                                device=pred.device, dtype=torch.long)
        sub = pred.index_select(1, idx_p).index_select(2, idx_t)  # (B, 2, T)
        return sub.permute(0, 2, 1).float().cpu().numpy()        # (B, T, 2)

    def predict_heads(self, seqs: list[str], heads: tuple[str, ...]) -> dict:
        """Raw output dict for ``heads``, tensors left on the device.

        The low-level escape hatch for analyses that need whole per-position
        tracks rather than two positions -- the cryptic-site scan, chiefly.
        Callers slice on the device and move only the summary to the host.
        """
        return self.model.predict(self._onehot(list(seqs)), 0, heads=heads)

    def predict_junctions(self, seqs: list[str], positions):
        """``splice_junctions`` evaluated at CALLER-SUPPLIED positions.

        ``positions`` is (B, 4, P) of absolute indices into the 16,384 axis,
        in the head's own row order:
            row 0 pos_donor, row 1 pos_acceptor,
            row 2 neg_donor, row 3 neg_acceptor
        Negative entries are padding; the head builds its mask from
        ``idx >= 0`` and zeroes those cells, so -1 is the safe filler.

        Supplying positions makes the model SKIP its own top-512 candidate
        generation (model.py:777-799). That is the whole trick: pred_counts
        is (B, P, P, 2*367), so P=8 costs 188 KB per sequence where the
        default P=512 costs 779 MB. The last axis is
        ``cat([pos_counts, neg_counts])``, i.e. [367 plus][367 minus] --
        the same layout as the usage head, so the plus-strand tissue index
        is reused and its minus twin is +367.

        Values are ``softplus`` outputs: non-negative unbounded counts, not
        probabilities. Only ratios between them are meaningful.
        """
        torch = self.torch
        x = self._onehot(list(seqs))
        pos = torch.as_tensor(positions, dtype=torch.long, device=x.device)
        if pos.ndim != 3 or pos.shape[1] != 4:
            raise ValueError(
                f"positions must be (B, 4, P), got {tuple(pos.shape)}")
        if pos.shape[0] != x.shape[0]:
            raise ValueError(
                f"positions batch {pos.shape[0]} != {x.shape[0]} sequences")
        out = self.model.predict(x, 0, heads=("splice_junctions",),
                                 splice_site_positions=pos)
        blk = out["splice_junctions"]
        pc = blk["pred_counts"]
        P = pos.shape[-1]
        if tuple(pc.shape[1:3]) != (P, P):
            raise ValueError(
                f"pred_counts is {tuple(pc.shape)}, expected (B,{P},{P},*) -- "
                f"splice_site_positions was probably ignored")
        if pc.shape[-1] != 2 * N_JUNCTION_TISSUES:
            raise ValueError(
                f"pred_counts last axis {pc.shape[-1]} != "
                f"2 x {N_JUNCTION_TISSUES}")
        return blk

    def usage(self, seqs: list[str]) -> np.ndarray:
        """``(B, 2)`` plus-strand usage at (acceptor, donor), OOM-adaptive.

        Accepts any number of sequences and returns exactly that many rows.
        An OOM lowers ``batch_size`` for the rest of the run and retries the
        same sequences in smaller chunks -- it never drops the tail, which is
        the difference between a slower run and silently missing labels.
        """
        seqs = list(seqs)
        chunks: list[np.ndarray] = []
        i = 0
        while i < len(seqs):
            size = min(self.batch_size, len(seqs) - i)
            try:
                got = self._forward(seqs[i:i + size],
                                    tracks=(self.track_index,))
            except self.torch.cuda.OutOfMemoryError:
                self.torch.cuda.empty_cache()
                nxt = next((b for b in BATCH_LADDER if b < size), None)
                if nxt is None:
                    raise
                print(f"[local] CUDA OOM at batch {size}; dropping to {nxt} "
                      "for the rest of the run", flush=True)
                self.batch_size = nxt
                continue                      # same i, smaller chunk
            chunks.append(got[:, 0, :])
            i += size
            if not self._size_logged:
                self._size_logged = True
                print(f"[local] first batch of {size} OK", flush=True)
        return np.concatenate(chunks, axis=0)

    def score(self, seqs: list[str]) -> list[dict[str, float]]:
        """One record per sequence, with ``logit_average`` as the target."""
        u = self.usage(list(seqs))
        return [{"AG_acceptor_usage": float(a),
                 "AG_donor_usage": float(d),
                 "AG_mean_usage": (float(a) + float(d)) / 2.0,
                 "logit_average": F.logit_average(float(a), float(d))}
                for a, d in u]

    # ----------------------------------------------------------- provenance

    def provenance(self) -> dict:
        import alphagenome_pytorch as apt
        pol = getattr(self.model, "dtype_policy", None)
        return {
            "backend": "local",
            "package_version": getattr(apt, "__version__", "unknown"),
            "weights": self.weights,
            "weights_sha256_prefix": _sha_prefix(self.weights),
            "catalog_sha256": F.CATALOG_SHA256,
            "track_index_plus": self.track_index,
            "track_index_minus": self.track_index_minus,
            "ontology": F.ONTOLOGY[0],
            "head": HEAD,
            "acceptor_padded": self.acc_pos,
            "donor_padded": self.don_pos,
            "target_len": F.TARGET_LEN,
            "batch_size": self.batch_size,
            "compute_dtype": str(getattr(pol, "compute_dtype", "unknown")),
            "device": self.device,
        }


def _sha_prefix(path: str | Path, nbytes: int = 1 << 20) -> str:
    """sha256 of the first MiB -- a cheap fingerprint for a 921 MB file."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(nbytes))
    return h.hexdigest()[:16]


# ==========================================================================
# preflight: the correctness gate that runs INSIDE the scoring job
# ==========================================================================

def preflight(scorer: LocalScorer, *, strict: bool = True) -> dict:
    """Score the native construct and check the index, strand and scale.

    This is deliberately part of the scoring job rather than a separate
    submission: it costs one forward pass and it is the only thing standing
    between a wrong track index and 1.7M wrong labels.

    Checks, in order of how badly each would corrupt the run:

    1. usage is in [0, 1] -- i.e. ``predictions``, not ``logits``
    2. the PLUS track exceeds its MINUS twin at both the acceptor and the
       donor -- the construct is built 5'->3', so if the minus track were
       higher we would have the strand backwards
    3. both positions stand out against the construct's own background --
       they are real splice sites in a working minigene, so they should not
       look like arbitrary positions
    """
    native = F.build_construct(F.load_context()[F.GEOMETRY["b1_start"]:
                                                F.GEOMETRY["b1_end"]],
                               _native_syn70())
    padded, _, _ = F.center_pad(native)

    both = scorer._forward([padded],
                           tracks=(scorer.track_index,
                                   scorer.track_index_minus))[0]
    (acc_p, don_p), (acc_m, don_m) = both[0], both[1]
    la = F.logit_average(acc_p, don_p)

    # background: the whole plus track for this one construct
    full = scorer.model.predict(scorer._onehot([padded]), 0, heads=(HEAD,))
    col = full[HEAD]["predictions"][0, :, scorer.track_index].float().cpu().numpy()
    rank_acc = int((col > col[scorer.acc_pos]).sum()) + 1
    rank_don = int((col > col[scorer.don_pos]).sum()) + 1

    report = {
        "native_acceptor_plus": float(acc_p),
        "native_donor_plus": float(don_p),
        "native_acceptor_minus": float(acc_m),
        "native_donor_minus": float(don_m),
        "native_logit_average": float(la),
        "track_max": float(col.max()),
        "track_median": float(np.median(col)),
        "acceptor_rank_of_16384": rank_acc,
        "donor_rank_of_16384": rank_don,
    }

    print("\n" + "=" * 70)
    print("PREFLIGHT -- native construct, track "
          f"{scorer.track_index} (+) vs {scorer.track_index_minus} (-)")
    print("=" * 70)
    print(f"  acceptor  plus {acc_p:.6f}   minus {acc_m:.6f}")
    print(f"  donor     plus {don_p:.6f}   minus {don_m:.6f}")
    print(f"  logit_average  {la:+.6f}")
    print(f"  track median {np.median(col):.3e}  max {col.max():.6f}")
    print(f"  acceptor ranks {rank_acc:,} of {F.TARGET_LEN:,}")
    print(f"  donor    ranks {rank_don:,} of {F.TARGET_LEN:,}")

    problems = []
    for nm, v in (("acceptor", acc_p), ("donor", don_p)):
        if not 0.0 <= v <= 1.0:
            problems.append(f"{nm} usage {v} outside [0,1] -- reading logits?")
    if acc_p <= acc_m:
        problems.append(
            f"acceptor: minus track ({acc_m:.6f}) >= plus ({acc_p:.6f}) "
            "-- strand looks inverted")
    if don_p <= don_m:
        problems.append(
            f"donor: minus track ({don_m:.6f}) >= plus ({don_p:.6f}) "
            "-- strand looks inverted")
    if rank_acc > 200 or rank_don > 200:
        problems.append(
            f"acceptor/donor rank {rank_acc}/{rank_don} of {F.TARGET_LEN} -- "
            "the annotated sites are not standing out; geometry or track "
            "index is suspect")

    report["problems"] = problems
    if problems:
        print("\n  FAILED:")
        for p in problems:
            print(f"    - {p}")
        if strict:
            raise RuntimeError(
                f"preflight failed ({len(problems)} problem(s)); refusing to "
                "score. Re-run with --no-strict-preflight to override.")
    else:
        print("\n  PASSED")
    print("=" * 70 + "\n", flush=True)
    return report


def _native_syn70() -> str:
    exons = pd.read_csv(_THIS_DIR.parent / "outputs" / "upstream_exon"
                        / "b1_pilot_exons.csv")
    return dict(zip(exons.sequence_id, exons.synthetic_70mer))[F.PRIMARY_EXON]


# ==========================================================================
# the batched resumable loop
# ==========================================================================

def run_scoring_local(jobs, out_csv: Path, scorer: LocalScorer, *,
                      flush_every: int = 2_000,
                      failed_csv: Path | None = None,
                      progress_every: int = 2_000,
                      skip_ids: set[str] | None = None,
                      done_before: int = 0, shard_total: int = 0) -> int:
    """Score ``jobs`` in batches, APPENDING to ``out_csv``; resume on row_id.

    Same on-disk contract as ``fullctx_lib.run_scoring`` -- append-mode writes
    (not the O(n^2) full rewrite), resume by ``row_id``, failures appended to
    a sidecar -- so the two are interchangeable from the caller's side and a
    run started on one backend can be finished on the other.

    A failed batch is retried once per sequence, so one bad row costs its
    batch-mates nothing.
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
    if not shard_total:
        print(f"[scoring] {total:,} of {len(jobs):,} rows to call", flush=True)
    if not total:
        return 0

    buf: list[dict] = []
    written = 0
    n_failed = 0
    t0 = time.time()

    def flush() -> None:
        nonlocal written
        if not buf:
            return
        pd.DataFrame(buf).to_csv(out_csv, mode="a", index=False,
                                 header=not out_csv.exists())
        written += len(buf)
        buf.clear()

    def record_failure(job: dict, exc: BaseException) -> None:
        nonlocal n_failed
        n_failed += 1
        rec = {k: v for k, v in job.items() if k != "seq"}
        rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
        if failed_csv:
            pd.DataFrame([rec]).to_csv(failed_csv, mode="a", index=False,
                                       header=not failed_csv.exists())

    i = 0
    while i < total:
        batch = todo[i:i + scorer.batch_size]
        try:
            results = scorer.score([j["seq"] for j in batch])
        except Exception as exc:                          # noqa: BLE001
            # Isolate: one unparseable row should not lose its batch-mates.
            print(f"  [batch error] {type(exc).__name__}: {exc} -- "
                  f"retrying {len(batch)} rows singly", flush=True)
            results = []
            for job in batch:
                try:
                    results.append(scorer.score([job["seq"]])[0])
                except Exception as exc2:                 # noqa: BLE001
                    record_failure(job, exc2)
                    results.append(None)

        for job, res in zip(batch, results):
            if res is None:
                continue
            rec = {k: v for k, v in job.items() if k != "seq"}
            rec.update(res)
            buf.append(rec)

        i += len(batch)
        if len(buf) >= flush_every:
            flush()
        # Report after the FIRST batch too, not just every flush: the whole
        # point of a self-measuring run is seeing seq/s within seconds, so a
        # bad rate can be killed instead of discovered hours later.
        if progress_every and (i <= scorer.batch_size
                               or i % progress_every < scorer.batch_size):
            el = time.time() - t0
            rate = i / max(el, 1e-9)
            did = done_before + i
            tot = shard_total or total
            print(f"  [{did:,}/{tot:,}] {rate:.2f} seq/s | "
                  f"elapsed {el / 3600:.2f}h | "
                  f"ETA {(tot - did) / max(rate, 1e-9) / 3600:.2f}h",
                  flush=True)

    flush()
    el = time.time() - t0
    if not shard_total:
        print(f"\n[done] wrote {written:,} new rows -> {out_csv} in "
              f"{el / 3600:.2f}h ({written / max(el, 1e-9):.2f} seq/s)",
              flush=True)
    if n_failed:
        print(f"[warn] {n_failed:,} failures -> {failed_csv}", flush=True)
    return written


def write_provenance(scorer: LocalScorer, path: Path,
                     extra: dict | None = None) -> None:
    rec = scorer.provenance()
    rec.update(extra or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2, default=str) + "\n")
    print(f"[provenance] -> {path}", flush=True)
