#!/usr/bin/env python
"""Map the local alphagenome_pytorch model onto our scoring contract.

The weights at /scratch/yl12316/models/alphagenome are the PyTorch port
(alphagenome-pytorch, weights converted from the official JAX checkpoint).
It differs from the gRPC client in three ways that matter to us:

  * ``predict(dna_sequence, organism_index, ...)`` takes a ONE-HOT TENSOR,
    not a string, and has no ``ontology_terms`` argument;
  * ``splice_site_usage`` returns ALL 734 tracks, so the CL:0002518 track has
    to be selected by index rather than requested from a server;
  * ``splice_sites`` is a 5-class head (D+, A+, D-, A-, none) rather than the
    API's named donor/acceptor channels.

This script answers, in one run and without assuming any of it:

  1. what ``predict`` actually returns -- keys, shapes, dtypes, resolutions;
  2. whether 16,384 is an accepted input length (our context is 8,612 padded);
  3. where the track metadata lives and which index is CL:0002518;
  4. whether the designed acceptor/donor are local maxima of that usage track;
  5. what logit-average the local model gives for the native construct, so it
     can be compared against the API's value for the same sequence.

Nothing is written outside ``outputs/upstream_exon/fullctx_probe/``.

    python upstream_exon/probe_local_alphagenome.py --dry-run   # no model load
    python upstream_exon/probe_local_alphagenome.py
    python upstream_exon/probe_local_alphagenome.py --device cpu
"""
from __future__ import annotations

import argparse
import json
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

#: track indices matching the target ontology, filled in from the
#: built-in TrackMetadataCatalog once it is loaded
ONTOLOGY_IDX: list[int] = []
DEFAULT_WEIGHTS = Path("/scratch/yl12316/models/alphagenome/"
                       "model_all_folds.safetensors")
TARGET_ONTOLOGY = "CL:0002518"


# ==========================================================================
# one-hot encoding
# ==========================================================================

def find_onehot_fn():
    """Locate the package's own one-hot helper, whatever it is called."""
    import importlib
    candidates = [
        ("alphagenome_pytorch", "sequence_to_onehot_tensor"),
        ("alphagenome_pytorch.utils", "sequence_to_onehot_tensor"),
        ("alphagenome_pytorch.data", "sequence_to_onehot_tensor"),
        ("alphagenome_pytorch.utils", "seq_to_onehot"),
        ("alphagenome_pytorch.utils", "one_hot_encode"),
        ("alphagenome_pytorch.data", "one_hot_encode"),
    ]
    for mod, name in candidates:
        try:
            m = importlib.import_module(mod)
        except Exception:                                   # noqa: BLE001
            continue
        fn = getattr(m, name, None)
        if callable(fn):
            return fn, f"{mod}.{name}"
    return None, None


def manual_onehot(seq: str) -> np.ndarray:
    """ACGT one-hot, N as all-zeros. (L, 4) float32.

    Used only if the package exposes no helper. The ACGT column order and
    the all-zero encoding of N are the AlphaGenome convention; if the port
    disagrees the probe's own argmax checks below will show it.
    """
    idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    out = np.zeros((len(seq), 4), dtype=np.float32)
    for i, b in enumerate(seq):
        j = idx.get(b)
        if j is not None:
            out[i, j] = 1.0
    return out


# ==========================================================================
# output introspection
# ==========================================================================

def walk(obj, name: str, depth: int = 0, max_depth: int = 4,
         lines: list[str] | None = None) -> list[str]:
    """Print the structure of whatever predict() returned."""
    import torch
    lines = [] if lines is None else lines
    pad = "  " * depth
    if isinstance(obj, torch.Tensor):
        t = obj.detach().float()
        lines.append(f"{pad}{name}: Tensor {tuple(obj.shape)} {obj.dtype} "
                     f"min={t.min().item():.6g} max={t.max().item():.6g}")
        return lines
    if isinstance(obj, np.ndarray):
        lines.append(f"{pad}{name}: ndarray {obj.shape} {obj.dtype}")
        return lines
    if isinstance(obj, dict):
        lines.append(f"{pad}{name}: dict({len(obj)}) keys={list(obj)[:12]}")
        if depth < max_depth:
            for k, v in list(obj.items())[:24]:
                walk(v, str(k), depth + 1, max_depth, lines)
        return lines
    if isinstance(obj, (list, tuple)):
        lines.append(f"{pad}{name}: {type(obj).__name__}({len(obj)})")
        if depth < max_depth:
            for i, v in enumerate(obj[:4]):
                walk(v, f"[{i}]", depth + 1, max_depth, lines)
        return lines
    if isinstance(obj, (str, int, float, bool, type(None))):
        lines.append(f"{pad}{name}: {type(obj).__name__} = {obj!r}")
        return lines
    lines.append(f"{pad}{name}: {type(obj).__name__}")
    if depth < max_depth:
        for a in sorted(x for x in dir(obj) if not x.startswith("_")):
            try:
                v = getattr(obj, a)
            except Exception:                               # noqa: BLE001
                continue
            if callable(v):
                continue
            walk(v, a, depth + 1, max_depth, lines)
    return lines


def hunt_metadata(sp: Path) -> list[Path]:
    """Any metadata table shipped inside the installed package."""
    pats = ("*.csv", "*.tsv", "*.parquet", "*.json", "*.txt", "*.npy", "*.npz")
    hits: list[Path] = []
    for p in pats:
        hits += [f for f in sp.rglob(p)
                 if "__pycache__" not in str(f) and f.stat().st_size > 200]
    return sorted(hits, key=lambda f: -f.stat().st_size)[:25]


def grep_ontology(sp: Path, term: str, max_hits: int = 12) -> list[str]:
    """Which files in the installed package mention the ontology at all.

    The first probe found no shipped CSV/JSON, so the 734-track ordering is
    not a data file. It is then either embedded in Python, produced by
    ``named_outputs``, or fetched at run time -- and a plain text search over
    the package says which.
    """
    out: list[str] = []
    for f in sp.rglob("*"):
        if not f.is_file() or "__pycache__" in str(f):
            continue
        if f.stat().st_size > 50_000_000:
            continue
        try:
            txt = f.read_text(errors="ignore")
        except Exception:                                   # noqa: BLE001
            continue
        if term in txt:
            n = txt.count(term)
            out.append(f"{f.relative_to(sp)}  ({n} occurrence(s))")
            if len(out) >= max_hits:
                break
    return out


def describe_named_outputs(sp: Path, n_tracks: int | None) -> None:
    """What the package's own metadata machinery exposes."""
    import importlib
    print("\n--- alphagenome_pytorch.named_outputs ---")
    try:
        N = importlib.import_module("alphagenome_pytorch.named_outputs")
    except Exception as e:                                  # noqa: BLE001
        print(f"  not importable: {type(e).__name__}: {e}")
        return
    names = [a for a in dir(N) if not a.startswith("_")]
    print(f"  members: {names}")
    for a in names:
        try:
            v = getattr(N, a)
        except Exception:                                   # noqa: BLE001
            continue
        # anything sized like the track axis is a strong candidate
        try:
            ln = len(v)
        except Exception:                                   # noqa: BLE001
            continue
        if n_tracks and ln == n_tracks:
            print(f"  *** {a}: len == n_tracks ({ln}) -- likely the "
                  f"track table")
            try:
                print(f"      first 3: {list(v)[:3]}")
            except Exception:                               # noqa: BLE001
                pass
        elif ln in (5, 734, 768, 256, 384, 640):
            print(f"  {a}: len {ln}")


def find_ontology_tracks(meta, ontology: str) -> list[int]:
    """Row indices of a metadata table whose any column mentions ``ontology``."""
    if not isinstance(meta, pd.DataFrame):
        return []
    mask = pd.Series(False, index=meta.index)
    for c in meta.columns:
        try:
            mask |= meta[c].astype(str).str.contains(ontology, case=False,
                                                     na=False, regex=False)
        except Exception:                                   # noqa: BLE001
            continue
    return list(np.flatnonzero(mask.to_numpy()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--device", default=None,
                    help="cuda / cpu; default cuda when available")
    ap.add_argument("--organism-index", type=int, default=0,
                    help="0 = human, 1 = mouse")
    ap.add_argument("--ontology", default=TARGET_ONTOLOGY)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--dry-run", action="store_true",
                    help="report geometry and package layout, load no weights")
    args = ap.parse_args()

    g = F.GEOMETRY
    left_pad, acc_pos, don_pos = F.padded_positions()
    print("=" * 74)
    print("LOCAL ALPHAGENOME PROBE")
    print("=" * 74)
    print(f"  context {F.CONTEXT_LEN:,} nt -> padded {F.TARGET_LEN:,} "
          f"(left_pad {left_pad:,})")
    print(f"  acceptor {g['acceptor']} -> padded {acc_pos:,}")
    print(f"  donor    {g['donor']} -> padded {don_pos:,}")
    print(f"  weights  {args.weights}")

    try:
        import alphagenome_pytorch as AP
    except ImportError as e:
        sys.exit(f"FATAL: alphagenome_pytorch not importable ({e}).\n"
                 f"       Run this with the env that has it, e.g.\n"
                 f"         /scratch/yl12316/envs/alphagenome/bin/python "
                 f"{Path(__file__).name}")
    sp = Path(AP.__file__).parent
    print(f"  package  {sp}  (v{getattr(AP, '__version__', '?')})")

    onehot_fn, onehot_name = find_onehot_fn()
    print(f"  one-hot  {onehot_name or 'NOT FOUND -> using manual ACGT/N=0'}")

    print("\n--- metadata tables shipped with the package ---")
    tables = hunt_metadata(sp)
    for f in tables:
        print(f"  {f.stat().st_size / 1e3:9.1f} kB  {f.relative_to(sp)}")
    if not tables:
        print("  (none found)")

    print(f"\n--- files mentioning {args.ontology} ---")
    hits = grep_ontology(sp, args.ontology)
    for h in hits:
        print(f"  {h}")
    if not hits:
        print(f"  (none -- {args.ontology} does not appear anywhere in the "
              f"installed package)")
        generic = grep_ontology(sp, "CL:", max_hits=8)
        print("  files mentioning any 'CL:' ontology term:")
        for h in generic:
            print(f"    {h}")
        if not generic:
            print("    (none -- the track ordering is not shipped as text; "
                  "it must come from named_outputs, the checkpoint, or the "
                  "API's output_metadata)")

    describe_named_outputs(sp, None)

    # The package was installed as alphagenome-pytorch[hf] with
    # huggingface_hub, so the track metadata may be fetched from the Hub
    # rather than shipped -- which would explain finding nothing above. Any
    # such file is cached on disk, and compute nodes may have no outbound
    # network, so what is already cached is what we get.
    print("\n--- huggingface cache ---")
    import os
    for var in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE",
                "HF_HUB_OFFLINE", "TRANSFORMERS_CACHE"):
        if os.environ.get(var):
            print(f"  {var}={os.environ[var]}")
    cands = [Path(os.environ.get("HF_HOME", "")) / "hub" if os.environ.get("HF_HOME") else None,
             Path(os.environ.get("HF_HUB_CACHE", "")) if os.environ.get("HF_HUB_CACHE") else None,
             Path.home() / ".cache" / "huggingface" / "hub",
             Path("/scratch/yl12316/.cache/huggingface/hub"),
             Path("/scratch/yl12316/models/alphagenome/.cache")]
    seen_any = False
    for c in cands:
        if c is None or not c.exists():
            continue
        seen_any = True
        print(f"  {c}:")
        for f in sorted(c.rglob("*"))[:200]:
            if f.is_file() and f.suffix.lower() in (
                    ".csv", ".tsv", ".json", ".parquet", ".txt", ".yaml", ".yml"):
                print(f"    {f.stat().st_size / 1e3:9.1f} kB  "
                      f"{f.relative_to(c)}")
    if not seen_any:
        print("  (no cache directory found at the usual locations)")

    # the native construct, so the number is comparable to the API probe
    import splice_site_partition as SP
    exons = pd.read_csv(ROOT / "outputs" / "upstream_exon" / "b1_pilot_exons.csv")
    syn = dict(zip(exons.sequence_id, exons.synthetic_70mer))[F.PRIMARY_EXON]
    native_b1 = SP.pnas_blocks8()["B1"]
    core = F.build_construct(native_b1, syn)
    fails = F.validate_construct(core, native_b1, syn)
    if fails:
        sys.exit(f"FATAL: native construct failed validation: {fails}")

    # F.center_pad, not the API pipeline's: that module needs dotenv, which
    # the alphagenome env does not have and a local run does not want.
    padded, lp, _ = F.center_pad(core, target_len=F.TARGET_LEN)
    assert lp == left_pad and len(padded) == F.TARGET_LEN
    print(f"\n  native construct built and validated "
          f"({len(padded):,} nt, {padded.count('N')} N)")

    if args.dry_run:
        print("\n--dry-run: no weights loaded")
        return 0

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- loading weights on {device} ---", flush=True)
    if not args.weights.exists():
        sys.exit(f"FATAL: {args.weights} not found")
    model = AP.AlphaGenome.from_pretrained(args.weights, device=device)
    model.eval()
    print("  loaded")

    if onehot_fn is not None:
        x = onehot_fn(padded)
        x = torch.as_tensor(x) if not isinstance(x, torch.Tensor) else x
    else:
        x = torch.from_numpy(manual_onehot(padded))
    if x.ndim == 2:
        x = x.unsqueeze(0)
    x = x.to(device)
    print(f"  input tensor {tuple(x.shape)} {x.dtype}")

    print("\n--- predict(named_outputs=False) ---", flush=True)
    with torch.no_grad():
        out = model.predict(x, organism_index=args.organism_index)
    for line in walk(out, "out"):
        print(line)

    # ---- the built-in track metadata catalog -----------------------------
    # The first run failed named_outputs=True with "requires a metadata
    # catalog"; the catalog ships with the package but is not a loose CSV,
    # and the earlier grep missed it because ontology ids are written
    # CL_0002518 (underscore), not CL:0002518.
    print("\n--- TrackMetadataCatalog.load_builtin() ---")
    catalog = None
    try:
        from alphagenome_pytorch.named_outputs import TrackMetadataCatalog
        for kw in ({}, {"organism": args.organism_index}, {"organism": 0}):
            try:
                catalog = TrackMetadataCatalog.load_builtin(**kw)
                print(f"  load_builtin({kw}) -> OK: {type(catalog).__name__}")
                break
            except Exception as e:                          # noqa: BLE001
                print(f"  load_builtin({kw}) -> {type(e).__name__}: {e}")
    except Exception as e:                                  # noqa: BLE001
        print(f"  import failed: {type(e).__name__}: {e}")

    if catalog is not None:
        print(f"  members: {[a for a in dir(catalog) if not a.startswith('_')]}")
        for attr in ("tracks", "by_output", "outputs", "metadata", "entries",
                     "catalog", "table"):
            v = getattr(catalog, attr, None)
            if v is None or callable(v):
                continue
            try:
                print(f"  {attr}: {type(v).__name__} len={len(v)}")
                if isinstance(v, dict):
                    print(f"    keys: {list(v)[:15]}")
            except Exception:                               # noqa: BLE001
                pass
        # try to pull the splice_site_usage track list out of it
        rows = None
        for getter in ("get", "tracks_for", "for_output", "__getitem__"):
            fn = getattr(catalog, getter, None)
            if not callable(fn):
                continue
            try:
                rows = fn("splice_site_usage")
                print(f"  {getter}('splice_site_usage') -> "
                      f"{type(rows).__name__} len={len(rows)}")
                break
            except Exception:                               # noqa: BLE001
                rows = None
        if rows is None:
            for attr in ("tracks", "by_output", "outputs", "metadata"):
                v = getattr(catalog, attr, None)
                if isinstance(v, dict) and "splice_site_usage" in v:
                    rows = v["splice_site_usage"]
                    print(f"  {attr}['splice_site_usage'] -> len={len(rows)}")
                    break
        if rows is not None:
            try:
                r0 = list(rows)[0]
                print(f"  first track: {r0}")
                flds = [f for f in dir(r0) if not f.startswith('_')
                        and not callable(getattr(r0, f, None))]
                print(f"  track fields: {flds}")
            except Exception as e:                          # noqa: BLE001
                print(f"  could not inspect a row: {e}")
            # the actual lookup, tolerant of ':' vs '_'
            wants = {args.ontology, args.ontology.replace(":", "_")}
            hits = []
            for i, r in enumerate(rows):
                blob = str(r)
                if any(w in blob for w in wants):
                    hits.append(i)
            print(f"\n  tracks matching {sorted(wants)}: {hits[:20]} "
                  f"({len(hits)} of {len(rows)})")
            for i in hits[:6]:
                print(f"    [{i}] {list(rows)[i]}")
            if hits:
                ONTOLOGY_IDX.extend(hits)

        for setter in ("set_track_metadata_catalog", "set_catalog"):
            fn = getattr(model, setter, None)
            if callable(fn):
                try:
                    fn(catalog)
                    print(f"  model.{setter}(catalog) -> OK")
                    break
                except Exception as e:                      # noqa: BLE001
                    print(f"  model.{setter} -> {type(e).__name__}: {e}")

    named = None
    print("\n--- predict(named_outputs=True) ---", flush=True)
    try:
        with torch.no_grad():
            named = model.predict(x, organism_index=args.organism_index,
                                  named_outputs=True)
        for line in walk(named, "named")[:60]:
            print(line)
    except Exception as e:                                  # noqa: BLE001
        print(f"  named_outputs=True raised {type(e).__name__}: {e}")

    # Does the NamedOutputs object itself carry the ontology labels?
    if named is not None:
        print(f"\n--- hunting {args.ontology} inside NamedOutputs ---")
        found = False

        def scan(o, path="named", depth=0):
            nonlocal found
            if depth > 5 or found:
                return
            if isinstance(o, str):
                if args.ontology in o:
                    print(f"  HIT at {path}: {o[:120]}")
                    found = True
                return
            if isinstance(o, dict):
                for k, v in list(o.items())[:64]:
                    scan(v, f"{path}[{k!r}]", depth + 1)
                return
            if isinstance(o, (list, tuple)):
                hits = [i for i, v in enumerate(o[:2000])
                        if isinstance(v, str) and args.ontology in v]
                if hits:
                    print(f"  HIT at {path}: indices {hits[:10]} of "
                          f"{len(o)} entries")
                    print(f"       e.g. {o[hits[0]]!r}")
                    found = True
                    return
                for i, v in enumerate(o[:8]):
                    scan(v, f"{path}[{i}]", depth + 1)
                return
            if hasattr(o, "__dict__"):
                for k, v in list(vars(o).items())[:64]:
                    scan(v, f"{path}.{k}", depth + 1)

        try:
            scan(named)
        except Exception as e:                              # noqa: BLE001
            print(f"  scan raised {type(e).__name__}: {e}")
        if not found:
            print(f"  {args.ontology} not present in NamedOutputs")

    # safetensors files carry a free-form __metadata__ header; the track
    # ordering may have been stashed there at conversion time.
    print("\n--- safetensors header metadata ---")
    try:
        from safetensors import safe_open
        with safe_open(str(args.weights), framework="pt") as f:
            md = f.metadata() or {}
        if md:
            for k, v in list(md.items())[:20]:
                print(f"  {k}: {str(v)[:160]}")
        else:
            print("  (empty)")
    except Exception as e:                                  # noqa: BLE001
        print(f"  unavailable: {type(e).__name__}: {e}")

    # ---- locate the usage tensor -----------------------------------------
    def dig(o, key):
        if isinstance(o, dict):
            if key in o:
                return o[key]
            for v in o.values():
                r = dig(v, key)
                if r is not None:
                    return r
        return getattr(o, key, None) if not isinstance(o, dict) else None

    usage = dig(out, "splice_site_usage")
    if usage is None:
        print("\nFATAL: no 'splice_site_usage' key in the output. The keys "
              "printed above are the ground truth -- tell me which one holds "
              "the usage head and I will wire that name in.")
        return 1
    def pick_tensor(obj, path="splice_site_usage", depth=0):
        """The tensor inside however many layers of dict the head returns.

        Guessing key names failed once already ('1bp'/'preds'/... matched
        nothing), so prefer a tensor whose shape actually contains the input
        length, and otherwise take the largest tensor present -- then SAY
        which was chosen, so a wrong pick is visible rather than silent.
        """
        import torch as _t
        if isinstance(obj, _t.Tensor):
            return obj, path
        if isinstance(obj, (list, tuple)) and obj:
            return pick_tensor(obj[0], f"{path}[0]", depth + 1)
        if isinstance(obj, dict):
            print(f"  {path} is a dict with keys {list(obj)}")
            if depth > 3:
                return None, path
            # 'predictions' holds probabilities in [0,1]; 'logits' is the
            # same shape and would pass every check below while silently
            # feeding logit_average the wrong scale. Order matters.
            for pref in ("predictions", "probs", "probabilities"):
                if pref in obj:
                    print(f"  -> preferring ['{pref}'] over {list(obj)}")
                    return pick_tensor(obj[pref], f"{path}[{pref!r}]",
                                       depth + 1)
            best = None
            for k, v in obj.items():
                if k == "logits" and len(obj) > 1:
                    continue      # only fall back to logits if nothing else
                t, p = pick_tensor(v, f"{path}[{k!r}]", depth + 1)
                if t is None:
                    continue
                if F.TARGET_LEN in tuple(t.shape):
                    return t, p            # exact input-length match wins
                if best is None or t.numel() > best[0].numel():
                    best = (t, p)
            return best if best else (None, path)
        return None, path

    usage, upath = pick_tensor(usage)
    if usage is None:
        print("\nFATAL: could not find a tensor under 'splice_site_usage'. "
              "The keys printed above are the ground truth -- tell me which "
              "one holds the per-base usage and I will wire that name in.")
        return 1
    print(f"  -> using {upath}  shape {tuple(usage.shape)}")
    u = usage.detach().float().cpu().numpy()
    print(f"\n  usage tensor {u.shape}")
    u = u[0] if u.ndim == 3 else u
    if u.shape[0] < u.shape[1]:          # (T, S) -> (S, T)
        print(f"  transposing {u.shape} -> {(u.shape[1], u.shape[0])}")
        u = u.T
    n_pos, n_tracks = u.shape
    print(f"  interpreted as (positions={n_pos:,}, tracks={n_tracks})")
    if n_pos != F.TARGET_LEN:
        print(f"  WARNING: {n_pos:,} positions != input {F.TARGET_LEN:,}; "
              f"the head may crop padding (try include_padding=True)")

    # ---- which track is the target ontology? -----------------------------
    print(f"\n--- locating ontology {args.ontology} among {n_tracks} tracks ---")
    idxs: list[int] = []
    meta_src = None
    for f in hunt_metadata(sp):
        if f.suffix.lower() not in (".csv", ".tsv", ".parquet"):
            continue
        try:
            m = (pd.read_parquet(f) if f.suffix.lower() == ".parquet"
                 else pd.read_csv(f, sep="\t" if f.suffix.lower() == ".tsv"
                                  else ","))
        except Exception:                                   # noqa: BLE001
            continue
        if len(m) != n_tracks:
            continue
        hit = find_ontology_tracks(m, args.ontology)
        if hit:
            idxs, meta_src = hit, f
            print(f"  {f.relative_to(sp)}: {len(m)} rows, "
                  f"{len(hit)} match -> indices {hit[:10]}")
            print(m.iloc[hit].head(6).to_string())
            break
    if not idxs and ONTOLOGY_IDX:
        idxs = list(ONTOLOGY_IDX)
        meta_src = "TrackMetadataCatalog.load_builtin()"
        print(f"  using the built-in catalog: indices {idxs[:10]}")
    if not idxs:
        print(f"  no shipped table of length {n_tracks} mentions "
              f"{args.ontology}.")
        print(f"  Tables of the right length are the candidates; if none, the "
              f"ordering must come from the API's output_metadata.")

    report = {
        "weights": str(args.weights), "device": device,
        "package_version": getattr(AP, "__version__", None),
        "onehot_fn": onehot_name,
        "usage_shape": [int(n_pos), int(n_tracks)],
        "acceptor_padded": acc_pos, "donor_padded": don_pos,
        "ontology": args.ontology,
        "ontology_track_indices": [int(i) for i in idxs],
        "metadata_source": (meta_src if isinstance(meta_src, str)
                            else (str(meta_src.relative_to(sp))
                                  if meta_src else None)),
    }

    # ---- read the metric -------------------------------------------------
    if idxs:
        col = u[:, idxs[0]]
        u_acc, u_don = float(col[acc_pos]), float(col[don_pos])
        win = 30
        a_off = int(np.argmax(col[acc_pos - win:acc_pos + win + 1])) - win
        d_off = int(np.argmax(col[don_pos - win:don_pos + win + 1])) - win
        la = F.logit_average(u_acc, u_don)
        print(f"\n--- the metric, native construct, track {idxs[0]} ---")
        print(f"  usage acceptor      {u_acc:.6f}")
        print(f"  usage donor         {u_don:.6f}")
        print(f"  logit_average       {la:.6f}")
        print(f"  argmax offset  acceptor {a_off:+d}   donor {d_off:+d}  "
              f"(0 = the designed site is the local peak)")
        print(f"  track range         {col.min():.6g} .. {col.max():.6g}")
        report.update(usage_acceptor=u_acc, usage_donor=u_don,
                      logit_average=la, argmax_offset_acceptor=a_off,
                      argmax_offset_donor=d_off,
                      track_min=float(col.min()), track_max=float(col.max()))
        if not (-1e-6 <= col.min() and col.max() <= 1 + 1e-6):
            print("  WARNING: usage is outside [0, 1]; the head may emit "
                  "logits, in which case a sigmoid is needed before "
                  "logit_average")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dest = args.out_dir / "local_probe.json"
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
