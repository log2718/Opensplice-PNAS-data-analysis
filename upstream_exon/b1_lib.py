"""
B1 (upstream exon body) randomization library for the upstream-exon pilot.

Only ``B1`` varies. ``B2``-``B8`` and the middle exon are taken from
``splice_site_partition.pnas_blocks8()`` and never modified, so every construct
is the original 1,309-nt PNAS cassette with its upstream exon body replaced.

Nothing here calls AlphaGenome and nothing here writes outside
``outputs/upstream_exon/``. The scoring metric is not redefined: the pilot
scores through ``synthetic_context_lib.score_core`` unchanged.

Three generators, all seeded deterministically from a single master seed:

``uniform``       i.i.d. uniform over ACGT (25% each)
``compmatched``   i.i.d. with the native B1 base frequencies
``dinuc``         Altschul-Erikson dinucleotide-preserving shuffle of native B1

The native PNAS B1 and the FAS/TP53 B1 are included as single reference
sequences. The FAS B1 is 59 nt, not 113, so it changes the construct length and
is flagged ``fixed_length=False``; it is excluded from the fixed-length
analysis and reported separately.
"""

from __future__ import annotations

import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
_ANALYSIS = ROOT / "analysis_script"
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

import synthetic_context_lib as L  # noqa: E402
import splice_site_partition as SP  # noqa: E402

#: master seed for the whole pilot; every per-variant seed derives from it
MASTER_SEED = 20261020

B1_LEN = 113
ALPHABET = "ACGT"

#: five plateau / near-plateau candidates + two rising-region controls
PLATEAU_IDS = [
    "syn_a5.0_00482",
    "syn_a5.0_00535",
    "syn_a0.5_00475",
    "syn_a1.0_00422",
    "syn_a5.0_01784",
]
CONTROL_IDS = [
    "syn_a1.0_02380",
    "syn_a1.0_00359",
]
EXON_IDS = PLATEAU_IDS + CONTROL_IDS

EXON_CLASS = {**{i: "plateau" for i in PLATEAU_IDS},
              **{i: "rising_control" for i in CONTROL_IDS}}

#: variant counts per arm (shared across every middle exon)
N_UNIFORM = 16
N_COMPMATCHED = 12
N_DINUC = 8

#: the native construct is scored twice per exon to measure call-to-call
#: reproducibility; this is the only deliberate duplicate
N_NATIVE_REPLICATES = 2


# ==========================================================================
# seeding
# ==========================================================================

def _rng(*parts) -> np.random.Generator:
    """Deterministic generator keyed on the master seed plus string parts."""
    key = "|".join([str(MASTER_SEED), *(str(p) for p in parts)])
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big") % 2**32)


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# ==========================================================================
# generators
# ==========================================================================

def native_b1() -> str:
    """The original PNAS upstream exon body, 113 nt."""
    return SP.pnas_blocks8()["B1"]


def fas_b1(target: str = "TP53_e6") -> str:
    """The FAS/OpenSplice upstream exon body. 59 nt -- NOT length-matched."""
    u70, d25 = L.FAS_TARGETS[target]
    return SP.fas_blocks8(u70, d25)["B1"]


def gen_uniform(n: int, length: int = B1_LEN) -> list[str]:
    """i.i.d. uniform over ACGT."""
    out = []
    for i in range(n):
        r = _rng("uniform", i)
        out.append("".join(np.asarray(list(ALPHABET))[r.integers(0, 4, length)]))
    return out


def gen_composition_matched(n: int, reference: str,
                            length: int = B1_LEN) -> list[str]:
    """i.i.d. with the reference sequence's base frequencies.

    Distinct from a mononucleotide shuffle: expected composition matches, the
    realized composition fluctuates, and no positional information survives.
    """
    probs = np.array([reference.count(b) for b in ALPHABET], dtype=float)
    probs /= probs.sum()
    out = []
    for i in range(n):
        r = _rng("compmatched", i)
        idx = r.choice(4, size=length, p=probs)
        out.append("".join(np.asarray(list(ALPHABET))[idx]))
    return out


def dinuc_shuffle(seq: str, rng: np.random.Generator) -> str:
    """Altschul-Erikson dinucleotide-preserving shuffle.

    Builds the de Bruijn multigraph of the sequence's dinucleotides, draws a
    random arborescence into the final vertex, and walks the resulting Eulerian
    path. Output has exactly the same length, the same base counts and the same
    dinucleotide counts as the input.
    """
    if len(seq) < 2:
        return seq

    edges: dict[str, list[str]] = defaultdict(list)
    for a, b in zip(seq, seq[1:]):
        edges[a].append(b)

    first, last = seq[0], seq[-1]
    verts = sorted(edges)

    # Draw last-edges until they form a tree rooted at `last`.
    for _ in range(10_000):
        last_edge: dict[str, str] = {}
        for v in verts:
            if v == last:
                continue
            choices = edges[v]
            last_edge[v] = choices[int(rng.integers(len(choices)))]
        if _is_arborescence(last_edge, last):
            break
    else:  # pragma: no cover
        raise RuntimeError("dinuc_shuffle: no valid arborescence found")

    ordered: dict[str, list[str]] = {}
    for v in verts:
        rest = list(edges[v])
        if v in last_edge:
            rest.remove(last_edge[v])
        perm = rng.permutation(len(rest))
        rest = [rest[i] for i in perm]
        if v in last_edge:
            rest.append(last_edge[v])
        ordered[v] = rest

    cursor = {v: 0 for v in verts}
    out = [first]
    cur = first
    for _ in range(len(seq) - 1):
        nxt = ordered[cur][cursor[cur]]
        cursor[cur] += 1
        out.append(nxt)
        cur = nxt
    return "".join(out)


def _is_arborescence(last_edge: dict[str, str], root: str) -> bool:
    """True when following last_edge from every vertex reaches root."""
    for v in last_edge:
        seen = set()
        cur = v
        while cur != root:
            if cur in seen or cur not in last_edge:
                return False
            seen.add(cur)
            cur = last_edge[cur]
    return True


def gen_dinuc(n: int, reference: str) -> list[str]:
    """``n`` dinucleotide-preserving shuffles of the reference."""
    return [dinuc_shuffle(reference, _rng("dinuc", i)) for i in range(n)]


def dinuc_counts(seq: str) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for a, b in zip(seq, seq[1:]):
        c[a + b] += 1
    return dict(c)


def base_counts(seq: str) -> dict[str, int]:
    return {b: seq.count(b) for b in ALPHABET}


# ==========================================================================
# construct assembly
# ==========================================================================

def build_construct(b1: str, synthetic70: str) -> tuple[str, int]:
    """Assemble the PNAS cassette with ``B1`` replaced.

    All of B2-B8 come from ``pnas_blocks8()``; passing the same dict as both
    the P and the F source means no FAS block can enter regardless of label.

    Returns ``(core, exon_start)`` with ``exon_start`` the 0-based index of the
    middle exon's first base, exactly as ``score_core`` expects.
    """
    blocks = dict(SP.pnas_blocks8())
    blocks["B1"] = b1
    return SP.assemble8("PPPPPPPP", synthetic70, blocks, blocks)


def validate_construct(core: str, exon_start: int, b1: str,
                       synthetic70: str, *, require_length: int | None = B1_LEN
                       ) -> list[str]:
    """Every structural assertion for one construct. Returns failure strings."""
    native = SP.pnas_blocks8()
    exon = L.EXON_PREFIX + synthetic70 + L.EXON_SUFFIX
    fails: list[str] = []

    if set(b1) - set(ALPHABET):
        fails.append(f"B1 has non-ACGT characters: {sorted(set(b1) - set(ALPHABET))}")
    if require_length is not None and len(b1) != require_length:
        fails.append(f"B1 length {len(b1)} != {require_length}")

    expect_len = (len(b1) + sum(len(native[b]) for b in SP.BLOCK_ORDER_8[1:])
                  + L.SYN_LEN)
    if len(core) != expect_len:
        fails.append(f"core length {len(core)} != expected {expect_len}")

    if core[exon_start:exon_start + len(exon)] != exon:
        fails.append("middle exon not located at exon_start")
    if core.count(synthetic70) != 1:
        fails.append(f"synthetic70 appears {core.count(synthetic70)} times")

    # re-extract every block from the assembled core by offset
    pos = 0
    got: dict[str, str] = {}
    for b in ("B1", "B2", "B3", "B4"):
        n = len(b1) if b == "B1" else len(native[b])
        got[b] = core[pos:pos + n]
        pos += n
    syn_seen = core[pos:pos + L.SYN_LEN]
    pos += L.SYN_LEN
    for b in ("B5", "B6", "B7", "B8"):
        got[b] = core[pos:pos + len(native[b])]
        pos += len(native[b])
    if pos != len(core):
        fails.append(f"block offsets consumed {pos} of {len(core)} nt")
    if syn_seen != synthetic70:
        fails.append("synthetic70 not at the expected offset")
    if got["B1"] != b1:
        fails.append("B1 in core does not match the requested variant")
    for b in SP.BLOCK_ORDER_8[1:]:
        if got[b] != native[b]:
            fails.append(f"{b} differs from native PNAS block")

    # splice-site dinucleotides must survive
    if got["B2"][3:5] != "GT":
        fails.append("upstream 5'SS intronic half does not start GT")
    if got["B4"][18:20] != "AG":
        fails.append("middle 3'SS intronic half does not end AG")
    if got["B5"][3:5] != "GT":
        fails.append("middle 5'SS intronic half does not start GT")
    if got["B7"][18:20] != "AG":
        fails.append("downstream 3'SS intronic half does not end AG")

    return fails


# ==========================================================================
# variant panel
# ==========================================================================

def build_variant_panel() -> list[dict]:
    """The B1 panel, shared by every middle exon.

    Sharing the panel costs no extra AlphaGenome calls -- each (exon, B1) pair
    is a distinct construct either way -- and allows paired comparison of the
    same B1 across exons.
    """
    nat = native_b1()
    panel: list[dict] = [
        {"b1_id": "native_PNAS", "arm": "native", "b1": nat, "fixed_length": True},
        {"b1_id": "fas_TP53", "arm": "fas_reference", "b1": fas_b1(),
         "fixed_length": False},
    ]
    for i, s in enumerate(gen_uniform(N_UNIFORM)):
        panel.append({"b1_id": f"uniform_{i:02d}", "arm": "uniform",
                      "b1": s, "fixed_length": True})
    for i, s in enumerate(gen_composition_matched(N_COMPMATCHED, nat)):
        panel.append({"b1_id": f"compmatched_{i:02d}", "arm": "compmatched",
                      "b1": s, "fixed_length": True})
    for i, s in enumerate(gen_dinuc(N_DINUC, nat)):
        panel.append({"b1_id": f"dinuc_{i:02d}", "arm": "dinuc",
                      "b1": s, "fixed_length": True})
    for p in panel:
        p["b1_sha256"] = sha(p["b1"])
        p["b1_len"] = len(p["b1"])
        p["b1_gc"] = round((p["b1"].count("G") + p["b1"].count("C")) / len(p["b1"]), 4)
    return panel


def load_exons() -> "list[dict]":
    """The seven selected middle exons, read from the existing library CSV."""
    import pandas as pd
    lib = pd.read_csv(ROOT / "outputs" / "synthetic_context" /
                      "synthetic_exon_library.csv")
    sel = lib[lib.sequence_id.isin(EXON_IDS)]
    missing = set(EXON_IDS) - set(sel.sequence_id)
    if missing:
        raise SystemExit(f"missing exon ids in library: {sorted(missing)}")
    out = []
    for eid in EXON_IDS:  # preserve the documented order
        r = sel[sel.sequence_id == eid].iloc[0]
        out.append({
            "sequence_id": eid,
            "exon_class": EXON_CLASS[eid],
            "alpha": float(r.alpha),
            "synthetic_70mer": r.synthetic_70mer,
            "full_middle_exon": r.full_middle_exon,
            "PNAS_common": float(r.PNAS_common),
        })
    return out
