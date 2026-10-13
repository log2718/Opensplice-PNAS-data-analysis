"""
Corrected splice-site-aware context partition for the chimera experiment.

This REPLACES the six-block ``L1 L2 L3 | exon | R3 R2 R1`` decomposition in
``synthetic_context_lib``. It is a standalone module so the existing endpoint
and chimera results stay reproducible from the old code while the new design is
validated; nothing here imports from or mutates the old block functions.

Why the old partition was wrong
-------------------------------
The old ``L3`` / ``R3`` blocks were a flat 7 intronic nt on each side of the
middle exon. That is not a splice site: an acceptor spans the intron *and* the
first exonic bases, and a donor spans the last exonic bases *and* the intron.
The old cut therefore (a) used the wrong window on both sides, (b) put the
exonic half of each splice site inside the never-swapped middle exon, and
(c) gave the upstream exon's donor and the downstream exon's acceptor no block
of their own at all -- they were buried inside ``L1``/``L2`` and ``R2``/``R1``.

Corrected windows (Oded)
------------------------
* 3' splice site = 20 intronic nt + 3 exonic nt
* 5' splice site = 3 exonic nt + 6 intronic nt

Every splice-site window is a SELF-CONTAINED block, so no splice site is ever
split between an exon-body block and an intron-body block.

The eight binary context blocks
-------------------------------
::

    B1  upstream exon body
    B2  upstream exon 5'SS        3 exonic + 6 intronic
    B3  upstream intron body
    B4  middle exon 3'SS         20 intronic + 3 exonic   <- carries GTT
        synthetic70                                        [FIXED, not a block]
    B5  middle exon 5'SS          3 exonic + 6 intronic   <- carries CAG
    B6  downstream intron body
    B7  downstream exon 3'SS     20 intronic + 3 exonic
    B8  downstream exon body

The middle exon is ``GTT + synthetic70 + CAG``. ``GTT`` is the 3 exonic nt of
the middle exon's acceptor and therefore belongs entirely to ``B4``; ``CAG`` is
the 3 exonic nt of its donor and belongs entirely to ``B5``. Only
``synthetic70`` is held fixed across chimeras.

Note that the 3 exonic nt inside ``B4`` (``GTT``) and inside ``B5`` (``CAG``)
are the same in both contexts by construction -- those two blocks differ between
P and F only in their intronic halves (20 nt and 6 nt respectively). They are
still genuine binary blocks; the exonic trinucleotide is simply invariant.

Labels are eight characters, ``B1..B8`` left to right, e.g. ``PPFFPPPP``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import synthetic_context_lib as L  # noqa: E402

#: splice-site window geometry
ACCEPTOR_INTRONIC = 20
ACCEPTOR_EXONIC = 3
DONOR_EXONIC = 3
DONOR_INTRONIC = 6

BLOCK_ORDER_8 = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8")

BLOCK_ROLE = {
    "B1": "upstream exon body",
    "B2": "upstream exon 5'SS (3 exonic + 6 intronic)",
    "B3": "upstream intron body",
    "B4": "middle exon 3'SS (20 intronic + 3 exonic = GTT)",
    "B5": "middle exon 5'SS (3 exonic = CAG + 6 intronic)",
    "B6": "downstream intron body",
    "B7": "downstream exon 3'SS (20 intronic + 3 exonic)",
    "B8": "downstream exon body",
}


# ==========================================================================
# block construction
# ==========================================================================

def _split(up_exon: str, intron_up: str, intron_dn: str,
           dn_exon: str) -> dict[str, str]:
    """Cut one context's four scaffold pieces into the eight blocks.

    ``intron_up`` is everything between the upstream exon and the middle exon;
    ``intron_dn`` is everything between the middle exon and the downstream exon.
    """
    for name, s, need in (("upstream exon", up_exon, DONOR_EXONIC),
                          ("downstream exon", dn_exon, ACCEPTOR_EXONIC)):
        if len(s) < need:
            raise ValueError(f"{name} is {len(s)} nt; need >= {need}")
    for name, s in (("upstream intron", intron_up), ("downstream intron", intron_dn)):
        if len(s) < DONOR_INTRONIC + ACCEPTOR_INTRONIC:
            raise ValueError(
                f"{name} is {len(s)} nt; need >= "
                f"{DONOR_INTRONIC + ACCEPTOR_INTRONIC} so its 5'SS and 3'SS "
                "windows do not overlap")
    return {
        "B1": up_exon[:-DONOR_EXONIC],
        "B2": up_exon[-DONOR_EXONIC:] + intron_up[:DONOR_INTRONIC],
        "B3": intron_up[DONOR_INTRONIC:-ACCEPTOR_INTRONIC],
        "B4": intron_up[-ACCEPTOR_INTRONIC:] + L.EXON_PREFIX,
        "B5": L.EXON_SUFFIX + intron_dn[:DONOR_INTRONIC],
        "B6": intron_dn[DONOR_INTRONIC:-ACCEPTOR_INTRONIC],
        "B7": intron_dn[-ACCEPTOR_INTRONIC:] + dn_exon[:ACCEPTOR_EXONIC],
        "B8": dn_exon[ACCEPTOR_EXONIC:],
    }


def pnas_blocks8() -> dict[str, str]:
    """Eight-block cut of the original 1,309-nt PNAS reporter."""
    return _split(L.PNAS_EXON1, L.PNAS_INTRON1, L.PNAS_INTRON2, L.PNAS_EXON3)


def fas_blocks8(upstream70: str, downstream25: str) -> dict[str, str]:
    """Eight-block cut of the FAS/OpenSplice context for one TP53 exon.

    The scaffold is ``FAS_E5 + U + exon + D + FAS_E7`` with
    ``U = FAS_I5 + upstream70`` and ``D = downstream25 + FAS_I6``, i.e. exactly
    what ``alphagenome_local_prediction_pipeline_minigene.build_fas_minigene``
    produces.
    """
    import alphagenome_local_prediction_pipeline_minigene as ag
    if len(upstream70) != 70 or len(downstream25) != 25:
        raise ValueError("FAS flanks must be 70 / 25 nt")
    return _split(ag.FAS_E5, ag.FAS_I5 + upstream70,
                  downstream25 + ag.FAS_I6, ag.FAS_E7)


def assemble8(label: str, synthetic70: str, pb: dict[str, str],
              fb: dict[str, str]) -> tuple[str, int]:
    """Assemble a core from an eight-character P/F label.

    Returns ``(core, exon_start)`` where ``exon_start`` is the 0-based index of
    the middle exon's first base (the ``G`` of ``GTT``) in the unpadded core --
    the same convention the old ``assemble`` used, so ``score_core`` is unchanged.
    """
    if len(label) != 8 or set(label) - set("PF"):
        raise ValueError(f"bad context label {label!r}")
    pick = {b: (pb if c == "P" else fb)[b] for b, c in zip(BLOCK_ORDER_8, label)}
    left = pick["B1"] + pick["B2"] + pick["B3"] + pick["B4"]
    right = pick["B5"] + pick["B6"] + pick["B7"] + pick["B8"]
    core = left + synthetic70 + right
    # the exon starts 3 nt before synthetic70, at the GTT that ends B4
    return core, len(left) - len(L.EXON_PREFIX)


def all_labels8() -> list[str]:
    """All 256 eight-bit P/F labels, ``PPPPPPPP`` first."""
    return [f"{i:08b}".replace("0", "P").replace("1", "F") for i in range(256)]


# ==========================================================================
# validation
# ==========================================================================

def validate(synthetic70: str | None = None, verbose: bool = True) -> list[str]:
    """Every structural check for the corrected partition."""
    import alphagenome_local_prediction_pipeline_minigene as ag

    if synthetic70 is None:
        synthetic70 = "ACGT" * 17 + "AC"  # 70 nt, arbitrary
    assert len(synthetic70) == L.SYN_LEN
    exon = L.EXON_PREFIX + synthetic70 + L.EXON_SUFFIX

    fails: list[str] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
                  f"{'  ' + detail if detail else ''}")
        if not ok:
            fails.append(name)

    pb = pnas_blocks8()
    contexts = {"PNAS": pb}
    for tgt, (u70, d25) in L.FAS_TARGETS.items():
        contexts[tgt] = fas_blocks8(u70, d25)

    # ---- 4. window geometry -------------------------------------------
    for name, b in contexts.items():
        chk(f"4. {name}: B2 is 3 exonic + 6 intronic = 9 nt",
            len(b["B2"]) == DONOR_EXONIC + DONOR_INTRONIC, f"len={len(b['B2'])}")
        chk(f"4. {name}: B4 is 20 intronic + 3 exonic = 23 nt",
            len(b["B4"]) == ACCEPTOR_INTRONIC + ACCEPTOR_EXONIC,
            f"len={len(b['B4'])}")
        chk(f"4. {name}: B5 is 3 exonic + 6 intronic = 9 nt",
            len(b["B5"]) == DONOR_EXONIC + DONOR_INTRONIC, f"len={len(b['B5'])}")
        chk(f"4. {name}: B7 is 20 intronic + 3 exonic = 23 nt",
            len(b["B7"]) == ACCEPTOR_INTRONIC + ACCEPTOR_EXONIC,
            f"len={len(b['B7'])}")

    # ---- 9/10. GTT and CAG are wholly inside B4 / B5 -------------------
    for name, b in contexts.items():
        chk(f"9. {name}: GTT is entirely inside B4 and is its last 3 nt",
            b["B4"].endswith(L.EXON_PREFIX)
            and b["B4"].count(L.EXON_PREFIX) >= 1, b["B4"])
        chk(f"10. {name}: CAG is entirely inside B5 and is its first 3 nt",
            b["B5"].startswith(L.EXON_SUFFIX), b["B5"])

    # ---- 5. byte-for-byte reconstruction ------------------------------
    import run_alphagenome_pnas_fulltrack_sanity as PS
    manual = PS.build_construct(synthetic70)
    pnas_ref = manual["padded"][manual["left_pad"]:
                                manual["left_pad"] + manual["core_length"]]
    core_p, start_p = assemble8("PPPPPPPP", synthetic70, pb, pb)
    chk("5. PPPPPPPP reconstructs the ORIGINAL 1,309-nt PNAS construct "
        "byte-for-byte", core_p == pnas_ref and len(core_p) == 1309,
        f"len={len(core_p)}")
    chk("5b. the old 6-block PPPPPP and the new 8-block PPPPPPPP are the "
        "same construct",
        core_p == L.assemble("PPPPPP", exon, L.pnas_blocks(), L.pnas_blocks())[0])
    chk("5c. exon_start locates GTT+synthetic70+CAG in PPPPPPPP",
        core_p[start_p:start_p + len(exon)] == exon)

    for tgt, (u70, d25) in L.FAS_TARGETS.items():
        fb = contexts[tgt]
        ref = ag.build_fas_minigene(ag.build_variable_region(exon, u70, d25))
        core_f, start_f = assemble8("FFFFFFFF", synthetic70, pb, fb)
        chk(f"5. {tgt}: FFFFFFFF reconstructs the FAS construct byte-for-byte "
            "(vs build_fas_minigene)", core_f == ref, f"len={len(core_f)}")
        chk(f"5b. {tgt}: old 6-block FFFFFF == new 8-block FFFFFFFF",
            core_f == L.assemble("FFFFFF", exon, L.pnas_blocks(),
                                 L.fas_blocks(u70, d25))[0])
        chk(f"5c. {tgt}: exon_start locates the exon in FFFFFFFF",
            core_f[start_f:start_f + len(exon)] == exon)

    # ---- 6/7/8. no base dropped, duplicated, or exon touched ----------
    for tgt in ("TP53_e6", "TP53_e7"):
        fb = contexts[tgt]
        bad_len, bad_exon, bad_syn = [], [], []
        for lab in all_labels8():
            core, st = assemble8(lab, synthetic70, pb, fb)
            expect = sum(len((pb if c == "P" else fb)[b])
                         for b, c in zip(BLOCK_ORDER_8, lab)) + L.SYN_LEN
            if len(core) != expect:
                bad_len.append(lab)
            if core[st:st + len(exon)] != exon:
                bad_exon.append(lab)
            if core.count(synthetic70) != 1:
                bad_syn.append(lab)
        chk(f"6/7. {tgt}: all 256 chimeras concatenate with no dropped or "
            "duplicated base", not bad_len, f"{len(bad_len)} bad")
        chk(f"8. {tgt}: synthetic70 appears exactly once in all 256 chimeras, "
            "and GTT+syn+CAG is contiguous",
            not bad_syn and not bad_exon,
            f"{len(bad_syn)} syn, {len(bad_exon)} exon")

    # ---- blocks must actually differ between P and F -------------------
    for tgt in ("TP53_e6", "TP53_e7"):
        fb = contexts[tgt]
        same = [b for b in BLOCK_ORDER_8 if pb[b] == fb[b]]
        chk(f"B. {tgt}: every one of the 8 blocks differs between P and F "
            "(a non-degenerate factorial)", not same,
            f"identical blocks: {same}" if same else "")

    # ---- canonical dinucleotides at every junction ---------------------
    for name, b in contexts.items():
        chk(f"C. {name}: upstream 5'SS intronic half starts GT",
            b["B2"][DONOR_EXONIC:DONOR_EXONIC + 2] == "GT", b["B2"])
        chk(f"C. {name}: middle 3'SS intronic half ends AG",
            b["B4"][ACCEPTOR_INTRONIC - 2:ACCEPTOR_INTRONIC] == "AG", b["B4"])
        chk(f"C. {name}: middle 5'SS intronic half starts GT",
            b["B5"][DONOR_EXONIC:DONOR_EXONIC + 2] == "GT", b["B5"])
        chk(f"C. {name}: downstream 3'SS intronic half ends AG",
            b["B7"][ACCEPTOR_INTRONIC - 2:ACCEPTOR_INTRONIC] == "AG", b["B7"])

    # ---- 11. labels ----------------------------------------------------
    labs = all_labels8()
    chk("11. 256 unique block combinations",
        len(labs) == 256 and len(set(labs)) == 256)
    chk("11b. endpoints present", "PPPPPPPP" in labs and "FFFFFFFF" in labs)

    return fails


def block_table() -> "list[dict]":
    """Per-context block lengths and splice-site sequences, for reporting."""
    pb = pnas_blocks8()
    ctx = {"PNAS": pb}
    for tgt, (u70, d25) in L.FAS_TARGETS.items():
        ctx[tgt] = fas_blocks8(u70, d25)
    rows = []
    for name, b in ctx.items():
        for k in BLOCK_ORDER_8:
            rows.append({"context": name, "block": k, "role": BLOCK_ROLE[k],
                         "length": len(b[k]), "sequence": b[k]})
    return rows


if __name__ == "__main__":
    print("Validating the corrected 8-block splice-site partition\n" + "=" * 70)
    fails = validate()
    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURES: {fails}")
    raise SystemExit(1 if fails else 0)
