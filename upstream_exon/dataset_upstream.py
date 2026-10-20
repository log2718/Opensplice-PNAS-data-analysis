"""
Encoding, splits and torch datasets for the upstream-exon (B1) experiment.

Inputs are B1 sequences; the target is AlphaGenome's middle-exon mean-logit.
Encoders are imported from the untouched ``PNAS_model.utils`` so the one-hot
convention (``ACGT`` channel order), dot-bracket structure channels and wobble
indicator are identical to the original model's.

Splits are read from the manifest's ``split`` column, which was assigned
deterministically and alpha-stratified at dataset-build time, so training and
AlphaGenome scoring can never disagree about which sequence is held out.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PNAS_model.utils import one_hot_batch  # noqa: E402

TARGET_COL = "AG_mean_logit"


# ==========================================================================
# encoding
# ==========================================================================

def encode_sequences(seqs: list[str]) -> np.ndarray:
    """(N, 4, L) float32 one-hot, ACGT channel order."""
    return one_hot_batch([s.upper() for s in seqs]).astype(np.float32)


def encode_structure(seqs: list[str], *, rnafold_bin: str = "RNAfold",
                     temperature: float = 37.0,
                     num_threads: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """(N, 3, L) structure one-hot and (N, 1, L) wobble indicator.

    Folds each B1 sequence in ISOLATION. That is a real modelling assumption,
    not a neutral choice -- the 113-mer does not fold alone inside the
    1,309-nt transcript, and any long-range pairing with the middle exon is
    invisible here. Recorded in the config so it is never silently assumed.
    """
    from PNAS_model.utils import compute_structure, compute_wobbles, rnafold_available
    if not rnafold_available(rnafold_bin):
        raise RuntimeError(
            f"{rnafold_bin} not on PATH; structure features unavailable. "
            "Install ViennaRNA or train the sequence-only model.")
    seqs = [s.upper() for s in seqs]
    struct_oh, structures, _ = compute_structure(
        seqs, rnafold_bin=rnafold_bin, temperature=temperature,
        num_threads=num_threads)
    wobbles = compute_wobbles(seqs, structures)
    return struct_oh.astype(np.float32), wobbles.astype(np.float32)


# ==========================================================================
# dataset
# ==========================================================================

class B1Dataset(Dataset):
    """B1 one-hot (+ optional structure) with a scalar regression target."""

    def __init__(self, x_seq: np.ndarray, y: np.ndarray,
                 x_struct: np.ndarray | None = None,
                 x_wobble: np.ndarray | None = None,
                 alpha: np.ndarray | None = None):
        self.x_seq = torch.as_tensor(x_seq, dtype=torch.float32)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.float32)
        self.has_struct = x_struct is not None and x_wobble is not None
        if self.has_struct:
            self.x_struct = torch.as_tensor(x_struct, dtype=torch.float32)
            self.x_wobble = torch.as_tensor(x_wobble, dtype=torch.float32)
        self.alpha = (torch.as_tensor(np.asarray(alpha), dtype=torch.float32)
                      if alpha is not None else torch.zeros(len(self.y)))
        if len(self.x_seq) != len(self.y):
            raise ValueError("x_seq and y must have the same length")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        if self.has_struct:
            return (self.x_seq[i], self.x_struct[i], self.x_wobble[i],
                    self.y[i], self.alpha[i])
        z = torch.zeros(0)
        return self.x_seq[i], z, z, self.y[i], self.alpha[i]


def load_scored(results_csv: Path, manifest_csv: Path) -> pd.DataFrame:
    """Join AlphaGenome results back to the manifest on ``row_id``.

    Re-asserts the B1 digest so a mismatched or reordered results file cannot
    silently train the model on the wrong sequences.
    """
    import hashlib
    res = pd.read_csv(results_csv)
    man = pd.read_csv(manifest_csv)
    keep = [c for c in man.columns if c not in res.columns or c == "row_id"]
    df = res.merge(man[keep], on="row_id", how="inner", validate="one_to_one")
    if len(df) != len(res):
        raise ValueError(
            f"join lost rows: {len(res)} results -> {len(df)} joined")
    bad = [r.row_id for r in df.itertuples()
           if hashlib.sha256(r.b1.encode()).hexdigest() != r.b1_sha256]
    if bad:
        raise ValueError(f"{len(bad)} rows have a B1 digest mismatch, "
                         f"first {bad[0]}")
    if df[TARGET_COL].isna().any():
        raise ValueError(f"{int(df[TARGET_COL].isna().sum())} rows have no target")
    return df


def build_datasets(df: pd.DataFrame, *, use_structure: bool = False,
                   splits: tuple[str, ...] = ("train", "val", "test"),
                   rnafold_bin: str = "RNAfold",
                   num_threads: int = 8) -> dict[str, B1Dataset]:
    out: dict[str, B1Dataset] = {}
    for name in splits:
        g = df[df.split == name]
        if g.empty:
            continue
        seqs = g.b1.tolist()
        xs = encode_sequences(seqs)
        xt = xw = None
        if use_structure:
            xt, xw = encode_structure(seqs, rnafold_bin=rnafold_bin,
                                      num_threads=num_threads)
        out[name] = B1Dataset(xs, g[TARGET_COL].to_numpy(), xt, xw,
                              g.alpha.to_numpy())
    return out


def make_loaders(datasets: dict[str, B1Dataset], batch_size: int = 64,
                 num_workers: int = 0, *,
                 shuffle_train: bool = True) -> dict[str, DataLoader]:
    """Loaders for each split.

    ``shuffle_train=False`` gives EVALUATION loaders: every split is then
    emitted in manifest order, so predictions can be joined row-for-row back
    to group ids and composition statistics.
    """
    return {
        name: DataLoader(ds, batch_size=batch_size,
                         shuffle=(shuffle_train and name == "train"),
                         num_workers=num_workers,
                         drop_last=False)
        for name, ds in datasets.items()
    }


def assert_no_leakage(df: pd.DataFrame) -> None:
    """Hard guarantee that no B1 sequence appears in two splits."""
    seen: dict[str, str] = {}
    for r in df.itertuples():
        prev = seen.get(r.b1_sha256)
        if prev is not None and prev != r.split:
            raise ValueError(
                f"B1 {r.b1_id} appears in both {prev!r} and {r.split!r}")
        seen[r.b1_sha256] = r.split
