"""
Reference models for the upstream-exon regression.

These exist so that the interpretable CNN is judged against something, not
against zero. Three levels:

``constant``     predict the training mean. The floor -- R^2 = 0 by definition.
``composition``  ridge on 4 base frequencies + GC + entropy + max homopolymer.
                 The pilot found composition explains nothing (|r| <= 0.16), so
                 this one is expected to fail; if it does not, the CNN's
                 advantage is not about motifs.
``kmer``         ridge on k-mer counts (k=6 by default). Position-blind but
                 motif-aware. This is the comparator that matters: if the CNN
                 cannot beat it, the learned position biases are buying nothing.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

ALPHABET = "ACGT"


def kmer_index(k: int) -> dict[str, int]:
    return {"".join(p): i for i, p in enumerate(product(ALPHABET, repeat=k))}


def kmer_counts(seqs: list[str], k: int = 6) -> np.ndarray:
    idx = kmer_index(k)
    x = np.zeros((len(seqs), len(idx)), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = s.upper()
        for j in range(len(s) - k + 1):
            key = s[j:j + k]
            col = idx.get(key)
            if col is not None:
                x[i, col] += 1.0
    return x


def composition_features(seqs: list[str]) -> np.ndarray:
    rows = []
    for s in seqs:
        s = s.upper()
        n = len(s)
        f = [s.count(b) / n for b in ALPHABET]
        gc = f[1] + f[2]
        p = np.array([v for v in f if v > 0])
        ent = float(-(p * np.log2(p)).sum())
        best = run = 1
        for a, b in zip(s, s[1:]):
            run = run + 1 if a == b else 1
            best = max(best, run)
        rows.append([*f, gc, ent, best / n])
    return np.asarray(rows, dtype=np.float32)


class ConstantBaseline:
    name = "constant"

    def fit(self, seqs, y):
        self.mu = float(np.mean(y))
        return self

    def predict(self, seqs):
        return np.full(len(seqs), self.mu, dtype=np.float32)


class RidgeFeatureBaseline:
    def __init__(self, featurizer, name: str,
                 alphas=(0.1, 1.0, 10.0, 100.0, 1000.0)):
        self.featurizer = featurizer
        self.name = name
        self.alphas = alphas

    def fit(self, seqs, y):
        x = self.featurizer(seqs)
        self.scaler = StandardScaler().fit(x)
        xs = self.scaler.transform(x)
        # RidgeCV needs >= 2 folds worth of data; fall back for tiny inputs
        if len(y) >= 10:
            self.model = RidgeCV(alphas=self.alphas).fit(xs, y)
            self.alpha_ = float(self.model.alpha_)
        else:
            self.model = Ridge(alpha=self.alphas[-1]).fit(xs, y)
            self.alpha_ = float(self.alphas[-1])
        return self

    def predict(self, seqs):
        return self.model.predict(self.scaler.transform(
            self.featurizer(seqs))).astype(np.float32)


def composition_baseline() -> RidgeFeatureBaseline:
    return RidgeFeatureBaseline(composition_features, "composition_ridge")


def kmer_baseline(k: int = 6) -> RidgeFeatureBaseline:
    return RidgeFeatureBaseline(lambda s: kmer_counts(s, k), f"kmer{k}_ridge")


def all_baselines(k: int = 6) -> list:
    return [ConstantBaseline(), composition_baseline(), kmer_baseline(k)]
