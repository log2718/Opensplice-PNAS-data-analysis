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
                 motif-aware. This is the comparator that matters: it asks
                 whether the CNN predicts better than position-blind k-mer
                 counting does.
``dinucleotide`` ridge on the 16 dinucleotide frequencies. For the v3
                 fixed-composition library this REPLACES the composition
                 ridge, which is degenerate there: every sequence has the
                 same base counts, so its features have zero variance and it
                 can only predict the training mean. Dinucleotide frequency
                 is the simplest thing that still varies once composition is
                 held fixed, which makes it the right floor for v3.

NOTE on interpretation: beating the k-mer ridge shows the CNN as a whole adds
predictive value over position-blind motif counts. It does NOT isolate the
contribution of the learned positional biases -- the two models differ in
architecture, fitting and capacity, not only in whether position is used. That
question needs an ablation of the same CNN with and without its position bias,
which is not implemented here.
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


# ==========================================================================
# order-only features (v3, fixed composition)
# ==========================================================================

#: v2's 6-mer RidgeCV selected the largest alpha on its grid (1,000), i.e. the
#: grid was the binding constraint rather than the data. Every ridge here uses
#: a grid wide enough that the choice is interior.
WIDE_ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6)


DINUCS = ["".join(p) for p in product(ALPHABET, repeat=2)]


def dinucleotide_features(seqs: list[str]) -> np.ndarray:
    """(N, 16) dinucleotide frequencies.

    With composition held exactly fixed, mononucleotide frequencies carry no
    information at all, so this is the first feature set that can. It stays
    position-blind, like the k-mer ridge.
    """
    idx = {d: i for i, d in enumerate(DINUCS)}
    x = np.zeros((len(seqs), len(DINUCS)), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = s.upper()
        denom = max(len(s) - 1, 1)
        for j in range(len(s) - 1):
            col = idx.get(s[j:j + 2])
            if col is not None:
                x[i, col] += 1.0
        x[i] /= denom
    return x


def dinucleotide_baseline() -> "RidgeFeatureBaseline":
    return RidgeFeatureBaseline(dinucleotide_features, "dinucleotide_ridge",
                                alphas=WIDE_ALPHAS)


# ==========================================================================
# memory-aware ridge for large k-mer designs
# ==========================================================================

class GramRidgeBaseline:
    """Ridge on a large feature matrix, without ever holding it in memory.

    A 6-mer design for the v3 training split is 200,000 x 4,096. As float32
    that is 3.3 GB, and ``StandardScaler`` then copies it, and ``RidgeCV``'s
    leave-one-out path copies it again -- comfortably over 10 GB before any
    arithmetic. Here the features are built in chunks and folded straight into
    the normal equations::

        G = X^T X   (4096 x 4096, 134 MB)      b = X^T y

    so peak memory is one chunk, and every alpha is then a cheap solve of
    ``(G + alpha*I) w = b`` against the SAME factorized Gram.

    Alpha is chosen on an explicit validation split when one is given (the
    honest choice here, since the splits already exist), otherwise on a
    deterministic 10% holdout carved out of train.
    """

    def __init__(self, featurizer, name: str, alphas=WIDE_ALPHAS,
                 chunk: int = 20_000):
        self.featurizer = featurizer
        self.name = name
        self.alphas = tuple(alphas)
        self.chunk = chunk

    # -- internals ------------------------------------------------------
    def _chunks(self, seqs):
        for i in range(0, len(seqs), self.chunk):
            yield seqs[i:i + self.chunk]

    def _moments(self, seqs):
        """Feature mean and sd in one streaming pass."""
        n = len(seqs)
        s1 = s2 = None
        for part in self._chunks(seqs):
            x = np.asarray(self.featurizer(part), dtype=np.float64)
            s1 = x.sum(0) if s1 is None else s1 + x.sum(0)
            s2 = (x * x).sum(0) if s2 is None else s2 + (x * x).sum(0)
        mean = s1 / n
        var = np.maximum(s2 / n - mean ** 2, 0.0)
        sd = np.sqrt(var)
        sd[sd < 1e-12] = 1.0          # constant columns contribute nothing
        return mean, sd

    def _normal_equations(self, seqs, y):
        """Accumulate G = X^T X and b = X^T y over standardized chunks."""
        y = np.asarray(y, dtype=np.float64)
        G = b = None
        off = 0
        for part in self._chunks(seqs):
            x = np.asarray(self.featurizer(part), dtype=np.float64)
            x -= self.mean_
            x /= self.sd_
            yc = y[off:off + len(part)] - self.y_mean_
            off += len(part)
            g = x.T @ x
            v = x.T @ yc
            G = g if G is None else G + g
            b = v if b is None else b + v
        return G, b

    def _predict_centered(self, seqs, w):
        out = np.empty(len(seqs), dtype=np.float64)
        off = 0
        for part in self._chunks(seqs):
            x = np.asarray(self.featurizer(part), dtype=np.float64)
            x -= self.mean_
            x /= self.sd_
            out[off:off + len(part)] = x @ w
            off += len(part)
        return out

    # -- public ---------------------------------------------------------
    def fit(self, seqs, y, val_seqs=None, val_y=None):
        seqs = list(seqs)
        y = np.asarray(y, dtype=np.float64)
        if val_seqs is None:
            rng = np.random.default_rng(0)
            order = rng.permutation(len(seqs))
            n_val = max(1, int(0.1 * len(seqs)))
            vi, ti = order[:n_val], order[n_val:]
            val_seqs = [seqs[i] for i in vi]
            val_y = y[vi]
            seqs = [seqs[i] for i in ti]
            y = y[ti]
            self.alpha_selected_on_ = "10% holdout of train"
        else:
            val_seqs = list(val_seqs)
            val_y = np.asarray(val_y, dtype=np.float64)
            self.alpha_selected_on_ = "explicit validation split"

        self.mean_, self.sd_ = self._moments(seqs)
        self.y_mean_ = float(y.mean())
        G, b = self._normal_equations(seqs, y)

        # one eigendecomposition, then every alpha is a diagonal solve
        evals, evecs = np.linalg.eigh(G)
        bt = evecs.T @ b
        best = (np.inf, None, None)
        self.alpha_path_ = []
        for a in self.alphas:
            w = evecs @ (bt / (evals + a))
            pred = self._predict_centered(val_seqs, w) + self.y_mean_
            mse = float(np.mean((val_y - pred) ** 2))
            self.alpha_path_.append({"alpha": float(a), "val_mse": mse})
            if mse < best[0]:
                best = (mse, float(a), w)
        self.val_mse_, self.alpha_, self.w_ = best
        if self.alpha_ in (self.alphas[0], self.alphas[-1]):
            import warnings
            warnings.warn(
                f"{self.name}: selected alpha={self.alpha_:g} sits at the edge "
                "of the grid -- widen WIDE_ALPHAS before trusting it",
                stacklevel=2)
        return self

    def predict(self, seqs):
        seqs = list(seqs)
        return (self._predict_centered(seqs, self.w_)
                + self.y_mean_).astype(np.float32)


def kmer_gram_baseline(k: int = 6, chunk: int = 20_000) -> GramRidgeBaseline:
    return GramRidgeBaseline(lambda s: kmer_counts(s, k), f"kmer{k}_ridge",
                             chunk=chunk)


#: which reference models go with which library design
BASELINE_SETS = ("dirichlet", "fixed_composition")


def all_baselines_for(design: str, k: int = 6) -> list:
    """Reference models appropriate to the library design.

    ``dirichlet``          v1 / v2: composition varies, so the composition
                           ridge is informative and is the key control.
    ``fixed_composition``  v3: composition is identical everywhere, so the
                           composition ridge is degenerate and is replaced by
                           a dinucleotide ridge; the k-mer ridge uses the
                           streaming Gram solver so it fits 200,000 rows.
    """
    if design == "dirichlet":
        return all_baselines(k)
    if design == "fixed_composition":
        return [ConstantBaseline(), dinucleotide_baseline(),
                kmer_gram_baseline(k)]
    raise ValueError(f"unknown baseline set {design!r}; "
                     f"expected one of {BASELINE_SETS}")
