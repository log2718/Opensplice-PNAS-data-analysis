"""
Train the PNAS-derived interpretable regressor on B1 -> AlphaGenome mean-logit.

Reproducible, resumable, and always evaluated against baselines. Writes
checkpoints, history, config, predictions and plots under a run directory.

Usage::

    # CPU smoke test on whatever is already scored
    python upstream_exon/train_upstream.py --smoke --run-name smoke

    # full run
    python upstream_exon/train_upstream.py --run-name seq_v1 --epochs 300

    # resume
    python upstream_exon/train_upstream.py --run-name seq_v1 --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR.parent
for _p in (str(_THIS_DIR), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import baselines as BL  # noqa: E402
import dataset_upstream as D  # noqa: E402
from model_upstream import UpstreamExonModel  # noqa: E402

OUT = ROOT / "outputs" / "upstream_exon"
DATA = OUT / "dataset_v2"   # v2 is the current library; pass
                            # --results-csv/--manifest-csv for v1
RUNS = OUT / "models"

logger = logging.getLogger("train_upstream")


# ==========================================================================
# metrics
# ==========================================================================

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from scipy import stats
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    var = float(np.var(y_true))
    out = {
        "n": int(len(y_true)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1.0 - mse / var) if var > 0 else float("nan"),
        "target_sd": float(np.std(y_true)),
    }
    if len(y_true) > 2 and np.std(y_pred) > 0 and np.std(y_true) > 0:
        out["pearson_r"] = float(stats.pearsonr(y_true, y_pred).statistic)
        out["spearman_rho"] = float(stats.spearmanr(y_true, y_pred).statistic)
    else:
        out["pearson_r"] = float("nan")
        out["spearman_rho"] = float("nan")
    return out


def pairwise_difference_metrics(y_true, y_pred, n_pairs: int = 200_000,
                                seed: int = 0) -> dict:
    """Does the model capture RELATIVE differences between B1 sequences?

    Absolute accuracy can be dominated by a constant offset; this asks whether
    y_i - y_j is predicted, which is the quantity the experiment is about.
    """
    from scipy import stats
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n < 3:
        return {}
    r = np.random.default_rng(seed)
    n_pairs = min(n_pairs, n * (n - 1) // 2)
    i = r.integers(0, n, n_pairs)
    j = r.integers(0, n, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    dt = y_true[i] - y_true[j]
    dp = y_pred[i] - y_pred[j]
    nz = np.abs(dt) > 1e-12
    return {
        "pair_n": int(len(dt)),
        "pair_sign_agreement": float(np.mean(np.sign(dp[nz]) == np.sign(dt[nz]))),
        "pair_pearson_r": float(stats.pearsonr(dt, dp).statistic),
        "pair_spearman_rho": float(stats.spearmanr(dt, dp).statistic),
        "pair_mae": float(np.mean(np.abs(dp - dt))),
    }


def within_group_metrics(df: pd.DataFrame) -> dict:
    """Does the model track differences AMONG sequences sharing one p?

    In the v2 grouped design the four replicates of a group are drawn from the
    same Dirichlet probability vector, so composition is matched in
    expectation and a within-group difference is attributable to nucleotide
    ORDER. This is the sharpest available test that the CNN is not simply a
    composition readout.
    """
    from scipy import stats
    if "group_id" not in df or df.group_id.isna().all():
        return {}
    g = df.groupby("group_id")
    df = df.copy()
    df["dy_true"] = df.y_true - g.y_true.transform("mean")
    df["dy_pred"] = df.y_pred - g.y_pred.transform("mean")
    sizes = g.size()
    multi = df[df.group_id.isin(sizes[sizes > 1].index)]
    if len(multi) < 3:
        return {"within_group_n": int(len(multi))}
    gm = g[["y_true", "y_pred"]].mean()
    out = {
        "within_group_n": int(len(multi)),
        "within_group_n_groups": int(multi.group_id.nunique()),
        "within_group_true_sd": float(g.y_true.std().mean()),
        "within_group_pred_sd": float(g.y_pred.std().mean()),
    }
    # A model that only reproduces group means has no within-group variation
    # at all. That is a meaningful answer -- zero order sensitivity -- so
    # report it as 0 rather than dropping the metric.
    if multi.dy_true.std() == 0 or multi.dy_pred.std() == 0:
        out.update({"within_group_pearson_r": 0.0,
                    "within_group_spearman_rho": 0.0,
                    "within_group_sign_agreement": 0.0,
                    "within_group_degenerate": True})
    else:
        nz = multi.dy_true.abs() > 1e-12
        out.update({
            "within_group_pearson_r": float(
                stats.pearsonr(multi.dy_true, multi.dy_pred).statistic),
            "within_group_spearman_rho": float(
                stats.spearmanr(multi.dy_true, multi.dy_pred).statistic),
            "within_group_sign_agreement": float(
                np.mean(np.sign(multi.dy_pred[nz])
                        == np.sign(multi.dy_true[nz]))),
            "within_group_degenerate": False,
        })
    if len(gm) > 2 and gm.y_pred.std() > 0:
        out["between_group_pearson_r"] = float(
            stats.pearsonr(gm.y_true, gm.y_pred).statistic)
        out["between_group_true_sd"] = float(gm.y_true.std())
    return out


def metrics_by_composition(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Performance across nucleotide-composition ranges (GC quantile bins)."""
    if "gc" not in df:
        return pd.DataFrame()
    d = df.copy()
    try:
        d["gc_bin"] = pd.qcut(d.gc, n_bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    rows = []
    for b, g in d.groupby("gc_bin", observed=True):
        rows.append({"gc_bin": str(b), "gc_mid": float(g.gc.mean()),
                     **regression_metrics(g.y_true.values, g.y_pred.values)})
    return pd.DataFrame(rows)


def beyond_composition_metrics(train_df: pd.DataFrame, eval_df: pd.DataFrame,
                               kmer_k: int = 6) -> dict:
    """How much of the model's skill survives removing composition?

    Fits a composition-only ridge on train, then asks how well the CNN
    predicts the composition-RESIDUALIZED target on the evaluation split. A
    model that is only reading base content scores ~0 here.
    """
    import baselines as _BL
    from scipy import stats
    if len(train_df) < 10 or len(eval_df) < 5:
        return {}
    comp = _BL.composition_baseline().fit(
        train_df.b1.tolist(), train_df.y_true.to_numpy())
    comp_pred = comp.predict(eval_df.b1.tolist())
    resid_true = eval_df.y_true.to_numpy() - comp_pred
    resid_pred = eval_df.y_pred.to_numpy() - comp_pred
    out = {
        "composition_only_r2": float(
            regression_metrics(eval_df.y_true.to_numpy(), comp_pred)["r2"]),
        "model_r2_on_composition_residual": float(
            regression_metrics(resid_true, resid_pred)["r2"]),
    }
    if resid_true.std() > 0 and resid_pred.std() > 0:
        out["residual_pearson_r"] = float(
            stats.pearsonr(resid_true, resid_pred).statistic)
        out["residual_spearman_rho"] = float(
            stats.spearmanr(resid_true, resid_pred).statistic)
    return out


def metrics_by_alpha(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, g in df.groupby("alpha"):
        m = regression_metrics(g.y_true.values, g.y_pred.values)
        rows.append({"alpha": a, **m})
    overall = regression_metrics(df.y_true.values, df.y_pred.values)
    rows.append({"alpha": "ALL", **overall})
    return pd.DataFrame(rows)


# ==========================================================================
# loops
# ==========================================================================

class Standardizer:
    """z-scores the target using TRAIN statistics only.

    The model optimizes in standardized units; every reported metric is
    converted back to logit units first, so MSE/MAE/R2 are always comparable
    to AG_mean_logit and to the baselines.
    """

    def __init__(self, mu: float = 0.0, sd: float = 1.0, active: bool = True):
        self.mu = float(mu)
        self.sd = float(sd) if sd and abs(sd) > 1e-12 else 1.0
        self.active = active

    def forward(self, y):
        return (y - self.mu) / self.sd if self.active else y

    def inverse(self, y):
        return y * self.sd + self.mu if self.active else y

    def as_dict(self) -> dict:
        return {"mu": self.mu, "sd": self.sd, "active": self.active}


def _unpack(batch, device, use_structure):
    xs, xt, xw, y, a = batch
    xs = xs.to(device)
    y = y.to(device)
    if use_structure:
        return xs, xt.to(device), xw.to(device), y, a
    return xs, None, None, y, a


def run_epoch(model, loader, device, loss_fn, optimizer=None, *,
              l1_lambda=0.0, l2_smooth=0.0, scaler: "Standardizer | None" = None):
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    preds, trues = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            xs, xt, xw, y, _ = _unpack(batch, device, model.use_structure)
            out = model(xs, xt, xw)
            loss = loss_fn(out, y)
            if train:
                full = loss
                if l1_lambda:
                    full = full + l1_lambda * model.l1_filters()
                if l2_smooth:
                    full = full + l2_smooth * model.l2_position_smoothness()
                optimizer.zero_grad(set_to_none=True)
                full.backward()
                optimizer.step()
            total += float(loss.item()) * y.size(0)
            n += y.size(0)
            preds.append(out.detach().cpu().numpy())
            trues.append(y.detach().cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    if scaler is not None:
        preds, trues = scaler.inverse(preds), scaler.inverse(trues)
    return {"loss": total / max(n, 1), **regression_metrics(trues, preds)}


@torch.no_grad()
def predict(model, loader, device, scaler: "Standardizer | None" = None):
    model.eval()
    preds, trues, alphas = [], [], []
    for batch in loader:
        xs, xt, xw, y, a = _unpack(batch, device, model.use_structure)
        preds.append(model(xs, xt, xw).cpu().numpy())
        trues.append(y.cpu().numpy())
        alphas.append(np.asarray(a))
    p, t = np.concatenate(preds), np.concatenate(trues)
    if scaler is not None:
        p, t = scaler.inverse(p), scaler.inverse(t)
    return p, t, np.concatenate(alphas)


# ==========================================================================
# plotting
# ==========================================================================

def plot_run(history: pd.DataFrame, test_df: pd.DataFrame, run_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axes[0].plot(history.epoch, history.train_loss, label="train")
    axes[0].plot(history.epoch, history.val_loss, label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("training curve"); axes[0].legend(); axes[0].grid(alpha=.3)

    axes[1].plot(history.epoch, history.val_r2, color="#2f6f9f")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val R²")
    axes[1].set_title("validation R²"); axes[1].grid(alpha=.3)

    ax = axes[2]
    for a, g in test_df.groupby("alpha"):
        ax.scatter(g.y_true, g.y_pred, s=12, alpha=.6, label=f"α={a}")
    lo = float(min(test_df.y_true.min(), test_df.y_pred.min()))
    hi = float(max(test_df.y_true.max(), test_df.y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "--", color="#777", linewidth=1)
    ax.set_xlabel("AlphaGenome mean-logit (true)")
    ax.set_ylabel("predicted")
    ax.set_title("test set"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout()
    p = run_dir / "training_summary.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ==========================================================================
# main
# ==========================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    d = ap.add_argument_group("data")
    d.add_argument("--results-csv", default=str(DATA / "b1_primary_ag_results.csv"))
    d.add_argument("--manifest-csv", default=str(DATA / "b1_primary_manifest.csv"))
    d.add_argument("--use-structure", action="store_true",
                   help="add the structure branch (needs RNAfold on PATH)")
    d.add_argument("--rnafold-bin", default="RNAfold")

    m = ap.add_argument_group("model")
    m.add_argument("--input-length", type=int, default=113)
    m.add_argument("--num-seq-filters", type=int, default=20)
    m.add_argument("--head", choices=["linear", "residual_tuner", "identity"],
                   default="linear")
    m.add_argument("--batchnorm", action="store_true",
                   help="only with --head residual_tuner; off by default "
                        "because the original README advises against it")

    o = ap.add_argument_group("optimization")
    o.add_argument("--epochs", type=int, default=300)
    o.add_argument("--batch-size", type=int, default=64)
    o.add_argument("--lr", type=float, default=1e-3)
    o.add_argument("--weight-decay", type=float, default=0.0)
    o.add_argument("--l1-lambda", type=float, default=0.0)
    o.add_argument("--l2-smooth", type=float, default=0.0)
    o.add_argument("--loss", choices=["mse", "huber"], default="mse")
    o.add_argument("--huber-delta", type=float, default=1.0)
    o.add_argument("--patience", type=int, default=30)
    o.add_argument("--no-standardize-target", action="store_true",
                   help="train on raw logit units instead of z-scored "
                        "target (standardization is on by default)")

    r = ap.add_argument_group("runtime")
    r.add_argument("--run-name", default="seq_v1")
    r.add_argument("--out-dir", default=str(RUNS))
    r.add_argument("--device", default=None)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--num-workers", type=int, default=0)
    r.add_argument("--resume", action="store_true")
    r.add_argument("--skip-baselines", action="store_true")
    r.add_argument("--kmer-k", type=int, default=6)
    r.add_argument("--smoke", action="store_true",
                   help="tiny run: 3 epochs, subsample, CPU-friendly")
    return ap


def resolve_device(spec: str | None) -> torch.device:
    if spec:
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = Path(args.out_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    logger.info("run_dir=%s  device=%s", run_dir, device)

    # ---- data -------------------------------------------------------
    df = D.load_scored(Path(args.results_csv), Path(args.manifest_csv))
    D.assert_no_leakage(df)
    if args.smoke:
        args.epochs = min(args.epochs, 3)
        df = pd.concat([g.head(min(len(g), 64))
                        for _, g in df.groupby("split")], ignore_index=True)
        logger.info("smoke mode: subsampled to %d rows", len(df))
    logger.info("data: %s", df.split.value_counts().to_dict())
    logger.info("target %s: mean %.4f  sd %.4f  range [%.3f, %.3f]",
                D.TARGET_COL, df[D.TARGET_COL].mean(), df[D.TARGET_COL].std(),
                df[D.TARGET_COL].min(), df[D.TARGET_COL].max())

    # Target standardization, fitted on TRAIN only. Reported metrics are
    # always converted back to logit units.
    tr_y = df.loc[df.split == "train", D.TARGET_COL]
    scaler = Standardizer(tr_y.mean(), tr_y.std(),
                          active=not args.no_standardize_target)
    df = df.copy()
    df[D.TARGET_COL] = scaler.forward(df[D.TARGET_COL].to_numpy())
    logger.info("target standardizer: %s", scaler.as_dict())

    datasets = D.build_datasets(df, use_structure=args.use_structure,
                                rnafold_bin=args.rnafold_bin)
    loaders = D.make_loaders(datasets, args.batch_size, args.num_workers)
    for need in ("train", "val"):
        if need not in loaders:
            raise SystemExit(f"split {need!r} missing from the scored data")

    # ---- model ------------------------------------------------------
    model = UpstreamExonModel(
        input_length=args.input_length,
        num_seq_filters=args.num_seq_filters,
        use_structure=args.use_structure,
        head=args.head,
        use_batchnorm=args.batchnorm).to(device)
    logger.info("model: %s", model.describe())

    # One-shot affine head calibration so training starts near the target's
    # mean and scale rather than recovering from the raw SumDiff magnitude.
    ds_tr = datasets["train"]
    k = min(len(ds_tr), 1024)
    cal_struct = (ds_tr.x_struct[:k].to(device) if ds_tr.has_struct else None)
    cal_wob = (ds_tr.x_wobble[:k].to(device) if ds_tr.has_struct else None)
    model.calibrate_head_(ds_tr.x_seq[:k].to(device),
                          ds_tr.y[:k].to(device), cal_struct, cal_wob)

    loss_fn = (nn.MSELoss() if args.loss == "mse"
               else nn.HuberLoss(delta=args.huber_delta))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    start_epoch, best_val, bad = 1, float("inf"), 0
    history: list[dict] = []
    last_ckpt = run_dir / "last.pt"
    best_ckpt = run_dir / "best.pt"
    if args.resume and last_ckpt.exists():
        state = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch = state["epoch"] + 1
        best_val = state["best_val"]
        bad = state.get("bad_epochs", 0)
        history = state.get("history", [])
        logger.info("resumed from epoch %d (best_val=%.6f)",
                    state["epoch"], best_val)

    cfg = {**vars(args), "device": str(device), "model": model.describe(),
           "target_standardizer": scaler.as_dict(),
           "n_rows": len(df),
           "splits": {k: int(v) for k, v in df.split.value_counts().items()},
           "structure_note": ("B1 folded in isolation" if args.use_structure
                              else "sequence-only")}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    # ---- train ------------------------------------------------------
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        tr = run_epoch(model, loaders["train"], device, loss_fn, optimizer,
                       l1_lambda=args.l1_lambda, l2_smooth=args.l2_smooth,
                       scaler=scaler)
        va = run_epoch(model, loaders["val"], device, loss_fn,
                       scaler=scaler)
        history.append({"epoch": epoch, "train_loss": tr["loss"],
                        "train_r2": tr["r2"], "val_loss": va["loss"],
                        "val_r2": va["r2"], "val_mae": va["mae"],
                        "val_pearson": va["pearson_r"]})
        improved = va["loss"] < best_val - 1e-9
        if improved:
            best_val, bad = va["loss"], 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "best_val": best_val, "config": cfg,
                        "scaler": scaler.as_dict()}, best_ckpt)
        else:
            bad += 1
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val": best_val, "bad_epochs": bad,
                    "history": history, "config": cfg}, last_ckpt)
        if epoch % 10 == 0 or epoch == start_epoch or improved:
            logger.info("epoch %4d | train %.5f (R²%.3f) | val %.5f (R²%.3f)%s",
                        epoch, tr["loss"], tr["r2"], va["loss"], va["r2"],
                        "  *" if improved else "")
        if bad >= args.patience:
            logger.info("early stop at epoch %d (patience %d)", epoch, args.patience)
            break
    logger.info("training done in %.1f s", time.time() - t0)

    hist = pd.DataFrame(history)
    hist.to_csv(run_dir / "history.csv", index=False)

    # ---- evaluate ---------------------------------------------------
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device,
                                         weights_only=False)["model_state_dict"])
    # Evaluation loaders are unshuffled, so predictions come back in manifest
    # order and can be joined row-for-row to group ids and composition stats.
    eval_loaders = D.make_loaders(datasets, args.batch_size, args.num_workers,
                                  shuffle_train=False)
    carry = [c for c in ("b1", "b1_id", "b1_sha256", "group_id", "replicate",
                         "gc", "max_homopolymer", "entropy_bits",
                         "max_base_frac", "distinct_3mers") if c in df.columns]

    results: dict[str, dict] = {}
    pred_frames = []
    for split, loader in eval_loaders.items():
        yp, yt, al = predict(model, loader, device, scaler)
        meta = df[df.split == split].reset_index(drop=True)
        if len(meta) != len(yt):
            raise RuntimeError(
                f"split {split}: {len(yt)} predictions vs {len(meta)} rows")
        frame = pd.DataFrame({"split": split, "alpha": al,
                              "y_true": yt, "y_pred": yp})
        for c in carry:
            frame[c] = meta[c].to_numpy()
        pred_frames.append(frame)
        results[split] = {**regression_metrics(yt, yp),
                          **pairwise_difference_metrics(yt, yp),
                          **within_group_metrics(frame)}
    preds = pd.concat(pred_frames, ignore_index=True)
    preds.to_csv(run_dir / "predictions.csv", index=False)

    # how much skill survives removing composition
    if "b1" in preds.columns:
        tr_pred = preds[preds.split == "train"]
        for split in preds.split.unique():
            if split == "train":
                continue
            results[split].update(beyond_composition_metrics(
                tr_pred, preds[preds.split == split], args.kmer_k))

    per_alpha = []
    for s in preds.split.unique():
        t = metrics_by_alpha(preds[preds.split == s])
        t.insert(0, "split", s)
        per_alpha.append(t)
    pd.concat(per_alpha, ignore_index=True).to_csv(
        run_dir / "metrics_by_alpha.csv", index=False)

    by_comp = []
    for s in preds.split.unique():
        t = metrics_by_composition(preds[preds.split == s])
        if not t.empty:
            t.insert(0, "split", s)
            by_comp.append(t)
    if by_comp:
        pd.concat(by_comp, ignore_index=True).to_csv(
            run_dir / "metrics_by_composition.csv", index=False)

    # ---- baselines --------------------------------------------------
    base_rows = []
    if not args.skip_baselines:
        # Baselines are fitted and scored in ORIGINAL logit units so their
        # MSE/MAE/R2 are directly comparable to the model's reported metrics.
        tr_df = df[df.split == "train"]
        for b in BL.all_baselines(args.kmer_k):
            b.fit(tr_df.b1.tolist(),
                  scaler.inverse(tr_df[D.TARGET_COL].to_numpy()))
            for s in ("val", "test"):
                g = df[df.split == s]
                if g.empty:
                    continue
                yp = b.predict(g.b1.tolist())
                base_rows.append({"model": b.name, "split": s,
                                  **regression_metrics(
                                      scaler.inverse(g[D.TARGET_COL].to_numpy()),
                                      yp)})
        for s in ("val", "test"):
            if s in results:
                base_rows.append({"model": "UpstreamExonModel", "split": s,
                                  **{k: v for k, v in results[s].items()
                                     if not k.startswith("pair_")}})
    base = pd.DataFrame(base_rows)
    if not base.empty:
        base.to_csv(run_dir / "baseline_comparison.csv", index=False)

    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    test_key = "test" if "test" in preds.split.unique() else "val"
    plot_run(hist, preds[preds.split == test_key], run_dir)

    # ---- report -----------------------------------------------------
    print("\n" + "=" * 72)
    print(f"RUN {args.run_name}   device={device}   epochs_run={len(hist)}")
    print("=" * 72)
    for s, m in results.items():
        print(f"\n{s}:  n={m['n']}  MSE={m['mse']:.4f}  MAE={m['mae']:.4f}  "
              f"R²={m['r2']:.4f}  Pearson={m['pearson_r']:.4f}  "
              f"Spearman={m['spearman_rho']:.4f}")
        if "pair_sign_agreement" in m:
            print(f"      relative:  pair sign agreement="
                  f"{m['pair_sign_agreement']:.4f}  "
                  f"pair Pearson={m['pair_pearson_r']:.4f}")
        if "within_group_pearson_r" in m:
            print(f"      ORDER (within shared-p groups): "
                  f"Pearson={m['within_group_pearson_r']:+.4f}  "
                  f"Spearman={m['within_group_spearman_rho']:+.4f}  "
                  f"sign={m['within_group_sign_agreement']:.4f}  "
                  f"(true within-group SD {m['within_group_true_sd']:.4f})")
        if "model_r2_on_composition_residual" in m:
            print(f"      BEYOND COMPOSITION: composition-only R²="
                  f"{m['composition_only_r2']:+.4f}  ->  "
                  f"model R² on residual={m['model_r2_on_composition_residual']:+.4f}"
                  f"  (residual r={m.get('residual_pearson_r', float('nan')):+.4f})")
    if not base.empty:
        print("\nBaseline comparison:")
        print(base[["model", "split", "n", "mse", "mae", "r2",
                    "pearson_r", "spearman_rho"]].round(4).to_string(index=False))
    print(f"\nper-alpha metrics -> {run_dir / 'metrics_by_alpha.csv'}")
    print(f"artifacts         -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
