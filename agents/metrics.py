"""Hàm đo lường dùng chung."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, precision_recall_curve,
                             roc_auc_score)


def ece(y, p, bins: int = 10) -> float:
    """Expected Calibration Error."""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def best_threshold_f2(y, p) -> float:
    """Ngưỡng tối đa hóa F2 (ưu tiên recall gấp đôi precision) - phù hợp sàng lọc bệnh."""
    prec, rec, thr = precision_recall_curve(y, p)
    f2 = 5 * prec[:-1] * rec[:-1] / (4 * prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[int(np.argmax(f2))])


def threshold_for_recall(y, p, target: float) -> float:
    """Ngưỡng CAO NHẤT (ít báo động giả nhất) mà trên dữ liệu đã cho vẫn đạt recall >= target."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    i = int(np.argmax(tpr >= target))
    return float(min(max(thr[i], 1e-6), 1.0))


def classification_metrics(y, proba, thr: float = 0.5) -> dict:
    y, proba = np.asarray(y), np.asarray(proba)
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall else 0.0
    f2 = 5 * prec * recall / (4 * prec + recall) if prec + recall else 0.0
    return dict(
        auc=float(roc_auc_score(y, proba)),
        pr_auc=float(average_precision_score(y, proba)),
        brier=float(brier_score_loss(y, proba)),
        ece=ece(y, proba),
        threshold=float(thr),
        accuracy=float((tp + tn) / len(y)),
        balanced_accuracy=float((recall + spec) / 2),
        precision=float(prec), recall=float(recall), specificity=float(spec),
        f1=float(f1), f2=float(f2),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def bootstrap_auc_ci(y, p, n_boot: int = 300, seed: int = 0, alpha: float = 0.05):
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    n, vals = len(y), []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        vals.append(roc_auc_score(y[i], p[i]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
