"""ModelAgent: huấn luyện nhiều mô hình ứng viên, chọn tốt nhất bằng CV, áp dụng chiến lược do Agent 2 yêu cầu."""
from __future__ import annotations

import time

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .base import Agent
from .metrics import bootstrap_auc_ci, classification_metrics, threshold_for_recall

MODEL_NAMES = ["logreg", "random_forest", "hist_gb"]
MODEL_LABEL = {"logreg": "Logistic Regression", "random_forest": "Random Forest",
               "hist_gb": "Gradient Boosting"}


def build_pipeline(name, num, cat, class_weight=None, regularize=False, n=1000, seed=42):
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if name == "logreg":
        num_steps.append(("sc", StandardScaler()))
    tf = [("num", Pipeline(num_steps), num)]
    if cat:
        tf.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat))
    pre = ColumnTransformer(tf)

    small = n < 5000
    if name == "logreg":
        clf = LogisticRegression(max_iter=3000, C=0.1 if regularize else 1.0, class_weight=class_weight)
    elif name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=150 if n <= 50000 else 100, class_weight=class_weight, n_jobs=-1, random_state=seed,
            min_samples_leaf=(10 if small else 100) if regularize else (2 if small else 25),
            max_depth=10 if regularize else None,
            max_samples=None if n <= 60000 else 60000 / n)
    else:
        kw = dict(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0,
                  min_samples_leaf=40) if regularize else {}
        clf = HistGradientBoostingClassifier(class_weight=class_weight, random_state=seed, **kw)
    return Pipeline([("pre", pre), ("clf", clf)])


class ModelAgent(Agent):
    name = "ModelAgent"

    def __init__(self, bb, cfg):
        super().__init__(bb, cfg)
        self.cache = {}   # tránh huấn luyện lại khi chỉ đổi ngưỡng

    # -------------------------------------------------------------- actions
    def on_train(self, m):
        ds, strategy, rnd = m.content["ds"], m.content["strategy"], m.content["round"]
        res = self.train(ds, strategy, rnd)
        self.bb.rounds.setdefault(ds, []).append(res)
        return [self.msg("EvaluatorAgent", "INFORM", "evaluate", ds=ds, round=rnd)]

    def on_ablation(self, m):
        ds, strategy, drop = m.content["ds"], dict(m.content["strategy"]), m.content["drop"]
        strategy["drop"] = drop
        strategy["threshold"] = "oof_recall"   # so sánh công bằng ở cùng mức độ nhạy
        res = self.train(ds, strategy, "ablation")
        self.bb.ablation[ds] = res
        self.say(f"{ds}: kiểm tra bỏ {drop} -> AUC {res['metrics']['auc']:.3f}")
        return [self.msg("Orchestrator", "INFORM", "ablation_done", ds=ds)]

    # ----------------------------------------------------------- core logic
    def train(self, ds, strategy, rnd):
        t0 = time.time()
        d = self.bb.data[ds]
        drop = list(strategy.get("drop", []))
        num = [c for c in d["numeric"] if c not in drop]
        cat = [c for c in d["categorical"] if c not in drop]
        cw, reg = strategy.get("class_weight"), bool(strategy.get("regularize"))
        key = (ds, cw, reg, tuple(sorted(drop)))
        if key not in self.cache:
            self.cache[key] = self._fit(ds, num, cat, cw, reg)
        c = self.cache[key]

        y_te = d["y_test"]
        mode = strategy.get("threshold", "default")
        # oof_recall: chọn ngưỡng trên xác suất out-of-fold của tập TRAIN (không dùng tập test)
        thr = 0.5 if mode == "default" else threshold_for_recall(
            d["y_train"], c["oof"], self.cfg.thr["recall_target"] + 0.05)
        metrics = classification_metrics(y_te, c["p_test"], thr)
        metrics_05 = classification_metrics(y_te, c["p_test"], 0.5)
        lo, hi = bootstrap_auc_ci(y_te, c["p_test"], self.cfg.n_boot, self.cfg.seed)
        fpr, tpr, _ = roc_curve(y_te, c["p_test"])
        sel = np.unique(np.linspace(0, len(fpr) - 1, 200).astype(int))

        best = c["best"]
        best_stats = next(s for s in c["stats"] if s["model"] == best)
        res = dict(
            ds=ds, round=rnd, strategy=dict(strategy), features=num + cat, n_train=len(d["y_train"]),
            n_test=len(y_te), candidates=c["stats"], best=best,
            cv_auc=best_stats["cv_auc"], cv_std=best_stats["cv_std"], train_auc=best_stats["train_auc"],
            gap=best_stats["train_auc"] - best_stats["cv_auc"],
            threshold=thr, metrics=metrics, metrics_default=metrics_05, auc_ci=(lo, hi),
            perm=c["perm"], roc=dict(fpr=fpr[sel].tolist(), tpr=tpr[sel].tolist()),
        )
        self.bb.models[(ds, rnd)] = dict(pipe=c["pipe"], thr=thr, features=num + cat, num=num, cat=cat, best=best)
        self.say(f"{ds} vòng {rnd}: {MODEL_LABEL[best]} | CV-AUC {res['cv_auc']:.3f} | test-AUC "
                 f"{metrics['auc']:.3f} | recall {metrics['recall']:.2f} @thr={thr:.2f} ({time.time()-t0:.0f}s)")
        return res

    def _fit(self, ds, num, cat, cw, reg):
        d, seed = self.bb.data[ds], self.cfg.seed
        cols = num + cat
        Xtr, ytr, Xte, yte = d["X_train"][cols], d["y_train"], d["X_test"][cols], d["y_test"]
        n = len(ytr)
        rng = np.random.default_rng(seed)
        skf = StratifiedKFold(self.cfg.cv_folds, shuffle=True, random_state=seed)

        stats, oofs = [], {}
        for name in MODEL_NAMES:
            t0 = time.time()
            oof, f_auc, t_auc = np.zeros(n), [], []
            for tr, va in skf.split(Xtr, ytr):
                pipe = build_pipeline(name, num, cat, cw, reg, len(tr), seed)
                pipe.fit(Xtr.iloc[tr], ytr[tr])
                p = pipe.predict_proba(Xtr.iloc[va])[:, 1]
                oof[va] = p
                f_auc.append(roc_auc_score(ytr[va], p))
                sub = tr if len(tr) <= 20000 else rng.choice(tr, 20000, replace=False)
                t_auc.append(roc_auc_score(ytr[sub], pipe.predict_proba(Xtr.iloc[sub])[:, 1]))
            oofs[name] = oof
            stats.append(dict(model=name, cv_auc=float(np.mean(f_auc)), cv_std=float(np.std(f_auc)),
                              train_auc=float(np.mean(t_auc))))
            self.say(f"  {ds} · {MODEL_LABEL[name]:<20} CV-AUC {np.mean(f_auc):.3f} ± {np.std(f_auc):.3f} "
                     f"(train {np.mean(t_auc):.3f}) [{time.time()-t0:.0f}s]")

        best = max(stats, key=lambda s: s["cv_auc"])["model"]
        pipe = build_pipeline(best, num, cat, cw, reg, n, seed).fit(Xtr, ytr)
        p_test = pipe.predict_proba(Xte)[:, 1]

        # Độ quan trọng đặc trưng (hoán vị trên tập test, thang AUC)
        k = min(len(yte), 5000)
        idx = rng.choice(len(yte), k, replace=False)
        pi = permutation_importance(pipe, Xte.iloc[idx], yte[idx], scoring="roc_auc",
                                    n_repeats=10 if k < 1000 else 3, random_state=seed, n_jobs=1)
        perm = {c: dict(mean=float(m), std=float(s))
                for c, m, s in sorted(zip(cols, pi.importances_mean, pi.importances_std), key=lambda t: -t[1])}
        return dict(stats=stats, best=best, pipe=pipe, p_test=p_test, oof=oofs[best], perm=perm)
