"""ComparatorAgent: so sánh sự khác biệt giữa các dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import AGE_COL, BMI_COL, BRFSS_AGE_MID, DATASETS, GLU_COL

from .base import Agent

MAIN = ["pima", "kaggle100k", "brfss"]


def harmonize(bb, ds, split):
    """Trả về DataFrame [age, bmi, (glucose)] có cùng đơn vị giữa các dataset."""
    X = bb.data[ds][f"X_{split}"]
    out = pd.DataFrame({"age": X[AGE_COL[ds]].astype(float), "bmi": X[BMI_COL[ds]].astype(float)})
    if ds.startswith("brfss"):
        out["age"] = X[AGE_COL[ds]].map(BRFSS_AGE_MID).astype(float)
    if ds in GLU_COL:
        out["glucose"] = X[GLU_COL[ds]].astype(float)
    return out


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp)


class ComparatorAgent(Agent):
    name = "ComparatorAgent"

    def on_compare(self, m):
        bb = self.bb
        dss = [d for d in DATASETS if d in bb.final]
        comp = {}

        # 1) Bảng tổng hợp hiệu năng của mô hình cuối
        rows = []
        for ds in dss:
            f = bb.final[ds]["result"]
            mt, mt05 = f["metrics"], f["metrics_default"]
            rows.append(dict(ds=ds, best=f["best"], round=f["round"], auc=mt["auc"], auc_lo=f["auc_ci"][0],
                             auc_hi=f["auc_ci"][1], pr_auc=mt["pr_auc"], recall=mt["recall"],
                             precision=mt["precision"], specificity=mt["specificity"], f2=mt["f2"],
                             accuracy=mt["accuracy"], brier=mt["brier"], ece=mt["ece"],
                             threshold=mt["threshold"], recall_at_05=mt05["recall"], precision_at_05=mt05["precision"],
                             prevalence=bb.profiles[ds]["prevalence"]))
        comp["summary"] = rows

        # 2) Đặc trưng dùng chung (tuổi, BMI): kích thước hiệu ứng giữa nhóm bệnh / không bệnh
        shared = {}
        for ds in dss:
            tr, te = harmonize(bb, ds, "train"), harmonize(bb, ds, "test")
            h = pd.concat([tr, te], ignore_index=True)
            y = np.concatenate([bb.data[ds]["y_train"], bb.data[ds]["y_test"]])
            shared[ds] = {}
            for c in ("age", "bmi"):
                shared[ds][c] = dict(mean_pos=float(h.loc[y == 1, c].mean()), mean_neg=float(h.loc[y == 0, c].mean()),
                                     std=float(h[c].std()), d=cohens_d(h.loc[y == 1, c], h.loc[y == 0, c]),
                                     auc=float(roc_auc_score(y, h[c].fillna(h[c].median()))))
        comp["shared_features"] = shared

        # 3) Chuyển giao chéo dataset (train ở A, test ở B) với đặc trưng chung
        def transfer(ds_list, cols):
            mat = {}
            for a in ds_list:
                mod = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000))
                mod.fit(harmonize(bb, a, "train")[cols], bb.data[a]["y_train"])
                mat[a] = {}
                for b in ds_list:
                    Xb = harmonize(bb, b, "test")[cols]
                    mat[a][b] = float(roc_auc_score(bb.data[b]["y_test"], mod.predict_proba(Xb)[:, 1]))
            return mat

        comp["transfer_age_bmi"] = transfer([d for d in MAIN if d in dss], ["age", "bmi"])
        gl = [d for d in GLU_COL if d in dss]
        if len(gl) == 2:
            comp["transfer_age_bmi_glucose"] = transfer(gl, ["age", "bmi", "glucose"])

        # 4) Đặc trưng quan trọng nhất của từng dataset
        comp["top_features"] = {ds: list(bb.final[ds]["result"]["perm"].items())[:6] for ds in dss}

        # 5) Ablation
        abl = {}
        for ds, r in bb.ablation.items():
            f = bb.final[ds]["result"]
            abl[ds] = dict(dropped=r["strategy"]["drop"], auc_full=f["metrics"]["auc"], auc_abl=r["metrics"]["auc"],
                           pr_full=f["metrics"]["pr_auc"], pr_abl=r["metrics"]["pr_auc"],
                           spec_abl=r["metrics"]["specificity"],
                           recall_full=f["metrics"]["recall"], recall_abl=r["metrics"]["recall"],
                           precision_full=f["metrics"]["precision"], precision_abl=r["metrics"]["precision"],
                           best_abl=r["best"])
        comp["ablation"] = abl

        bb.comparison = comp
        self.say("Đã so sánh xong: hiệu năng, đặc trưng chung, chuyển giao chéo, độ quan trọng, ablation.")
        return [self.msg("Orchestrator", "INFORM", "compare_done")]
