"""DataAgent: nạp, kiểm tra chất lượng, làm sạch và chia train/test cho từng dataset."""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATASETS, PROFILE_ONLY

from .base import Agent


class DataAgent(Agent):
    name = "DataAgent"

    # ------------------------------------------------------------------ I/O
    def _read(self, fname: str) -> pd.DataFrame:
        p = self.cfg.data_dir / fname
        if p.exists():
            return pd.read_csv(p)
        for z in sorted(self.cfg.data_dir.glob("*.zip")):
            with zipfile.ZipFile(z) as zf:
                for n in zf.namelist():
                    if Path(n).name == fname:
                        return pd.read_csv(zf.open(n))
        raise FileNotFoundError(f"Không tìm thấy {fname} trong {self.cfg.data_dir} (hoặc trong file .zip)")

    # -------------------------------------------------------------- actions
    def on_prepare(self, m):
        ds = m.content["ds"]
        spec = DATASETS[ds]
        df = self._read(spec["file"])
        rows_raw = len(df)
        target = spec["target"]
        prevalence_raw = float(df[target].mean())

        # 1) Phát hiện "thiếu dữ liệu ẩn": giá trị 0 vô lý (vd. Glucose = 0) -> NaN
        hidden = {}
        for c in spec["zero_as_nan"]:
            n0 = int((df[c] == 0).sum())
            hidden[c] = n0
            df.loc[df[c] == 0, c] = np.nan

        # 2) Bỏ cột hành chính, bỏ bản ghi trùng lặp (tránh rò rỉ train/test)
        df = df.drop(columns=spec["drop"])
        dups = int(df.duplicated().sum())
        df = df.drop_duplicates().reset_index(drop=True)

        if self.cfg.fast and len(df) > self.cfg.fast_n:
            df, _ = train_test_split(df, train_size=self.cfg.fast_n, stratify=df[target],
                                     random_state=self.cfg.seed)
            df = df.reset_index(drop=True)

        feats = [c for c in df.columns if c != target]
        cat = list(spec["categorical"])
        num = [c for c in feats if c not in cat]

        X, y = df[feats], df[target].astype(int)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=self.cfg.test_size,
                                              stratify=y, random_state=self.cfg.seed)
        self.bb.data[ds] = dict(
            X_train=Xtr.reset_index(drop=True), X_test=Xte.reset_index(drop=True),
            y_train=ytr.to_numpy(), y_test=yte.to_numpy(), numeric=num, categorical=cat,
        )
        profile = dict(
            ds=ds, label=spec["label"], rows_raw=rows_raw, rows_clean=len(df),
            n_features=len(feats), n_numeric=len(num), n_categorical=len(cat),
            target=target, prevalence_raw=prevalence_raw, prevalence=float(y.mean()),
            duplicates_removed=dups, hidden_zero_missing=hidden,
            missing_total=int(df.isna().sum().sum()) + int(sum(hidden.values())),
            n_train=len(Xtr), n_test=len(Xte), dropped=spec["drop"],
        )
        self.bb.profiles[ds] = profile
        self.say(f"{ds}: {rows_raw:,} -> {len(df):,} dòng sau làm sạch "
                 f"(trùng lặp {dups:,}, 0-ẩn {sum(hidden.values())}), tỉ lệ bệnh {y.mean():.1%}")
        return [self.msg("Orchestrator", "INFORM", "data_ready", ds=ds, profile=profile)]

    def on_profile_only(self, m):
        """Chỉ lập hồ sơ cho file nhãn 3 lớp (brfss012) và đối chiếu với brfss binary."""
        ds = m.content["ds"]
        spec = PROFILE_ONLY[ds]
        df = self._read(spec["file"])
        t = spec["target"]
        counts = df[t].value_counts(normalize=True).sort_index().to_dict()
        info = dict(ds=ds, label=spec["label"], rows=len(df),
                    class_share={int(k): float(v) for k, v in counts.items()})
        try:  # đối chiếu: Diabetes_binary có = (Diabetes_012 == 2) không?
            b = self._read(DATASETS["brfss"]["file"])
            if len(b) == len(df):
                info["binary_equals_012_eq2"] = float(((df[t] == 2).astype(float) == b["Diabetes_binary"]).mean())
                info["binary_equals_012_gt0"] = float(((df[t] > 0).astype(float) == b["Diabetes_binary"]).mean())
        except FileNotFoundError:
            pass
        self.bb.profiles[ds] = info
        self.say(f"{ds}: phân bố 3 lớp {info['class_share']}")
        return [self.msg("Orchestrator", "INFORM", "profile_ready", ds=ds, profile=info)]
