"""Dự đoán cho MỘT người bằng "hội đồng" các agent chuyên biệt (mỗi agent = 1 mô hình huấn luyện trên 1 dataset).

    python predict.py --input examples/patient_example.json

File JSON có thể chứa:
  "common": {"age": số năm, "bmi": ..., "glucose": ...}  -> tự ánh xạ sang tên cột của từng dataset
  "pima" / "kaggle100k" / "brfss": {tên_cột: giá_trị}     -> đặc trưng riêng (thiếu thì mô hình tự điền trung vị)
Agent chỉ tham gia nếu có >= 50% đặc trưng. Kết quả CHỈ để tham khảo/nghiên cứu, KHÔNG phải chẩn đoán y khoa.
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
if sys.platform == "win32" and "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(max(1, (os.cpu_count() or 2) - 1))

import joblib
import numpy as np
import pandas as pd

from config import AGE_COL, BMI_COL, GLU_COL, ROOT

BRFSS_AGE_BINS = [(24, 1), (29, 2), (34, 3), (39, 4), (44, 5), (49, 6), (54, 7), (59, 8), (64, 9), (69, 10), (74, 11), (79, 12)]


def age_to_brfss(age):
    for hi, code in BRFSS_AGE_BINS:
        if age <= hi:
            return code
    return 13


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, default=ROOT / "models")
    args = ap.parse_args()
    patient = json.loads(args.input.read_text(encoding="utf-8"))
    common = patient.get("common", {})

    rows, probs = [], []
    for ds in ("pima", "kaggle100k", "brfss"):
        f = args.model_dir / f"{ds}.joblib"
        if not f.exists():
            continue
        m = joblib.load(f)
        rec = dict(patient.get(ds, {}))
        if "age" in common:
            rec.setdefault(AGE_COL[ds], age_to_brfss(common["age"]) if ds == "brfss" else common["age"])
        if "bmi" in common:
            rec.setdefault(BMI_COL[ds], common["bmi"])
        if "glucose" in common and ds in GLU_COL:
            rec.setdefault(GLU_COL[ds], common["glucose"])
        feats = m["features"]
        have = [c for c in feats if c in rec]
        if len(have) < 0.5 * len(feats):
            rows.append((ds, "bỏ qua", f"chỉ có {len(have)}/{len(feats)} đặc trưng", "", ""))
            continue
        X = pd.DataFrame([{c: rec.get(c, np.nan) for c in feats}])
        for c in m["num"]:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        p = float(m["pipe"].predict_proba(X)[:, 1][0])
        flag = "NGUY CƠ CAO" if p >= m["thr"] else "thấp hơn ngưỡng"
        rows.append((ds, f"{p:.1%}", f"ngưỡng {m['thr']:.2f}", flag, f"{len(have)}/{len(feats)} đặc trưng"))
        probs.append(p >= m["thr"])

    print(f"{'Agent (dataset)':<16}{'Xác suất':<12}{'Ngưỡng':<14}{'Kết luận':<18}Dữ liệu cung cấp")
    for r in rows:
        print(f"{r[0]:<16}{r[1]:<12}{r[2]:<14}{r[3]:<18}{r[4]}")
    if probs:
        print(f"\nBỏ phiếu hội đồng: {sum(probs)}/{len(probs)} agent cảnh báo nguy cơ cao.")
    print("Lưu ý: mỗi agent học từ một quần thể khác nhau; kết quả chỉ mang tính tham khảo, không thay thế chẩn đoán y khoa.")


if __name__ == "__main__":
    main()
