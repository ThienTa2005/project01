"""Lớp dịch vụ dùng chung cho web: nạp mô hình 1 lần, dự đoán bằng "hội đồng" 3 agent."""
from __future__ import annotations

import math
import os
import sys
import threading
import warnings
from pathlib import Path

if sys.platform == "win32" and "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(max(1, (os.cpu_count() or 2) - 1))

import joblib
import numpy as np
import pandas as pd

from config import AGE_COL, BMI_COL, GLU_COL, ROOT

warnings.filterwarnings("ignore")
MODEL_DIR = ROOT / "models"
AGENTS = ("pima", "kaggle100k", "brfss")
AGENT_LABEL = {"pima": "Pima Indians (lâm sàng)", "kaggle100k": "Diabetes 100k (HbA1c + đường huyết)",
               "brfss": "CDC BRFSS 2015 (khảo sát tự khai)"}
BRFSS_AGE_BINS = [(24, 1), (29, 2), (34, 3), (39, 4), (44, 5), (49, 6), (54, 7), (59, 8), (64, 9), (69, 10), (74, 11), (79, 12)]

_YN = [[0, "Không"], [1, "Có"]]


def _f(name, label, kind="number", **kw):
    return dict(name=name, label=label, kind=kind, **kw)


# Schema form - giao diện dựng từ đây nên thêm/bớt trường chỉ cần sửa ở 1 chỗ
SCHEMA = {
    "common": [
        _f("age", "Tuổi (năm)", min=1, max=110, required=True),
        _f("bmi", "BMI (kg/m²)", min=10, max=70, step=0.1),
        _f("glucose", "Đường huyết (mg/dL)", min=30, max=500),
        _f("fasting_glucose", "Đường huyết lúc đói (mg/dL)", min=30, max=500),
        _f("sex", "Giới tính", "select", options=[["female", "Nữ"], ["male", "Nam"]]),
    ],
    "pima": [
        _f("Pregnancies", "Số lần mang thai", min=0, max=20),
        _f("Glucose", "Đường huyết (mg/dL)", min=30, max=300),
        _f("BloodPressure", "Huyết áp tâm trương (mmHg)", min=30, max=140),
        _f("SkinThickness", "Độ dày nếp da (mm)", min=5, max=100),
        _f("Insulin", "Insulin (µU/mL)", min=10, max=900),
        _f("BMI", "BMI", min=10, max=70, step=0.1),
        _f("DiabetesPedigreeFunction", "Chỉ số tiền sử gia đình (DPF)", min=0, max=3, step=0.01),
    ],
    "kaggle100k": [
        _f("hypertension", "Tăng huyết áp", "select", options=_YN),
        _f("heart_disease", "Bệnh tim", "select", options=_YN),
        _f("smoking_history", "Hút thuốc", "select", options=[
            ["never", "Chưa bao giờ"], ["former", "Đã bỏ"], ["current", "Đang hút"],
            ["not current", "Không còn hút"], ["ever", "Từng hút"], ["No Info", "Không rõ"]]),
        _f("hbA1c_level", "HbA1c (%)", min=3, max=15, step=0.1),
        _f("blood_glucose_level", "Đường huyết (mg/dL)", min=50, max=400),
    ],
    "brfss": [
        _f("HighBP", "Huyết áp cao", "select", options=_YN),
        _f("HighChol", "Cholesterol cao", "select", options=_YN),
        _f("CholCheck", "Đã kiểm tra cholesterol (5 năm)", "select", options=_YN),
        _f("Smoker", "Đã hút ≥100 điếu", "select", options=_YN),
        _f("Stroke", "Từng đột quỵ", "select", options=_YN),
        _f("HeartDiseaseorAttack", "Bệnh mạch vành / nhồi máu", "select", options=_YN),
        _f("PhysActivity", "Vận động 30 ngày qua", "select", options=_YN),
        _f("Fruits", "Ăn trái cây hằng ngày", "select", options=_YN),
        _f("Veggies", "Ăn rau hằng ngày", "select", options=_YN),
        _f("HvyAlcoholConsump", "Uống rượu nhiều", "select", options=_YN),
        _f("AnyHealthcare", "Có bảo hiểm y tế", "select", options=_YN),
        _f("NoDocbcCost", "Ngại đi khám vì chi phí", "select", options=_YN),
        _f("DiffWalk", "Khó đi bộ / leo cầu thang", "select", options=_YN),
        _f("GenHlth", "Sức khỏe tự đánh giá", "select", options=[
            [1, "1 - Rất tốt"], [2, "2 - Tốt"], [3, "3 - Bình thường"], [4, "4 - Kém"], [5, "5 - Rất kém"]]),
        _f("MentHlth", "Ngày sức khỏe tinh thần kém /30", min=0, max=30),
        _f("PhysHlth", "Ngày sức khỏe thể chất kém /30", min=0, max=30),
        _f("Education", "Học vấn (1-6)", min=1, max=6),
        _f("Income", "Thu nhập (1-8)", min=1, max=8),
    ],
}

_cache: dict = {}
_lock = threading.Lock()


def load_model(ds: str, model_dir: Path = MODEL_DIR):
    with _lock:
        if ds not in _cache:
            f = model_dir / f"{ds}.joblib"
            _cache[ds] = joblib.load(f) if f.exists() else None
        return _cache[ds]


def age_to_brfss(age: float) -> int:
    for hi, code in BRFSS_AGE_BINS:
        if age <= hi:
            return code
    return 13


def _num(v):
    """Chuyển về số; trả None nếu rỗng/không hợp lệ."""
    if v is None or v == "":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def clean_input(payload: dict) -> dict:
    """Chỉ giữ trường nằm trong SCHEMA; ép kiểu số; bỏ giá trị rỗng/lạ."""
    out = {}
    for group, fields in SCHEMA.items():
        src = payload.get(group) or {}
        if not isinstance(src, dict):
            continue
        dst = {}
        for fld in fields:
            v = src.get(fld["name"])
            if v is None or v == "":
                continue
            if fld["kind"] == "select" and not isinstance(fld["options"][0][0], (int, float)):
                allowed = {o[0] for o in fld["options"]}
                if v in allowed:
                    dst[fld["name"]] = v
            else:
                x = _num(v)
                if x is not None:
                    dst[fld["name"]] = max(fld.get("min", -1e9), min(fld.get("max", 1e9), x)) if fld["kind"] == "number" else x
        out[group] = dst
    return out


def predict(payload: dict) -> dict:
    data = clean_input(payload)
    common = data.get("common", {})
    if "age" not in common:
        raise ValueError("Cần nhập tuổi.")
    results, flags = [], []
    for ds in AGENTS:
        # Pima chỉ gồm phụ nữ trong quần thể nghiên cứu; không áp mô hình trực tiếp cho nam.
        if ds == "pima" and common.get("sex") == "male":
            results.append(dict(agent=ds, label=AGENT_LABEL[ds], status="skipped",
                                reason="Dataset Pima huấn luyện trên phụ nữ; không dùng dự đoán cho người khai giới tính nam."))
            continue
        m = load_model(ds)
        if m is None:
            results.append(dict(agent=ds, label=AGENT_LABEL[ds], status="missing"))
            continue
        rec = dict(data.get(ds, {}))
        for k in ("hypertension", "heart_disease"):  # ép int cho cờ nhị phân
            if k in rec:
                rec[k] = int(rec[k])
        rec.setdefault(AGE_COL[ds], age_to_brfss(common["age"]) if ds == "brfss" else common["age"])
        if "bmi" in common:
            rec.setdefault(BMI_COL[ds], common["bmi"])
        if "glucose" in common and ds in GLU_COL:
            rec.setdefault(GLU_COL[ds], common["glucose"])
        if "sex" in common:
            if ds == "kaggle100k":
                rec.setdefault("gender", "Female" if common["sex"] == "female" else "Male")
            elif ds == "brfss":
                rec.setdefault("Sex", 0 if common["sex"] == "female" else 1)
        feats = m["features"]
        # Cột chủng tộc (race:*) không phải đặc trưng lâm sàng, UI không hỏi -> không tính vào ngưỡng 50%
        asked = [c for c in feats if not c.startswith("race:")]
        have = [c for c in asked if c in rec]
        if len(have) < 0.5 * len(asked):
            results.append(dict(agent=ds, label=AGENT_LABEL[ds], status="skipped",
                                reason=f"Chỉ có {len(have)}/{len(asked)} đặc trưng (cần ≥ 50%)."))
            continue
        X = pd.DataFrame([{c: rec.get(c, np.nan) for c in feats}])
        for c in m["num"]:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        p = float(m["pipe"].predict_proba(X)[:, 1][0])
        high = p >= m["thr"]
        flags.append(high)
        results.append(dict(agent=ds, label=AGENT_LABEL[ds], status="ok", probability=p, threshold=float(m["thr"]),
                            high_risk=bool(high), features_used=len(have), features_total=len(asked), model=m.get("best")))
    votes_high, votes_total = int(sum(flags)), len(flags)
    if not votes_total:
        consensus = "insufficient_data"
    elif votes_high * 2 == votes_total:
        consensus = "tie"
    elif votes_high * 2 > votes_total:
        consensus = "majority_high"
    else:
        consensus = "majority_low"
    return dict(results=results, votes_high=votes_high, votes_total=votes_total,
                consensus=consensus, decision_rule="Bỏ phiếu đa số các agent đủ điều kiện; hòa khi chia đều.",
                caution="Đây là kết quả thuật toán học thuật, không phải xác suất nguy cơ lâm sàng đã kiểm định.")
