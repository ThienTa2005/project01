"""Deep-learning inference agent using the versioned MLP artifact."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import ROOT

from .contracts import AgentResult, error_result


class DLPredictionAgent:
    name = "DLPredictionAgent"
    version = "dl-agent-v1.0"

    def __init__(self, model_path: Path | None = None):
        self.model_path = Path(model_path or ROOT / "models" / "diabetes_dl_v1.joblib")
        self._artifact = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._artifact is None:
                self._artifact = joblib.load(self.model_path)
        return self._artifact

    @staticmethod
    def _record(features: dict) -> dict:
        common = features.get("common", {})
        specific = dict(features.get("kaggle100k", {}))
        if "age" in common:
            specific.setdefault("age", common["age"])
        if "bmi" in common:
            specific.setdefault("bmi", common["bmi"])
        if "glucose" in common:
            specific.setdefault("blood_glucose_level", common["glucose"])
        if "fasting_glucose" in common:
            specific.setdefault("blood_glucose_level", common["fasting_glucose"])
        if "sex" in common:
            specific.setdefault("gender", "Female" if common["sex"] == "female" else "Male")
        return specific

    def execute(self, request_id: str, features: dict) -> AgentResult:
        started = time.perf_counter()
        try:
            artifact = self._load()
        except Exception as exc:
            return error_result(
                request_id, self.name, self.version, "MODEL_UNAVAILABLE",
                f"Không thể nạp mô hình DL ({exc.__class__.__name__}).",
                duration_ms=(time.perf_counter() - started) * 1000,
            )

        required_keys = {"pipe", "features", "num", "cat", "thr", "model_version", "metrics"}
        missing_keys = sorted(required_keys - set(artifact))
        if missing_keys:
            return error_result(
                request_id, self.name, self.version, "INVALID_MODEL_ARTIFACT",
                f"Artifact thiếu metadata: {', '.join(missing_keys)}.",
                duration_ms=(time.perf_counter() - started) * 1000,
            )

        record = self._record(features)
        feature_names = list(artifact["features"])
        present = [name for name in feature_names if record.get(name) not in (None, "")]
        glycemic = any(record.get(name) not in (None, "") for name in ("hbA1c_level", "blood_glucose_level"))
        if len(present) < 4 or not glycemic:
            return error_result(
                request_id, self.name, artifact["model_version"], "INSUFFICIENT_FEATURES",
                "Cần ít nhất 4 đặc trưng và một chỉ số HbA1c/đường huyết để chạy mô hình DL.",
                status="insufficient_data",
                payload={"features_present": present, "features_required": feature_names,
                         "coverage": round(len(present) / max(1, len(feature_names)), 3)},
                duration_ms=(time.perf_counter() - started) * 1000,
            )

        frame = pd.DataFrame([{name: record.get(name, np.nan) for name in feature_names}])
        try:
            for name in artifact["num"]:
                frame[name] = pd.to_numeric(frame[name], errors="coerce")
            probability = float(artifact["pipe"].predict_proba(frame)[:, 1][0])
        except Exception as exc:
            return error_result(
                request_id, self.name, artifact["model_version"], "INFERENCE_FAILED",
                f"Suy luận mô hình thất bại ({exc.__class__.__name__}).",
                duration_ms=(time.perf_counter() - started) * 1000,
            )

        threshold = float(artifact["thr"])
        version = str(artifact["model_version"])
        return AgentResult(
            request_id=request_id,
            agent=self.name,
            status="ok",
            payload={
                "prediction": int(probability >= threshold),
                "score": probability,
                "threshold": threshold,
                "high_risk_flag": bool(probability >= threshold),
                "features_used": present,
                "feature_coverage": round(len(present) / len(feature_names), 3),
                "model_metadata": {
                    "architecture": artifact.get("model_architecture"),
                    "training_dataset": artifact.get("training_dataset"),
                    "preprocessing": artifact.get("preprocessing"),
                    "metrics": artifact.get("metrics"),
                },
                "interpretation": "Điểm mô hình nghiên cứu; không phải xác suất nguy cơ lâm sàng đã hiệu chuẩn.",
            },
            evidence=[{"type": "model_artifact", "id": version,
                       "path": self.model_path.name,
                       "rationale": "Suy luận từ MLP nhiều tầng đã huấn luyện và đánh giá trên tập test tách biệt."}],
            version=version,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
