"""Điều phối ASK: Router -> Graph + Vector -> Fusion -> Answer; ML gọi có điều kiện.

Đây là workflow có nhiều công cụ chuyên biệt (rule-based orchestration), không
phải nhiều LLM tự trị. Không sử dụng một đầu ra xác suất để xác nhận bệnh.
"""
from __future__ import annotations

from .engine import retrieve


def ask(question, method="hybrid", top_k=5, hops=2, alpha=0.5, extras=None,
        patient=None, history=None):
    r = retrieve(question, method, top_k, hops, alpha, extras, history)
    r["multi_agent_note"] = "Điều phối nhiều tác vụ theo luật; chatbot offline trích xuất (không phải LLM tự trị)."
    r["ml_prediction"] = None
    if patient and isinstance(patient, dict):
        try:
            import service
            pred = service.predict(patient)
            r["ml_prediction"] = pred
            r["trace"].append({"agent": "ClinicalPredictorAgent", "action": "Chạy hội đồng 3 mô hình ML theo ngưỡng từng tập", "detail": f"{pred['votes_total']} agent hợp lệ"})
        except ValueError as exc:
            r["trace"].append({"agent": "ClinicalPredictorAgent", "action": "Không chạy do thiếu dữ liệu", "detail": str(exc)})
        except Exception:
            r["trace"].append({"agent": "ClinicalPredictorAgent", "action": "Không chạy vì mô hình không sẵn sàng", "detail": "Kiểm tra models/*.joblib"})
    else:
        r["trace"].append({"agent": "ClinicalPredictorAgent", "action": "Bỏ qua theo chính sách", "detail": "Chỉ chạy nếu người dùng chọn đính kèm chỉ số."})
    return r
