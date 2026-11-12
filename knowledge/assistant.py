"""Điều phối ASK: Router -> Graph + Vector -> Fusion -> Answer; ML gọi có điều kiện.

Đây là workflow có nhiều công cụ chuyên biệt (rule-based orchestration), không
phải nhiều LLM tự trị. Không sử dụng một đầu ra xác suất để xác nhận bệnh.
"""
from __future__ import annotations

from .engine import retrieve
from .gemini import enhance_grounded_answer


def ask(question, method="hybrid", top_k=5, hops=2, alpha=0.5, extras=None,
        patient=None, history=None):
    r = retrieve(question, method, top_k, hops, alpha, extras, history)
    generated = enhance_grounded_answer(
        question=question,
        offline_answer=r["answer"],
        snippets=r.get("snippets", []),
        sources=r.get("sources", []),
        history=history,
        allow_without_evidence=r.get("intent") == "smalltalk",
    )
    r["answer"] = generated["answer"]
    r["llm"] = {k: v for k, v in generated.items() if k != "answer"}
    if generated["used"]:
        r["answer_mode"] = "gemini-grounded-rag"
        r["trace"].append({
            "agent": "GeminiExplanationAgent",
            "action": "Sinh câu trả lời tự nhiên từ bằng chứng đã truy xuất",
            "detail": generated["model"],
        })
    else:
        r["trace"].append({
            "agent": "GeminiExplanationAgent",
            "action": "Dùng câu trả lời offline dự phòng",
            "detail": generated["fallback_reason"],
        })
    r["multi_agent_note"] = "Điều phối nhiều tác vụ theo luật; chatbot offline trích xuất (không phải LLM tự trị)."
    if generated["used"]:
        r["multi_agent_note"] = "KG/RAG truy xuất bằng chứng; Gemini Explanation Agent diễn đạt có kiểm tra nguồn."
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
