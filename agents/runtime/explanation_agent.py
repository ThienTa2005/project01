"""Build an evidence-grounded decision-support report without diagnosing."""
from __future__ import annotations

import json
import time

from knowledge.gemini import enhance_grounded_answer

from .contracts import AgentResult


class ExplanationAgent:
    name = "ExplanationAgent"
    version = "explanation-gemini-fallback-v1.1"

    def execute(self, request_id: str, data_result: AgentResult, dl_result: AgentResult,
                rule_result: AgentResult, kg_result: AgentResult, *,
                question: str = "", history: object = None) -> AgentResult:
        started = time.perf_counter()
        limitations = [
            "Đây là nguyên mẫu giáo dục/nghiên cứu, không phải hệ thống chẩn đoán đã được thẩm định lâm sàng.",
            "Không sử dụng kết quả để kê thuốc, đổi liều hoặc tự thay đổi điều trị.",
            "Điểm mô hình phụ thuộc quần thể và dữ liệu huấn luyện; không phải xác suất nguy cơ cá nhân đã hiệu chuẩn.",
            "Nội dung KG/RAG là bằng chứng hỗ trợ có nguồn, không phải chẩn đoán độc lập.",
        ]
        prediction = dl_result.payload if dl_result.status == "ok" else {
            "status": dl_result.status,
            "reason": (dl_result.error or {}).get("message", "Không có kết quả mô hình."),
        }
        rules = rule_result.payload.get("flags", []) if rule_result.status == "ok" else []
        knowledge = kg_result.payload if kg_result.status == "ok" else {
            "answer": "Không truy xuất được bằng chứng KG/RAG.", "sources": [], "snippets": []
        }
        human_review = bool(rule_result.payload.get("requires_human_review"))
        if dl_result.status == "ok" and dl_result.payload.get("high_risk_flag"):
            human_review = True

        sentences = ["Báo cáo hỗ trợ quyết định học thuật; kết quả không xác nhận một người mắc đái tháo đường."]
        if dl_result.status == "ok":
            sentences.append(
                f"Mô hình DL tạo điểm {dl_result.payload['score']:.3f}, so với ngưỡng nghiên cứu "
                f"{dl_result.payload['threshold']:.3f}; cờ mô hình: "
                f"{'cao hơn ngưỡng' if dl_result.payload['high_risk_flag'] else 'thấp hơn ngưỡng'}."
            )
        else:
            sentences.append(f"Mô hình DL chưa cho kết quả: {(dl_result.error or {}).get('message', dl_result.status)}")
        if rules:
            sentences.append(f"Agent luật ghi nhận {len(rules)} cờ minh bạch; "
                             f"{sum(bool(x['requires_human_review']) for x in rules)} cờ yêu cầu người có chuyên môn xem xét.")
        else:
            sentences.append("Agent luật không tạo cờ từ những trường đã cung cấp; điều này không loại trừ bệnh.")
        sentences.append(knowledge.get("answer", ""))
        if human_review:
            sentences.append("Kết quả nên được nhân viên y tế đủ thẩm quyền xem xét cùng bệnh sử và xét nghiệm xác nhận.")

        evidence = []
        seen = set()
        for result in (dl_result, rule_result, kg_result):
            for item in result.evidence:
                key = (item.get("type"), item.get("id"))
                if key not in seen:
                    seen.add(key)
                    evidence.append(item)

        report = {
            "title": "Diabetes Decision-Support Report",
            "request_id": request_id,
            "narrative": "\n\n".join(x for x in sentences if x),
            "prediction": prediction,
            "rule_flags": rules,
            "retrieved_evidence": knowledge.get("snippets", []),
            "sources": knowledge.get("sources", []),
            "human_review_required": human_review,
            "input_warnings": data_result.payload.get("warnings", []),
            "limitations": limitations,
        }
        agent_context = json.dumps({
            "prediction": prediction,
            "rule_flags": rules,
            "human_review_required": human_review,
            "limitations": limitations,
        }, ensure_ascii=False, default=str)
        generated = enhance_grounded_answer(
            question=question,
            offline_answer=report["narrative"],
            snippets=report["retrieved_evidence"],
            sources=report["sources"],
            history=history,
            extra_context=agent_context,
        )
        report["narrative"] = generated["answer"]
        report["generation"] = {k: v for k, v in generated.items() if k != "answer"}
        return AgentResult(
            request_id=request_id,
            agent=self.name,
            status="ok",
            payload=report,
            evidence=evidence,
            version=self.version,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
