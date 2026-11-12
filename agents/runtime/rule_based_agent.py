"""Transparent educational rules with explicit provenance and review flags."""
from __future__ import annotations

import time

from .contracts import AgentResult


class RuleBasedAgent:
    name = "RuleBasedAgent"
    version = "clinical-rules-v1.0"

    @staticmethod
    def _value(features: dict, *paths):
        for group, field in paths:
            value = features.get(group, {}).get(field)
            if value is not None:
                return float(value)
        return None

    def execute(self, request_id: str, features: dict, dl_result: AgentResult) -> AgentResult:
        started = time.perf_counter()
        flags: list[dict] = []
        evidence: dict[str, dict] = {}

        def add(rule_id, severity, message, rationale, source_id, *, review=False):
            flags.append({"rule_id": rule_id, "severity": severity, "message": message,
                          "rationale": rationale, "source_id": source_id,
                          "requires_human_review": review})
            evidence[source_id] = {"type": "knowledge_source", "id": source_id,
                                   "rationale": rationale}

        hba1c = self._value(features, ("kaggle100k", "hbA1c_level"))
        if hba1c is not None:
            if hba1c >= 6.5:
                add("HBA1C_DIABETES_THRESHOLD", "high",
                    "HbA1c nằm từ ngưỡng 6,5%; cần nhân viên y tế đánh giá và thường cần xét nghiệm xác nhận.",
                    "NIDDK: HbA1c từ 6,5% có thể thuộc ngưỡng chẩn đoán.", "NIDDK-01", review=True)
            elif hba1c >= 5.7:
                add("HBA1C_PREDIABETES_RANGE", "moderate",
                    "HbA1c nằm trong khoảng 5,7%-6,4% được nguồn mô tả là khoảng tiền đái tháo đường.",
                    "NIDDK: khoảng HbA1c 5,7%-6,4%.", "NIDDK-01", review=True)

        fasting = self._value(features, ("common", "fasting_glucose"))
        if fasting is not None and fasting >= 126:
            add("FASTING_GLUCOSE_THRESHOLD", "high",
                "Đường huyết lúc đói từ 126 mg/dL; cần đánh giá chuyên môn và thường cần xác nhận.",
                "NIDDK: glucose huyết tương lúc đói từ 126 mg/dL là một ngưỡng chẩn đoán.",
                "NIDDK-01", review=True)

        bmi = self._value(features, ("common", "bmi"), ("kaggle100k", "bmi"), ("pima", "BMI"))
        if bmi is not None and bmi >= 30:
            add("BMI_OBESITY_RISK", "moderate",
                "BMI thuộc khoảng béo phì và là một yếu tố nguy cơ của đái tháo đường típ 2.",
                "CDC: thừa cân hoặc béo phì làm tăng nguy cơ típ 2.", "CDC-02")
        elif bmi is not None and bmi >= 25:
            add("BMI_OVERWEIGHT_RISK", "low",
                "BMI thuộc khoảng thừa cân; đây là một yếu tố nguy cơ cần xem cùng các yếu tố khác.",
                "CDC: thừa cân hoặc béo phì làm tăng nguy cơ típ 2.", "CDC-02")

        hypertension = self._value(features, ("kaggle100k", "hypertension"), ("brfss", "HighBP"))
        if hypertension == 1:
            add("HYPERTENSION_CONTEXT", "low", "Có khai báo tăng huyết áp; nên được xem trong đánh giá nguy cơ tổng thể.",
                "Thông tin bệnh nền hỗ trợ diễn giải, không tự xác lập chẩn đoán đái tháo đường.", "PATIENT-INPUT")

        if dl_result.status == "ok" and hba1c is not None:
            model_high = bool(dl_result.payload.get("high_risk_flag"))
            threshold_high = hba1c >= 6.5
            if model_high != threshold_high and (threshold_high or hba1c < 5.7):
                add("MODEL_RULE_CONFLICT", "high",
                    "Kết quả mô hình và cờ HbA1c không cùng hướng; bắt buộc chuyển người có chuyên môn xem xét.",
                    "Không gộp hoặc che giấu kết quả mâu thuẫn giữa mô hình và luật.", "SYSTEM-RULE", review=True)

        return AgentResult(
            request_id=request_id,
            agent=self.name,
            status="ok",
            payload={
                "flags": flags,
                "flag_count": len(flags),
                "requires_human_review": any(item["requires_human_review"] for item in flags),
                "review_status": "educational_rules_not_clinically_validated",
                "limitations": "Các luật minh bạch phục vụ nghiên cứu; phải được chuyên gia đủ thẩm quyền thẩm định trước sử dụng lâm sàng.",
            },
            evidence=list(evidence.values()),
            version=self.version,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
