"""Validate and normalize one patient request without hiding invalid input."""
from __future__ import annotations

import math
import time

import service

from .contracts import AgentResult, error_result


class PatientDataAgent:
    name = "PatientDataAgent"
    version = "patient-data-v1.0"

    def execute(self, request_id: str, raw: object) -> AgentResult:
        started = time.perf_counter()
        if not isinstance(raw, dict):
            return error_result(
                request_id, self.name, self.version, "INVALID_PATIENT",
                "patient phải là một JSON object.", status="invalid",
                duration_ms=(time.perf_counter() - started) * 1000,
            )

        errors: list[dict] = []
        warnings: list[dict] = []
        normalized: dict[str, dict] = {}
        known_groups = set(service.SCHEMA)
        unknown_groups = sorted(set(raw) - known_groups)
        if unknown_groups:
            warnings.append({"field": "patient", "message": f"Bỏ qua nhóm không hỗ trợ: {', '.join(unknown_groups)}"})

        for group, fields in service.SCHEMA.items():
            source = raw.get(group, {})
            if source is None:
                source = {}
            if not isinstance(source, dict):
                errors.append({"field": group, "message": "Nhóm dữ liệu phải là JSON object."})
                continue
            output: dict = {}
            allowed_names = {f["name"] for f in fields}
            unknown_fields = sorted(set(source) - allowed_names)
            if unknown_fields:
                warnings.append({"field": group, "message": f"Bỏ qua trường không hỗ trợ: {', '.join(unknown_fields)}"})
            for fld in fields:
                name = fld["name"]
                value = source.get(name)
                if value in (None, ""):
                    if fld.get("required"):
                        errors.append({"field": f"{group}.{name}", "message": "Trường bắt buộc."})
                    continue
                if fld["kind"] == "select" and not isinstance(fld["options"][0][0], (int, float)):
                    allowed = {item[0] for item in fld["options"]}
                    if value not in allowed:
                        errors.append({"field": f"{group}.{name}", "message": "Giá trị lựa chọn không hợp lệ."})
                    else:
                        output[name] = value
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    errors.append({"field": f"{group}.{name}", "message": "Phải là số hợp lệ."})
                    continue
                if not math.isfinite(number):
                    errors.append({"field": f"{group}.{name}", "message": "Phải là số hữu hạn."})
                    continue
                if "min" in fld and number < fld["min"]:
                    errors.append({"field": f"{group}.{name}", "message": f"Không được nhỏ hơn {fld['min']}."})
                    continue
                if "max" in fld and number > fld["max"]:
                    errors.append({"field": f"{group}.{name}", "message": f"Không được lớn hơn {fld['max']}."})
                    continue
                output[name] = number
            normalized[group] = output

        common = normalized.get("common", {})
        for group, field, common_field, tolerance in (
            ("pima", "Age", "age", 1),
            ("kaggle100k", "age", "age", 1),
            ("pima", "BMI", "bmi", 2),
            ("kaggle100k", "bmi", "bmi", 2),
            ("pima", "Glucose", "glucose", 30),
            ("kaggle100k", "blood_glucose_level", "glucose", 30),
        ):
            specific = normalized.get(group, {}).get(field)
            shared = common.get(common_field)
            if specific is not None and shared is not None and abs(specific - shared) > tolerance:
                warnings.append({
                    "field": f"{group}.{field}",
                    "message": f"Không nhất quán đáng kể với common.{common_field}; giữ giá trị chuyên biệt.",
                })

        duration = (time.perf_counter() - started) * 1000
        if errors:
            return error_result(
                request_id, self.name, self.version, "VALIDATION_FAILED",
                "Dữ liệu người dùng không hợp lệ.", status="invalid",
                payload={"errors": errors, "warnings": warnings}, duration_ms=duration,
            )
        provided = sum(len(group) for group in normalized.values())
        return AgentResult(
            request_id=request_id,
            agent=self.name,
            status="ok",
            payload={"features": normalized, "warnings": warnings, "provided_fields": provided},
            evidence=[{"type": "validation_policy", "id": self.version,
                       "rationale": "Kiểm tra kiểu dữ liệu, miền giá trị, trường bắt buộc và tính nhất quán."}],
            version=self.version,
            duration_ms=duration,
        )
