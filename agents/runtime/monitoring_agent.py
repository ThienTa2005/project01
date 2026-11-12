"""Privacy-minimizing execution audit for the patient-facing MAS."""
from __future__ import annotations

import json
import hashlib
import threading
import time
from pathlib import Path

from config import ROOT

from .contracts import AgentResult, error_result, utc_now


class MonitoringAgent:
    name = "MonitoringAgent"
    version = "monitoring-v1.0"
    _lock = threading.Lock()

    def __init__(self, audit_path: Path | None = None):
        self.audit_path = Path(audit_path) if audit_path is not None else ROOT / "outputs" / "runtime_audit.jsonl"

    def execute(self, request_id: str, results: list[AgentResult], overall_status: str,
                *, persist: bool = True) -> AgentResult:
        started = time.perf_counter()
        def summary(item: AgentResult):
            serialized = json.dumps(item.payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
            return {"agent": item.agent, "status": item.status, "version": item.version,
                    "duration_ms": round(item.duration_ms, 3),
                    "error_code": (item.error or {}).get("code"),
                    "evidence_count": len(item.evidence),
                    "payload_sha256": hashlib.sha256(serialized).hexdigest()}

        record = {
            "request_id": request_id,
            "created_at": utc_now(),
            "overall_status": overall_status,
            "agents": [summary(item) for item in results],
            "total_duration_ms": round(sum(item.duration_ms for item in results), 3),
            "privacy": "Không lưu đặc trưng bệnh nhân hoặc nội dung câu hỏi; chỉ lưu SHA-256 của output để kiểm chứng.",
        }
        if persist:
            try:
                self.audit_path.parent.mkdir(parents=True, exist_ok=True)
                with self._lock, self.audit_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                print("MAS_AUDIT " + json.dumps(record, ensure_ascii=False), flush=True)
            except OSError as exc:
                return error_result(
                    request_id, self.name, self.version, "AUDIT_WRITE_FAILED",
                    f"Không ghi được audit ({exc.__class__.__name__}).", payload={"audit": record},
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
        return AgentResult(
            request_id=request_id,
            agent=self.name,
            status="ok",
            payload={"audit": record, "persisted": bool(persist),
                     "path": self.audit_path.name if persist else None},
            evidence=[{"type": "audit_policy", "id": self.version,
                       "rationale": "Theo dõi trạng thái, phiên bản và thời gian; không ghi dữ liệu bệnh nhân."}],
            version=self.version,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
