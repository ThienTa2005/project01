"""Central orchestration for the complete diabetes decision-support workflow."""
from __future__ import annotations

import re
import time
import uuid
from pathlib import Path

from .contracts import AgentResult
from .dl_prediction_agent import DLPredictionAgent
from .explanation_agent import ExplanationAgent
from .kg_rag_agent import KGRAGAgent
from .monitoring_agent import MonitoringAgent
from .patient_data_agent import PatientDataAgent
from .rl_simulation_agent import RLSimulationAgent
from .rule_based_agent import RuleBasedAgent


class DecisionSupportCoordinator:
    name = "CoordinatorAgent"
    version = "decision-coordinator-v1.0"

    def __init__(self, *, model_path: Path | None = None, audit_path: Path | None = None,
                 persist_audit: bool = True):
        self.data_agent = PatientDataAgent()
        self.dl_agent = DLPredictionAgent(model_path)
        self.rule_agent = RuleBasedAgent()
        self.kg_agent = KGRAGAgent()
        self.explanation_agent = ExplanationAgent()
        self.monitoring_agent = MonitoringAgent(audit_path)
        self.rl_agent = RLSimulationAgent()
        self.persist_audit = persist_audit

    @staticmethod
    def _request_id(candidate: object) -> str:
        if isinstance(candidate, str) and re.fullmatch(r"[A-Za-z0-9_-]{4,80}", candidate):
            return candidate
        return f"req_{uuid.uuid4().hex}"

    @staticmethod
    def _result_map(results: list[AgentResult]) -> dict:
        return {item.agent: item.to_dict() for item in results}

    def process(self, request: object) -> dict:
        workflow_started = time.perf_counter()
        if not isinstance(request, dict):
            request = {}
        request_id = self._request_id(request.get("request_id"))
        results: list[AgentResult] = []

        data_result = self.data_agent.execute(request_id, request.get("patient"))
        results.append(data_result)
        if data_result.status != "ok":
            overall_status = "validation_error"
            monitor = self.monitoring_agent.execute(request_id, results, overall_status,
                                                    persist=self.persist_audit)
            results.append(monitor)
            return {
                "request_id": request_id,
                "status": overall_status,
                "coordinator_version": self.version,
                "report": {"title": "Diabetes Decision-Support Report",
                           "request_id": request_id,
                           "error": data_result.error,
                           "validation": data_result.payload,
                           "limitations": ["Không chạy mô hình khi dữ liệu đầu vào không hợp lệ."]},
                "agents": [item.to_dict() for item in results],
                "agent_results": self._result_map(results),
                "duration_ms": round((time.perf_counter() - workflow_started) * 1000, 3),
            }

        features = data_result.payload["features"]
        dl_result = self.dl_agent.execute(request_id, features)
        results.append(dl_result)
        rule_result = self.rule_agent.execute(request_id, features, dl_result)
        results.append(rule_result)
        kg_result = self.kg_agent.execute(
            request_id,
            str(request.get("question") or ""),
            method=request.get("method", "hybrid"),
            top_k=request.get("top_k", 5),
            hops=request.get("hops", 2),
            alpha=request.get("alpha", 0.5),
            extras=request.get("extra_documents"),
            history=request.get("history"),
        )
        results.append(kg_result)
        explanation = self.explanation_agent.execute(
            request_id,
            data_result,
            dl_result,
            rule_result,
            kg_result,
            question=str(request.get("question") or ""),
            history=request.get("history"),
        )
        results.append(explanation)

        if request.get("simulate_rl"):
            results.append(self.rl_agent.execute(request_id, request.get("simulation_state", {})))

        core_statuses = {dl_result.status, rule_result.status, kg_result.status, explanation.status}
        overall_status = "ok" if core_statuses == {"ok"} else "partial"
        monitor = self.monitoring_agent.execute(request_id, results, overall_status,
                                                persist=self.persist_audit)
        results.append(monitor)
        if monitor.status != "ok" and overall_status == "ok":
            overall_status = "partial"

        return {
            "request_id": request_id,
            "status": overall_status,
            "coordinator_version": self.version,
            "report": explanation.payload,
            "agents": [item.to_dict() for item in results],
            "agent_results": self._result_map(results),
            "duration_ms": round((time.perf_counter() - workflow_started) * 1000, 3),
            "safety": {"clinical_use": False, "diagnosis": False, "prescribing": False,
                       "human_review_required": explanation.payload.get("human_review_required", False)},
        }
