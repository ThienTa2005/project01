"""End-to-end tests for the PDF-specified patient-facing MAS workflow."""
import json
from pathlib import Path

from agents.runtime import DecisionSupportCoordinator


def patient_example():
    return json.loads(Path("examples/patient_example.json").read_text(encoding="utf-8"))


def test_complete_runtime_agent_sequence_and_contract():
    result = DecisionSupportCoordinator(persist_audit=False).process({
        "request_id": "req_contract_test",
        "patient": patient_example(),
        "question": "HbA1c bao nhiêu là đái tháo đường?",
    })
    assert result["status"] == "ok"
    names = [item["agent"] for item in result["agents"]]
    assert names == [
        "PatientDataAgent", "DLPredictionAgent", "RuleBasedAgent",
        "KGRAGAgent", "ExplanationAgent", "MonitoringAgent",
    ]
    required = {"request_id", "agent", "status", "payload", "evidence", "version", "error"}
    assert all(required <= set(item) for item in result["agents"])
    assert result["report"]["sources"]
    assert result["report"]["limitations"]
    assert any(flag["rule_id"] == "MODEL_RULE_CONFLICT" for flag in result["report"]["rule_flags"])
    audit_agents = result["agent_results"]["MonitoringAgent"]["payload"]["audit"]["agents"]
    assert all(len(item["payload_sha256"]) == 64 for item in audit_agents)
    assert result["safety"] == {
        "clinical_use": False, "diagnosis": False, "prescribing": False,
        "human_review_required": result["report"]["human_review_required"],
    }


def test_dl_artifact_metadata_and_inference():
    result = DecisionSupportCoordinator(persist_audit=False).process({
        "patient": patient_example(), "question": "Yếu tố nguy cơ tiểu đường là gì?",
    })
    dl = result["agent_results"]["DLPredictionAgent"]
    assert dl["status"] == "ok"
    assert 0 <= dl["payload"]["score"] <= 1
    meta = dl["payload"]["model_metadata"]
    assert "MLP" in meta["architecture"]
    assert meta["metrics"]["auc"] > 0.9
    assert meta["metrics"]["test_rows"] > 1000


def test_validation_terminates_workflow_and_is_audited():
    result = DecisionSupportCoordinator(persist_audit=False).process({
        "patient": {"common": {"age": 999}}, "question": "Tiểu đường là gì?",
    })
    assert result["status"] == "validation_error"
    assert [item["agent"] for item in result["agents"]] == ["PatientDataAgent", "MonitoringAgent"]
    assert result["agent_results"]["PatientDataAgent"]["error"]["code"] == "VALIDATION_FAILED"


def test_optional_rl_agent_is_simulation_only():
    result = DecisionSupportCoordinator(persist_audit=False).process({
        "patient": patient_example(),
        "question": "Cách phòng ngừa đái tháo đường típ 2?",
        "simulate_rl": True,
        "simulation_state": {"risk_band": "high", "engagement": "low"},
    })
    rl = result["agent_results"]["RLSimulationAgent"]
    assert rl["status"] == "ok"
    assert rl["payload"]["simulation_only"] is True
    assert "prescribe_medication" in rl["payload"]["prohibited_uses"]


def test_flask_decision_support_and_integrated_chat(monkeypatch):
    import app as app_module

    monkeypatch.setattr(app_module, "decision_coordinator", DecisionSupportCoordinator(persist_audit=False))
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as client:
        body = {"patient": patient_example(), "question": "HbA1c bao nhiêu là đái tháo đường?"}
        response = client.post("/api/decision-support", json=body)
        assert response.status_code == 200
        assert response.json["agent_results"]["ExplanationAgent"]["status"] == "ok"

        chat = client.post("/api/chat", json={**body, "include_patient": True, "method": "hybrid"})
        assert chat.status_code == 200
        assert chat.json["decision_support"]["status"] == "ok"
        assert chat.json["sources"]
        assert any(item["agent"] == "DLPredictionAgent" for item in chat.json["trace"])
