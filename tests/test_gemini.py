"""Gemini integration tests use a fake provider call and never spend API quota."""
from knowledge import gemini
from knowledge.engine import retrieve


SNIPPETS = [{
    "source_id": "NIDDK-01",
    "text": "HbA1c từ 6,5% là một ngưỡng chẩn đoán và thường cần xác nhận.",
}]
SOURCES = [{"id": "NIDDK-01", "title": "NIDDK – Diabetes Tests & Diagnosis"}]


def test_gemini_disabled_uses_offline_fallback(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_ENABLED", "false")
    result = gemini.enhance_grounded_answer(
        question="HbA1c bao nhiêu là đái tháo đường?",
        offline_answer="Câu trả lời offline [NIDDK-01]",
        snippets=SNIPPETS,
        sources=SOURCES,
    )
    assert result["used"] is False
    assert result["answer"].startswith("Câu trả lời offline")
    assert result["fallback_reason"] == "disabled"


def test_gemini_valid_grounded_answer_is_used(monkeypatch):
    monkeypatch.setenv("GEMINI_ENABLED", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test")
    monkeypatch.setattr(
        gemini,
        "_call_gemini",
        lambda api_key, model, prompt: (
            "HbA1c từ 6,5% là một ngưỡng cần được nhân viên y tế xác nhận [NIDDK-01]."
        ),
    )
    result = gemini.enhance_grounded_answer(
        question="HbA1c bao nhiêu là đái tháo đường?",
        offline_answer="offline",
        snippets=SNIPPETS,
        sources=SOURCES,
    )
    assert result["used"] is True
    assert result["provider"] == "gemini"
    assert result["model"] == "gemini-test"
    assert "[NIDDK-01]" in result["answer"]


def test_gemini_unknown_citation_is_rejected(monkeypatch):
    monkeypatch.setenv("GEMINI_ENABLED", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
    monkeypatch.setattr(
        gemini,
        "_call_gemini",
        lambda api_key, model, prompt: "Nội dung có nguồn không tồn tại [FAKE-99].",
    )
    result = gemini.enhance_grounded_answer(
        question="Câu hỏi",
        offline_answer="offline-safe",
        snippets=SNIPPETS,
        sources=SOURCES,
    )
    assert result["used"] is False
    assert result["answer"] == "offline-safe"
    assert result["fallback_reason"] == "unknown_citation"


def test_smalltalk_router_returns_friendly_answer_without_fake_sources():
    result = retrieve("hi")
    assert result["intent"] == "smalltalk"
    assert "Chào bạn" in result["answer"]
    assert result["sources"] == []
    assert result["snippets"] == []


def test_smalltalk_can_use_gemini_without_evidence(monkeypatch):
    monkeypatch.setenv("GEMINI_ENABLED", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
    monkeypatch.setattr(
        gemini,
        "_call_gemini",
        lambda api_key, model, prompt: (
            "Chào bạn! Mình là DiaBot. Bạn muốn hỏi gì về đái tháo đường?"
        ),
    )
    result = gemini.enhance_grounded_answer(
        question="hi",
        offline_answer="Chào bạn!",
        snippets=[],
        sources=[],
        allow_without_evidence=True,
    )
    assert result["used"] is True
    assert result["provider"] == "gemini"


def test_gemini_status_endpoint_does_not_expose_key(monkeypatch):
    monkeypatch.setenv("GEMINI_ENABLED", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "secret-must-not-leak")
    from app import app

    app.config["TESTING"] = True
    with app.test_client() as client:
        response = client.get("/api/llm/gemini")
    assert response.status_code == 200
    assert response.json["configured"] is True
    assert response.json["available"] is True
    assert "secret-must-not-leak" not in response.get_data(as_text=True)
