"""Knowledge Graph/RAG agent with source-attributed evidence."""
from __future__ import annotations

import time

from knowledge.engine import retrieve

from .contracts import AgentResult, error_result


class KGRAGAgent:
    name = "KGRAGAgent"
    version = "kg-rag-v1.0"

    def execute(self, request_id: str, question: str = "", *, method: str = "hybrid",
                top_k: int = 5, hops: int = 2, alpha: float = 0.5,
                extras=None, history=None) -> AgentResult:
        started = time.perf_counter()
        effective_question = (question or "").strip() or (
            "Các yếu tố nguy cơ và xét nghiệm nào liên quan đến đái tháo đường?"
        )
        try:
            result = retrieve(effective_question, method, top_k, hops, alpha, extras, history)
        except (ValueError, TypeError, OverflowError) as exc:
            return error_result(
                request_id, self.name, self.version, "RETRIEVAL_REQUEST_INVALID", str(exc),
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            return error_result(
                request_id, self.name, self.version, "RETRIEVAL_FAILED",
                f"Truy xuất tri thức thất bại ({exc.__class__.__name__}).",
                duration_ms=(time.perf_counter() - started) * 1000,
            )

        evidence = [{"type": "source", "id": src.get("id"), "title": src.get("title"),
                     "url": src.get("url")} for src in result.get("sources", [])]
        return AgentResult(
            request_id=request_id,
            agent=self.name,
            status="ok",
            payload=result,
            evidence=evidence,
            version=self.version,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
