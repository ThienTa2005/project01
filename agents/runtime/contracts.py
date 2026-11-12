"""Structured runtime contracts shared by all decision-support agents."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class AgentResult:
    """Message contract required by the MAS architecture specification."""

    request_id: str
    agent: str
    status: str
    payload: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    version: str = "1.0"
    error: dict[str, Any] | None = None
    started_at: str = field(default_factory=utc_now)
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["duration_ms"] = round(float(self.duration_ms), 3)
        return data


def error_result(
    request_id: str,
    agent: str,
    version: str,
    code: str,
    message: str,
    *,
    status: str = "error",
    payload: dict[str, Any] | None = None,
    duration_ms: float = 0.0,
) -> AgentResult:
    return AgentResult(
        request_id=request_id,
        agent=agent,
        status=status,
        payload=payload or {},
        version=version,
        error={"code": code, "message": message},
        duration_ms=duration_ms,
    )
