"""Read the project's constrained Cypher KB files into runtime structures.

The runtime deliberately treats ``knowledge/kb_*.cypher`` as its source of
truth.  This is not a general-purpose Cypher parser: it accepts the explicit
MERGE/MATCH statements emitted by :func:`knowledge.engine.cypher_export` and
fails closed when a KB file does not contain usable sources or facts.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path


KB_PATTERN = "kb_*.cypher"
_LITERAL = r"'(?:\\.|[^'\\])*'"
_SOURCE_RE = re.compile(
    rf"^MERGE\s*\(s:Source\s*\{{id:(?P<id>{_LITERAL})\}}\)\s*"
    rf"SET\s*(?P<assignments>.+)$",
    re.IGNORECASE,
)
_FACT_RE = re.compile(
    rf"^MATCH\s*\(a:Entity\s*\{{id:(?P<subject>{_LITERAL})\}}\)\s*,\s*"
    rf"\(b:Entity\s*\{{id:(?P<object>{_LITERAL})\}}\)\s*"
    rf"MERGE\s*\(a\)-\[r:(?P<predicate>[A-Z_]+)\s*"
    rf"\{{fact_id:(?P<fact_id>{_LITERAL})\}}\]->\(b\)\s*"
    rf"SET\s*(?P<assignments>.+)$",
)
_ASSIGNMENT_RE = re.compile(
    rf"(?P<owner>[a-zA-Z]+)\.(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"
    rf"(?P<value>{_LITERAL}|null|true|false|-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def cypher_kb_files(directory: Path) -> list[Path]:
    """Return deterministic KB files; export/helper Cypher files are excluded."""
    return sorted(
        (path for path in Path(directory).glob(KB_PATTERN) if path.is_file()),
        key=lambda path: path.name.casefold(),
    )


def _split_statements(text: str) -> list[str]:
    statements, current = [], []
    quoted = escaped = False
    for raw_line in text.splitlines():
        if not quoted and raw_line.lstrip().startswith("//"):
            continue
        for char in raw_line + "\n":
            if escaped:
                current.append(char)
                escaped = False
                continue
            if quoted and char == "\\":
                current.append(char)
                escaped = True
                continue
            if char == "'":
                quoted = not quoted
                current.append(char)
                continue
            if char == ";" and not quoted:
                statement = " ".join("".join(current).split())
                if statement:
                    statements.append(statement)
                current = []
                continue
            current.append(char)
    tail = " ".join("".join(current).split())
    if tail:
        statements.append(tail)
    if quoted:
        raise ValueError("Chuỗi Cypher chưa đóng dấu nháy đơn.")
    return statements


def _decode(value: str):
    lowered = value.lower()
    if lowered == "null":
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    if not value.startswith("'"):
        return float(value) if "." in value else int(value)
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Literal Cypher không hợp lệ: {value[:80]}") from exc


def _assignments(text: str, owner: str) -> dict:
    result = {}
    for match in _ASSIGNMENT_RE.finditer(text):
        if match.group("owner").lower() == owner.lower():
            result[match.group("key")] = _decode(match.group("value"))
    return result


def parse_cypher_kb(path: Path) -> dict:
    """Parse one project KB and attach its filename to every record."""
    path = Path(path)
    sources, facts = [], []
    for statement in _split_statements(path.read_text(encoding="utf-8")):
        source_match = _SOURCE_RE.match(statement)
        if source_match:
            props = _assignments(source_match.group("assignments"), "s")
            source = {"id": _decode(source_match.group("id")), **props}
            source.setdefault("title", source["id"])
            source.setdefault("publisher", "Không ghi trong KB Cypher")
            source.setdefault("url", "")
            source.setdefault("summary", source["title"])
            source.setdefault("kind", "cypher")
            source["kb_files"] = [path.name]
            sources.append(source)
            continue

        fact_match = _FACT_RE.match(statement)
        if fact_match:
            props = _assignments(fact_match.group("assignments"), "r")
            fact = {
                "id": _decode(fact_match.group("fact_id")),
                "subject": _decode(fact_match.group("subject")),
                "predicate": fact_match.group("predicate"),
                "object": _decode(fact_match.group("object")),
                "source_id": props.get("source_id"),
                "evidence": props.get("evidence", ""),
                "kb_file": path.name,
            }
            if not fact["source_id"] or not fact["evidence"]:
                raise ValueError(
                    f"{path.name}: fact {fact['id']} thiếu source_id hoặc evidence."
                )
            facts.append(fact)

        elif statement.upper().startswith("MERGE (S:SOURCE"):
            raise ValueError(f"{path.name}: câu lệnh Source không đúng định dạng hỗ trợ.")
        elif statement.upper().startswith("MATCH (A:ENTITY") and "MERGE (A)-[R:" in statement.upper():
            raise ValueError(f"{path.name}: câu lệnh fact không đúng định dạng hỗ trợ.")

    if not sources and not facts:
        raise ValueError(
            f"{path.name} không có Source hoặc fact theo định dạng KB Cypher của dự án."
        )
    return {"sources": sources, "facts": facts, "kb_files": [path.name]}


def load_cypher_kbs(directory: Path) -> dict:
    """Merge all KB Cypher files and reject conflicting IDs/provenance."""
    paths = cypher_kb_files(directory)
    if not paths:
        raise FileNotFoundError(f"Không tìm thấy {KB_PATTERN} trong {Path(directory)}")

    source_by_id, fact_by_id = {}, {}
    for path in paths:
        parsed = parse_cypher_kb(path)
        for source in parsed["sources"]:
            previous = source_by_id.get(source["id"])
            if previous:
                for key in set(previous) | set(source):
                    if key in {"id", "kb_files"}:
                        continue
                    old, new = previous.get(key), source.get(key)
                    if old not in {None, ""} and new not in {None, ""} and old != new:
                        raise ValueError(
                            f"Source ID {source['id']} xung đột thuộc tính {key} "
                            "giữa các file Cypher."
                        )
                    if old in {None, ""} and new not in {None, ""}:
                        previous[key] = new
                previous["kb_files"] = sorted(
                    set(previous["kb_files"] + source["kb_files"])
                )
            else:
                source_by_id[source["id"]] = source

        for fact in parsed["facts"]:
            previous = fact_by_id.get(fact["id"])
            if previous and previous != fact:
                raise ValueError(f"Fact ID {fact['id']} bị trùng/xung đột trong KB Cypher.")
            fact_by_id[fact["id"]] = fact

    missing = sorted(
        {fact["source_id"] for fact in fact_by_id.values()} - set(source_by_id)
    )
    if not source_by_id or not fact_by_id:
        raise ValueError("Tập KB Cypher phải có ít nhất một Source và một fact.")
    if missing:
        raise ValueError(f"Fact tham chiếu Source chưa khai báo trong Cypher: {missing}")

    return {
        "description": (
            "Kho tri thức runtime được nạp trực tiếp từ các file kb_*.cypher; "
            "mỗi fact giữ provenance nguồn và tên file KB."
        ),
        "sources": list(source_by_id.values()),
        "facts": list(fact_by_id.values()),
        "kb_files": [path.name for path in paths],
        "runtime_source": "cypher",
    }
