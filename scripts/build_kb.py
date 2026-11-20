"""Build reviewed medical facts into multiple topic-specific Cypher KBs."""
from pathlib import Path
import sys

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from knowledge.engine import authoring_data, cypher_export, graphml_export


KB_GROUPS = {
    "kb_diabetes_core.cypher": {
        "HAS_TYPE", "IS_A", "PRODUCED_BY", "RELATED_TO",
    },
    "kb_diabetes_symptoms.cypher": {"HAS_SYMPTOM"},
    "kb_diabetes_risk_prevention.cypher": {
        "HAS_RISK_FACTOR", "PREVENTION",
    },
    "kb_diabetes_diagnosis.cypher": {
        "ASSESSED_WITH", "REFLECTS", "HAS_PREDIABETES_RANGE",
        "HAS_DIABETES_THRESHOLD", "NOT_SUFFICIENT_FOR_DIAGNOSIS",
    },
    "kb_diabetes_complications.cypher": {
        "HAS_COMPLICATION", "CAN_DAMAGE",
    },
    "kb_diabetes_treatment.cypher": {
        "CAN_BE_TREATED_WITH", "REQUIRES",
    },
}
LEGACY_MONOLITH = "kb_diabetes.cypher"


def partition_data(data):
    """Assign every fact to exactly one topic KB and retain required sources."""
    predicate_to_file = {}
    for filename, predicates in KB_GROUPS.items():
        for predicate in predicates:
            if predicate in predicate_to_file:
                raise ValueError(f"Predicate {predicate} bị khai báo ở nhiều KB.")
            predicate_to_file[predicate] = filename

    actual = {fact["predicate"] for fact in data["facts"]}
    missing = sorted(actual - set(predicate_to_file))
    if missing:
        raise ValueError(f"Các predicate chưa được phân nhóm KB: {missing}")

    source_by_id = {source["id"]: source for source in data["sources"]}
    partitions = {}
    for filename in KB_GROUPS:
        facts = [
            fact for fact in data["facts"]
            if predicate_to_file[fact["predicate"]] == filename
        ]
        source_ids = {fact["source_id"] for fact in facts}
        partitions[filename] = {
            "description": f"KB y khoa theo chủ đề: {filename}",
            "sources": [source_by_id[source_id] for source_id in sorted(source_ids)],
            "facts": facts,
        }
    if sum(len(part["facts"]) for part in partitions.values()) != len(data["facts"]):
        raise ValueError("Số fact sau khi chia KB không khớp dữ liệu biên soạn.")
    return partitions


def main():
    out = Path(__file__).resolve().parent.parent / "knowledge"
    data = authoring_data()
    partitions = partition_data(data)
    fact_file = {}
    for filename, partition in partitions.items():
        target = out / filename
        target.write_text(
            cypher_export(partition, kb_file=filename), encoding="utf-8"
        )
        for fact in partition["facts"]:
            fact_file[fact["id"]] = filename

    legacy = out / LEGACY_MONOLITH
    if legacy.exists() and legacy.name not in partitions:
        legacy.unlink()

    runtime_data = {
        **data,
        "facts": [
            {**fact, "kb_file": fact_file[fact["id"]]}
            for fact in data["facts"]
        ],
    }
    graphml_export(out / "kb_diabetes.graphml", runtime_data)
    print("Đã build các KB Cypher theo chủ đề:")
    for filename, partition in partitions.items():
        print(f"- {filename}: {len(partition['facts'])} facts, "
              f"{len(partition['sources'])} sources")
    print("Tổng:", {"sources": len(data["sources"]), "relations": len(data["facts"])})


if __name__ == "__main__":
    main()
