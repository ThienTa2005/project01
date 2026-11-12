"""Xuất đồ thị để demo trên Neo4j Browser/NetworkX/Gephi.
python scripts/build_kb.py
"""
from pathlib import Path
import sys

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from knowledge.engine import cypher_export, graphml_export, graph_stats

if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "knowledge"
    (out / "kb_diabetes.cypher").write_text(cypher_export(), encoding="utf-8")
    graphml_export(out / "kb_diabetes.graphml")
    print("Đã xuất knowledge/kb_diabetes.cypher + kb_diabetes.graphml; stats:", graph_stats())
