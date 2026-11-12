"""Dựng lại REPORT.md + biểu đồ từ outputs/results.json (không huấn luyện lại)."""
import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from agents.report_builder import build_report
from config import ROOT, Config

out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs"
R = json.loads((out / "results.json").read_text(encoding="utf-8"))
cfg = Config()
R.setdefault('thr', cfg.thr)
R.setdefault('max_rounds', cfg.max_rounds)
print("Đã ghi", build_report(R, out))
