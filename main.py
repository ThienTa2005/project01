"""Chạy toàn bộ hệ đa agent:  python main.py [--fast] [--datasets pima kaggle100k brfss brfss5050]"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
if sys.platform == "win32" and "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(max(1, (os.cpu_count() or 2) - 1))

from agents.base import Blackboard, MessageBus
from agents.comparator_agent import ComparatorAgent
from agents.data_agent import DataAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.model_agent import ModelAgent
from agents.orchestrator import Orchestrator
from agents.reporter_agent import ReporterAgent
from config import DATASETS, Config


def main():
    ap = argparse.ArgumentParser(description="Hệ đa agent dự đoán & so sánh dataset bệnh tiểu đường")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--fast", action="store_true", help="lấy mẫu dataset lớn để chạy thử nhanh")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    args = ap.parse_args()

    cfg = Config(fast=args.fast)
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.out_dir:
        cfg.out_dir = args.out_dir
    cfg.ensure_dirs()
    warnings.filterwarnings("ignore")

    bb, bus = Blackboard(), MessageBus()
    orch = Orchestrator(bb, cfg, args.datasets)
    for a in (orch, DataAgent(bb, cfg), ModelAgent(bb, cfg), EvaluatorAgent(bb, cfg),
              ComparatorAgent(bb, cfg), ReporterAgent(bb, cfg)):
        bus.register(a)

    t0 = time.time()
    for msg in orch.kickoff():
        bus.send(msg)
    bus.run()
    bus.dump(cfg.out_dir / "agent_trace.jsonl")
    print(f"\nXong trong {time.time()-t0:.0f}s · {len(bus.trace)} message giữa các agent · kết quả: {cfg.out_dir}")


if __name__ == "__main__":
    main()
