"""ReporterAgent: xuất kết quả, biểu đồ và báo cáo."""
from __future__ import annotations

import json

import numpy as np

from .base import Agent


def to_py(o):
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


class ReporterAgent(Agent):
    name = "ReporterAgent"

    def on_report(self, m):
        bb = self.bb
        out = dict(profiles=bb.profiles, rounds=bb.rounds, verdicts=bb.verdicts,
                   final={k: dict(round=v["round"], index=v["index"]) for k, v in bb.final.items()},
                   n_messages=len(self.bus.trace), fast=self.cfg.fast, thr=self.cfg.thr, max_rounds=self.cfg.max_rounds,
                   ablation=bb.ablation, comparison=bb.comparison)
        out = to_py(out)
        (self.cfg.out_dir / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        from .report_builder import build_report
        path = build_report(out, self.cfg.out_dir)
        self.say(f"Đã xuất báo cáo: {path}")
        return [self.msg("Orchestrator", "INFORM", "report_done")]
