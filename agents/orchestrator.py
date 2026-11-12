"""Orchestrator: điều phối luồng làm việc và vòng lặp "huấn luyện -> tự đánh giá -> làm lại"."""
from __future__ import annotations

import joblib

from config import DATASETS, PROFILE_ONLY

from .base import Agent


class Orchestrator(Agent):
    name = "Orchestrator"

    def __init__(self, bb, cfg, datasets=None):
        super().__init__(bb, cfg)
        self.datasets = datasets or list(DATASETS)
        self.state = {}
        self.pending = set(self.datasets) | set(PROFILE_ONLY)
        self.finished = False

    # ------------------------------------------------------------- kickoff
    def kickoff(self):
        msgs = [self.msg("DataAgent", "REQUEST", "prepare", ds=ds) for ds in self.datasets]
        msgs += [self.msg("DataAgent", "REQUEST", "profile_only", ds=ds) for ds in PROFILE_ONLY]
        return msgs

    # ------------------------------------------------------------ handlers
    def on_data_ready(self, m):
        ds = m.content["ds"]
        self.state[ds] = dict(round=1, strategy=dict(threshold="default", class_weight=None,
                                                      regularize=False, drop=[]))
        return [self.msg("ModelAgent", "REQUEST", "train", ds=ds, round=1,
                         strategy=dict(self.state[ds]["strategy"]))]

    def on_profile_ready(self, m):
        return self._done(m.content["ds"])

    def on_verdict(self, m):
        ds, status, actions = m.content["ds"], m.content["status"], m.content["actions"]
        s = self.state[ds]
        if status == "REVISE":
            for a in actions:
                if a == "tune_threshold":
                    s["strategy"]["threshold"] = "oof_recall"
                elif a == "class_weight_balanced":
                    s["strategy"]["class_weight"] = "balanced"
                    s["strategy"]["threshold"] = "oof_recall"
                elif a == "regularize":
                    s["strategy"]["regularize"] = True
            s["round"] += 1
            self.say(f"{ds}: Agent 2 yêu cầu làm lại (vòng {s['round']}) với {actions}")
            return [self.msg("ModelAgent", "REQUEST", "train", ds=ds, round=s["round"],
                             strategy=dict(s["strategy"]))]
        return self._finalize(ds)

    def on_ablation_done(self, m):
        return self._done(m.content["ds"])

    def on_compare_done(self, m):
        return [self.msg("ReporterAgent", "REQUEST", "report")]

    def on_report_done(self, m):
        self.finished = True
        self.say("Hoàn tất toàn bộ quy trình.")
        return []

    # ------------------------------------------------------------- helpers
    def _finalize(self, ds):
        """Chọn vòng tốt nhất theo (ít FAIL, ít WARN, AUC cao) rồi lưu mô hình."""
        rounds, verdicts = self.bb.rounds[ds], self.bb.verdicts[ds]
        idx = min(range(len(rounds)), key=lambda i: (verdicts[i]["n_fail"], verdicts[i]["n_warn"],
                                                      -rounds[i]["metrics"]["auc"]))
        res, ver = rounds[idx], verdicts[idx]
        self.bb.final[ds] = dict(round=res["round"], index=idx, result=res, verdict=ver)
        mdl = self.bb.models[(ds, res["round"])]
        joblib.dump(dict(ds=ds, **mdl), self.cfg.model_dir / f"{ds}.joblib", compress=3)
        self.say(f"{ds}: chọn vòng {res['round']} ({res['best']}); đã lưu mô hình.")
        if ver["ablation_drop"]:
            return [self.msg("ModelAgent", "REQUEST", "ablation", ds=ds, strategy=dict(res["strategy"]),
                             drop=ver["ablation_drop"])]
        return self._done(ds)

    def _done(self, ds):
        self.pending.discard(ds)
        if not self.pending:
            return [self.msg("ComparatorAgent", "REQUEST", "compare")]
        return []
