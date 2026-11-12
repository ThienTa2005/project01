"""EvaluatorAgent (Agent 2 - TỰ ĐÁNH GIÁ).

Sau mỗi vòng huấn luyện, agent này kiểm tra kết quả của ModelAgent theo một bộ tiêu chí
(phân biệt, độ nhạy, overfitting, ổn định, hiệu chuẩn, phụ thuộc đặc trưng, thoái lui) rồi
đưa ra phán quyết + hành động khắc phục để Orchestrator yêu cầu ModelAgent làm lại.
"""
from __future__ import annotations

from .base import Agent


class EvaluatorAgent(Agent):
    name = "EvaluatorAgent"

    def on_evaluate(self, m):
        ds, rnd = m.content["ds"], m.content["round"]
        rounds = self.bb.rounds[ds]
        res = rounds[-1]
        prev = rounds[-2] if len(rounds) > 1 else None
        T, st, mt = self.cfg.thr, res["strategy"], res["metrics"]
        checks, actions = [], []

        def add(name, status, detail, action=None):
            checks.append(dict(name=name, status=status, detail=detail))
            if action and action not in actions:
                actions.append(action)

        # 1. Khả năng phân biệt
        add("Phân biệt (AUC test)", "PASS" if mt["auc"] >= T["auc_min"] else "FAIL",
            f"AUC = {mt['auc']:.3f} (yêu cầu ≥ {T['auc_min']})")

        # 2. Độ nhạy - không bỏ sót người bệnh
        if mt["recall"] >= T["recall_target"]:
            add("Độ nhạy (recall)", "PASS", f"recall = {mt['recall']:.2f} (yêu cầu ≥ {T['recall_target']})")
        else:
            if st.get("threshold", "default") == "default":
                act = "tune_threshold"
            elif st.get("class_weight") is None:
                act = "class_weight_balanced"
            else:
                act = None
            add("Độ nhạy (recall)", "FAIL",
                f"recall = {mt['recall']:.2f} < {T['recall_target']} -> đề xuất: {act or 'hết cách khắc phục'}", act)

        # 3. Không báo động tràn lan
        add("Độ đặc hiệu (specificity)", "PASS" if mt["specificity"] >= T["specificity_min"] else "WARN",
            f"specificity = {mt['specificity']:.2f} (yêu cầu ≥ {T['specificity_min']}); precision = {mt['precision']:.2f}")

        # 4. Overfitting
        gap = res["gap"]
        if gap <= T["overfit_gap"]:
            add("Overfitting (AUC train − CV)", "PASS", f"chênh lệch = {gap:.3f}")
        elif not st.get("regularize"):
            add("Overfitting (AUC train − CV)", "FAIL",
                f"chênh lệch = {gap:.3f} > {T['overfit_gap']} -> đề xuất: regularize", "regularize")
        else:
            add("Overfitting (AUC train − CV)", "WARN", f"vẫn còn chênh lệch {gap:.3f} dù đã regularize")

        # 5. Ổn định giữa các fold
        add("Ổn định giữa các fold", "PASS" if res["cv_std"] <= T["cv_std_max"] else "WARN",
            f"độ lệch chuẩn AUC CV = {res['cv_std']:.3f} (≤ {T['cv_std_max']})")

        # 6. Test có khớp CV không (phát hiện rò rỉ / may rủi do tập test nhỏ)
        d = abs(mt["auc"] - res["cv_auc"])
        lo, hi = res["auc_ci"]
        add("Nhất quán test ↔ CV", "PASS" if d <= T["test_cv_gap"] else "WARN",
            f"|AUC test − AUC CV| = {d:.3f}; khoảng tin cậy 95% của AUC test = [{lo:.3f}, {hi:.3f}]")

        # 7. Hiệu chuẩn xác suất
        add("Hiệu chuẩn xác suất (ECE)", "PASS" if mt["ece"] <= T["ece_max"] else "WARN",
            f"ECE = {mt['ece']:.3f} (≤ {T['ece_max']}); Brier = {mt['brier']:.3f}")

        # 8. Phụ thuộc quá mạnh vào một vài đặc trưng -> kiểm tra loại bỏ
        heavy = [c for c, v in res["perm"].items() if v["mean"] >= T["dependence_drop"]][:2]
        if heavy:
            add("Phụ thuộc đặc trưng", "WARN",
                "AUC tụt mạnh khi hoán vị: " + ", ".join(f"{c} (−{res['perm'][c]['mean']:.3f})" for c in heavy)
                + " -> sẽ chạy kiểm tra loại bỏ (ablation)")
        else:
            top = next(iter(res["perm"]))
            add("Phụ thuộc đặc trưng", "PASS",
                f"không đặc trưng nào làm AUC giảm ≥ {T['dependence_drop']} (cao nhất: {top} −{res['perm'][top]['mean']:.3f})")

        # 9. Thoái lui so với vòng trước
        if prev is not None:
            dd = prev["metrics"]["auc"] - mt["auc"]
            add("Thoái lui so với vòng trước", "PASS" if dd <= T["regression_drop"] else "WARN",
                f"AUC vòng trước {prev['metrics']['auc']:.3f} -> {mt['auc']:.3f}")

        n_fail = sum(c["status"] == "FAIL" for c in checks)
        n_warn = sum(c["status"] == "WARN" for c in checks)
        if n_fail == 0:
            status = "ACCEPT"
        elif actions and rnd < self.cfg.max_rounds:
            status = "REVISE"
        else:
            status = "ACCEPT_WITH_ISSUES"
        verdict = dict(ds=ds, round=rnd, status=status, checks=checks, actions=actions,
                       n_fail=n_fail, n_warn=n_warn, ablation_drop=heavy)
        self.bb.verdicts.setdefault(ds, []).append(verdict)
        self.say(f"{ds} vòng {rnd}: {status} (FAIL={n_fail}, WARN={n_warn}) hành động={actions}")
        return [self.msg("Orchestrator", "INFORM", "verdict", ds=ds, round=rnd, status=status,
                         actions=actions, n_fail=n_fail, n_warn=n_warn)]
