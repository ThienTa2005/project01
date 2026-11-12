"""Dựng biểu đồ + báo cáo Markdown (tiếng Việt) từ results.json.

Có thể chạy độc lập:  python make_report.py   (không cần huấn luyện lại)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from config import DATASETS, PROFILE_ONLY  # noqa: E402

# ---- Bảng màu (xanh dương / cam / ngọc / vàng - đã kiểm tra CVD), màu theo thực thể (dataset)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
COLOR = {"pima": "#2a78d6", "kaggle100k": "#eb6834", "brfss": "#1baf7a", "brfss5050": "#1baf7a"}
STYLE = {"pima": "-", "kaggle100k": "-", "brfss": "-", "brfss5050": "--"}
MODEL_VN = {"logreg": "Logistic Regression", "random_forest": "Random Forest", "hist_gb": "Gradient Boosting"}
SHORT = {k: v["short"] for k, v in DATASETS.items()}
FULL = {k: v["label"] for k, v in DATASETS.items()}


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False,
    })


def _save(fig, path):
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# ================================================================= FIGURES
def fig_roc(R, path):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.5, zorder=1)
    for ds in R["datasets"]:
        roc = R["rounds"][ds][R["final"][ds]["index"]]["roc"]
        auc = R["rounds"][ds][R["final"][ds]["index"]]["metrics"]["auc"]
        ax.plot(roc["fpr"], roc["tpr"], color=COLOR[ds], ls=STYLE[ds], lw=2, label=f"{SHORT[ds]}  (AUC {auc:.3f})")
    ax.set(xlabel="Tỉ lệ dương tính giả (1 − specificity)", ylabel="Độ nhạy (recall)", xlim=(0, 1), ylim=(0, 1.01),
           title="Đường ROC của mô hình cuối cùng (tập test)")
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    _save(fig, path)


def fig_self_eval(R, path):
    dss = R["datasets"]
    first = [R["rounds"][d][0]["metrics"]["recall"] for d in dss]
    last = [R["rounds"][d][R["final"][d]["index"]]["metrics"]["recall"] for d in dss]
    x = np.arange(len(dss))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    b1 = ax.bar(x - w / 2 - 0.01, first, w, color="#2a78d6", label="Vòng 1 (ngưỡng 0.5)")
    b2 = ax.bar(x + w / 2 + 0.01, last, w, color="#eb6834", label="Sau khi Agent 2 tự đánh giá")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012, f"{b.get_height():.2f}",
                    ha="center", va="bottom", fontsize=9, color=INK)
    ax.axhline(R["thr"]["recall_target"], color=INK2, ls=":", lw=1.4, label=f"Mục tiêu recall ≥ {R['thr']['recall_target']:.2f}")
    ax.set_xticks(x, [SHORT[d] for d in dss])
    ax.set(ylim=(0, 1.1), ylabel="Độ nhạy (recall) trên tập test", title="Agent 2 tự sửa mô hình để đạt độ nhạy mục tiêu")
    ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0, -0.1), fontsize=9)
    ax.grid(axis="x", visible=False)
    _save(fig, path)


def fig_transfer(R, path):
    mats = [("Đặc trưng: tuổi + BMI", R["comparison"]["transfer_age_bmi"])]
    if "transfer_age_bmi_glucose" in R["comparison"]:
        mats.append(("Đặc trưng: tuổi + BMI + đường huyết", R["comparison"]["transfer_age_bmi_glucose"]))
    cmap = LinearSegmentedColormap.from_list("blue", ["#eaf1fb", "#2a78d6", "#0f3c78"])
    fig, axes = plt.subplots(1, len(mats), figsize=(4.6 * len(mats) + 0.6, 4.2))
    axes = np.atleast_1d(axes)
    for ax, (title, mat) in zip(axes, mats):
        names = list(mat)
        Z = np.array([[mat[a][b] for b in names] for a in names])
        ax.imshow(Z, cmap=cmap, vmin=0.5, vmax=1.0)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{Z[i, j]:.3f}", ha="center", va="center", fontsize=11,
                        color="white" if Z[i, j] > 0.72 else INK, fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(len(names)), [SHORT[n] for n in names])
        ax.set_yticks(range(len(names)), [SHORT[n] for n in names])
        ax.set_xlabel("Test trên dataset"); ax.set_ylabel("Huấn luyện trên dataset")
        ax.set_title(title); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Chuyển giao chéo dataset – ROC-AUC (đường chéo = cùng dataset)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def fig_importance(R, path):
    dss = R["datasets"]
    fig, axes = plt.subplots(1, len(dss), figsize=(3.6 * len(dss), 3.8))
    for ax, ds in zip(np.atleast_1d(axes), dss):
        items = R["comparison"]["top_features"][ds][:6][::-1]
        ax.barh([k for k, _ in items], [v["mean"] for _, v in items], color=COLOR[ds], height=0.62)
        for i, (_, v) in enumerate(items):
            ax.text(v["mean"], i, f" {v['mean']:.3f}", va="center", fontsize=8.5, color=INK2)
        ax.set_title(SHORT[ds]); ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(v["mean"] for _, v in items) * 1.25)
        ax.tick_params(axis="y", labelsize=8.5)
    fig.suptitle("Độ quan trọng đặc trưng (mức AUC giảm khi hoán vị)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def fig_effect(R, path):
    dss = R["datasets"]
    sf = R["comparison"]["shared_features"]
    x, w = np.arange(len(dss)), 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for off, key, col, lab in ((-w / 2 - 0.01, "bmi", "#2a78d6", "BMI"), (w / 2 + 0.01, "age", "#eb6834", "Tuổi")):
        vals = [sf[d][key]["d"] for d in dss]
        bars = ax.bar(x + off, vals, w, color=col, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x, [SHORT[d] for d in dss])
    ax.set(ylabel="Cohen's d (nhóm bệnh − không bệnh)", title="Tuổi và BMI phân tách người bệnh mạnh/yếu ra sao ở từng dataset")
    ax.legend(loc="upper left"); ax.grid(axis="x", visible=False)
    _save(fig, path)


def fig_ablation(R, path):
    abl = R["comparison"]["ablation"]
    dss = [d for d in R["datasets"] if d in abl]
    if not dss:
        return False
    x, w = np.arange(len(dss)), 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    full = [abl[d]["auc_full"] for d in dss]
    cut = [abl[d]["auc_abl"] for d in dss]
    b1 = ax.bar(x - w / 2 - 0.01, full, w, color="#2a78d6", label="Đầy đủ đặc trưng")
    b2 = ax.bar(x + w / 2 + 0.01, cut, w, color="#eb6834", label="Bỏ đặc trưng mà Agent 2 nghi ngờ")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.006, f"{b.get_height():.3f}", ha="center", fontsize=9)
    ax.set_xticks(x, [f"{SHORT[d]}\nbỏ " + "\nbỏ ".join(abl[d]["dropped"]) for d in dss], fontsize=8.5)
    ax.set(ylim=(0.5, 1.03), ylabel="ROC-AUC (test)", title="Kiểm tra loại bỏ đặc trưng (ablation)")
    ax.legend(loc="upper right", ncol=1); ax.grid(axis="x", visible=False)
    _save(fig, path)
    return True


# ================================================================= MARKDOWN
def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def strat_vn(s, T):
    parts = ["ngưỡng 0.5" if s.get("threshold", "default") == "default"
             else f"ngưỡng tối ưu từ OOF (recall ≥ {T['recall_target'] + 0.05:.2f})"]
    if s.get("class_weight"):
        parts.append("class_weight=balanced")
    if s.get("regularize"):
        parts.append("regularize")
    if s.get("drop"):
        parts.append("bỏ " + ", ".join(s["drop"]))
    return "; ".join(parts)


def build_report(R, out_dir: Path):
    _style()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    R["datasets"] = [d for d in DATASETS if d in R["final"]]
    dss, C, T = R["datasets"], R["comparison"], R["thr"]
    fin = {d: R["rounds"][d][R["final"][d]["index"]] for d in dss}
    summ = {r["ds"]: r for r in C["summary"]}

    fig_roc(R, fig_dir / "fig_roc.png")
    fig_self_eval(R, fig_dir / "fig_self_eval.png")
    fig_transfer(R, fig_dir / "fig_transfer.png")
    fig_importance(R, fig_dir / "fig_importance.png")
    fig_effect(R, fig_dir / "fig_effect.png")
    has_abl = fig_ablation(R, fig_dir / "fig_ablation.png")

    L = []
    w = L.append
    mains = [d for d in dss if DATASETS[d]["role"] == "main"]
    ranking = sorted(mains, key=lambda d: -summ[d]["auc"])

    w("# Hệ đa agent dự đoán bệnh tiểu đường và so sánh 3 bộ dữ liệu\n")
    w(f"> Báo cáo được **sinh tự động** bởi `ReporterAgent` từ kết quả chạy thật"
      f"{' (chế độ --fast: dataset lớn bị lấy mẫu, số liệu chỉ để thử nghiệm)' if R.get('fast') else ''}. "
      f"Toàn bộ quá trình phối hợp giữa các agent ({R['n_messages']} message) nằm trong `agent_trace.jsonl`.\n")

    # ------------------------------------------------------------ 1
    w("## 1. Kiến trúc hệ đa agent\n")
    w(md_table(["Agent", "Vai trò", "Nhận → Gửi"], [
        ["**Orchestrator**", "Điều phối quy trình; quyết định làm lại hay chốt kết quả dựa trên phán quyết của Agent 2", "`data_ready`, `verdict` → `train`, `ablation`, `compare`"],
        ["**DataAgent**", "Nạp dữ liệu, phát hiện giá trị thiếu ẩn (số 0 vô lý), bỏ bản ghi trùng lặp, bỏ cột hành chính, chia train/test phân tầng", "`prepare` → `data_ready`"],
        ["**ModelAgent**", "Huấn luyện 3 mô hình ứng viên (Logistic Regression, Random Forest, Gradient Boosting) bằng CV 5-fold, chọn mô hình tốt nhất, áp dụng chiến lược được yêu cầu (ngưỡng, class_weight, regularize, bỏ đặc trưng)", "`train`, `ablation` → `evaluate`"],
        ["**EvaluatorAgent (Agent 2 – tự đánh giá)**", "Kiểm tra kết quả theo 9 tiêu chí (AUC, recall, specificity, overfitting, ổn định, nhất quán test–CV, hiệu chuẩn, phụ thuộc đặc trưng, thoái lui); ra phán quyết `ACCEPT / REVISE` và hành động khắc phục", "`evaluate` → `verdict`"],
        ["**ComparatorAgent**", "So sánh dataset: hiệu năng, đặc trưng chung (tuổi, BMI), chuyển giao chéo, độ quan trọng đặc trưng, ablation", "`compare` → `compare_done`"],
        ["**ReporterAgent**", "Xuất biểu đồ, bảng và báo cáo này", "`report` → `report_done`"],
    ]))
    w("""
```
Orchestrator ──prepare──▶ DataAgent ──data_ready──▶ Orchestrator
Orchestrator ──train────▶ ModelAgent ──evaluate──▶ EvaluatorAgent (Agent 2)
EvaluatorAgent ──verdict──▶ Orchestrator ──REVISE: train lại (ngưỡng / class_weight / regularize)──▶ ModelAgent
                                         └─ACCEPT: lưu mô hình, (ablation nếu bị nghi ngờ) ─▶ ComparatorAgent ─▶ ReporterAgent
```
""")
    w("**Cơ chế tự đánh giá (vòng lặp):** sau mỗi vòng huấn luyện, Agent 2 chấm điểm kết quả. Nếu có tiêu chí **FAIL** và còn hành động khắc phục, "
      f"Orchestrator yêu cầu ModelAgent làm lại (tối đa {R['max_rounds']} vòng). Vòng cuối được chọn theo thứ tự ưu tiên: ít FAIL → ít WARN → AUC cao. "
      "Ngưỡng phân loại chỉ được chọn trên xác suất out-of-fold của **tập train**, tập test không bao giờ được dùng để tinh chỉnh.\n")

    # ------------------------------------------------------------ 2
    w("## 2. Ba bộ dữ liệu\n")
    w("Bộ dữ liệu thứ 3 là `archive.zip`, chứa 3 file của khảo sát **CDC BRFSS 2015**: `binary` (nhãn 0/1, dùng làm bản chính), "
      "`5050split` (tập con cân bằng 50/50, chạy như một biến thể để xem tác động của việc cân bằng lớp) và `012` (nhãn 3 lớp, chỉ để mô tả).\n")
    rows = []
    for d in dss:
        p = R["profiles"][d]
        hid = sum(p["hidden_zero_missing"].values())
        rows.append([f"**{FULL[d]}**", f"{p['rows_raw']:,}", f"{p['rows_clean']:,}", p["n_features"],
                     f"{pct(p['prevalence_raw'])} → {pct(p['prevalence'])}", f"{p['duplicates_removed']:,}",
                     f"{hid:,}" if hid else "0", DATASETS[d]["collection"]])
    w(md_table(["Dataset", "Dòng gốc", "Sau làm sạch", "Số đặc trưng", "Tỉ lệ bệnh (gốc → sau)", "Trùng lặp đã bỏ", "Số 0 vô lý → thiếu", "Cách thu thập"], rows))
    w("")
    w(md_table(["Dataset", "Quần thể", "Định nghĩa nhãn"], [[f"**{SHORT[d]}**", DATASETS[d]["population"], DATASETS[d]["label_def"]] for d in dss]))
    p12 = R["profiles"].get("brfss012")
    if p12:
        cs = p12["class_share"]
        line = (f"\n**File `012`**: 0 = {pct(cs['0'])}, 1 (tiền đái tháo đường) = {pct(cs['1'])}, 2 (đái tháo đường) = {pct(cs['2'])}. ")
        if "binary_equals_012_eq2" in p12:
            line += (f"Đối chiếu từng dòng với file `binary`: nhãn `binary` trùng với `(012 == 2)` ở **{pct(p12['binary_equals_012_eq2'], 2)}** số dòng "
                     f"(và chỉ {pct(p12['binary_equals_012_gt0'], 2)} nếu coi tiền đái tháo đường là 1) – tức là trong bản `binary`, **người tiền đái tháo đường được tính là 0**.")
        w(line + "\n")
    pm = R["profiles"].get("pima")
    if pm:
        w("**Lưu ý Pima:** số giá trị 0 *không thể xảy ra về mặt sinh học* (vd. đường huyết = 0): "
          + ", ".join(f"{k} {v}" for k, v in pm["hidden_zero_missing"].items())
          + ". Chúng được coi là thiếu và điền trung vị **bên trong pipeline** (chỉ học từ tập train để không rò rỉ).\n")

    # ------------------------------------------------------------ 3
    w("## 3. Kết quả dự đoán (mô hình cuối, tập test)\n")
    rows = []
    for d in dss:
        s = summ[d]
        rows.append([f"**{SHORT[d]}**", MODEL_VN[s["best"]], f"{s['auc']:.3f} [{s['auc_lo']:.3f}–{s['auc_hi']:.3f}]", f"{s['pr_auc']:.3f}",
                     f"{s['recall']:.2f}", f"{s['precision']:.2f}", f"{s['specificity']:.2f}", f"{s['f2']:.2f}", f"{s['accuracy']:.2f}",
                     f"{s['brier']:.3f}", f"{s['threshold']:.2f}", pct(s["prevalence"])])
    w(md_table(["Dataset", "Mô hình", "ROC-AUC [KTC 95%]", "PR-AUC", "Recall", "Precision", "Specificity", "F2", "Accuracy", "Brier", "Ngưỡng", "Tỉ lệ bệnh"], rows))
    w("\n![ROC](figures/fig_roc.png)\n")
    w("*Precision và PR-AUC phụ thuộc mạnh vào tỉ lệ bệnh của từng dataset (precision ngẫu nhiên = tỉ lệ bệnh), vì vậy chỉ **ROC-AUC** mới so sánh trực tiếp được giữa các dataset.*\n")

    cand_rows = []
    for d in dss:
        for c in fin[d]["candidates"]:
            cand_rows.append([SHORT[d], MODEL_VN[c["model"]], f"{c['cv_auc']:.3f} ± {c['cv_std']:.3f}", f"{c['train_auc']:.3f}",
                              "✔" if c["model"] == fin[d]["best"] else ""])
    w("<details><summary>Điểm CV của các mô hình ứng viên (vòng được chọn)</summary>\n")
    w(md_table(["Dataset", "Mô hình", "AUC CV (5-fold)", "AUC train", "Được chọn"], cand_rows))
    w("\n</details>\n")

    # ------------------------------------------------------------ 4
    w("## 4. Agent 2 – Tự đánh giá\n")
    w("![self-eval](figures/fig_self_eval.png)\n")
    for d in dss:
        w(f"### {FULL[d]}\n")
        rows = []
        for i, (r, v) in enumerate(zip(R["rounds"][d], R["verdicts"][d])):
            m = r["metrics"]
            mark = " ⬅ chọn" if i == R["final"][d]["index"] else ""
            rows.append([r["round"], strat_vn(r["strategy"], T), MODEL_VN[r["best"]], f"{m['auc']:.3f}", f"{m['recall']:.2f}",
                         f"{m['precision']:.2f}", f"{m['specificity']:.2f}", f"{v['status']} (FAIL {v['n_fail']}, WARN {v['n_warn']}){mark}"])
        w(md_table(["Vòng", "Chiến lược", "Mô hình", "AUC", "Recall", "Precision", "Specificity", "Phán quyết"], rows))
        v = R["verdicts"][d][R["final"][d]["index"]]
        icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}
        w("\nChi tiết các tiêu chí ở vòng được chọn:\n")
        w(md_table(["", "Tiêu chí", "Chi tiết"], [[icon[c["status"]], c["name"], c["detail"]] for c in v["checks"]]))
        w("")

    # ------------------------------------------------------------ 5
    w("## 5. So sánh sự khác biệt giữa các dataset\n")
    w("### 5.1 Khác biệt về bản chất dữ liệu\n")
    w(md_table(["Khía cạnh", "Pima", "100k", "BRFSS"], [
        ["Quy mô", f"{R['profiles']['pima']['rows_clean']:,}", f"{R['profiles']['kaggle100k']['rows_clean']:,}", f"{R['profiles']['brfss']['rows_clean']:,}"],
        ["Tỉ lệ bệnh", pct(R['profiles']['pima']['prevalence']), pct(R['profiles']['kaggle100k']['prevalence']), pct(R['profiles']['brfss']['prevalence'])],
        ["Loại đặc trưng", "Chỉ số lâm sàng/xét nghiệm (đường huyết, insulin, huyết áp…), tiền sử gia đình, số lần mang thai", "Xét nghiệm HbA1c + đường huyết, BMI, tuổi, bệnh nền, hút thuốc, giới, chủng tộc", "Hành vi & tự khai: huyết áp/mỡ máu cao, vận động, ăn rau quả, sức khỏe tự đánh giá, thu nhập, học vấn…"],
        ["Có xét nghiệm?", "Có", "Có (HbA1c, đường huyết)", "Không – hoàn toàn tự khai"],
        ["Quần thể", "Chỉ nữ, một nhóm dân tộc", "Hỗn hợp, nghi tổng hợp", "Người trưởng thành Mỹ, đại diện quốc gia"],
        ["Mã hóa tuổi", "Số năm", "Số năm", "13 nhóm tuổi"],
    ]))
    w("")
    w("### 5.2 Đặc trưng chung: tuổi & BMI\n")
    sf = C["shared_features"]
    rows = []
    for d in dss:
        rows.append([SHORT[d], f"{sf[d]['age']['mean_neg']:.1f} → {sf[d]['age']['mean_pos']:.1f}", f"{sf[d]['age']['d']:.2f}", f"{sf[d]['age']['auc']:.3f}",
                     f"{sf[d]['bmi']['mean_neg']:.1f} → {sf[d]['bmi']['mean_pos']:.1f}", f"{sf[d]['bmi']['d']:.2f}", f"{sf[d]['bmi']['auc']:.3f}"])
    w(md_table(["Dataset", "Tuổi (không bệnh → bệnh)", "Cohen's d tuổi", "AUC đơn lẻ (tuổi)", "BMI (không bệnh → bệnh)", "Cohen's d BMI", "AUC đơn lẻ (BMI)"], rows))
    w("\n![effect](figures/fig_effect.png)\n")
    w("*BRFSS: tuổi được quy về trung điểm của nhóm tuổi.*\n")

    w("### 5.3 Chuyển giao chéo dataset\n")
    w("Huấn luyện Logistic Regression trên dataset A rồi kiểm tra trên tập test của dataset B, chỉ dùng các đặc trưng có ở cả hai bên. "
      "Nếu AUC chéo thấp hơn nhiều so với đường chéo chính, các dataset mô tả những quần thể/nhãn khác nhau.\n")
    w("![transfer](figures/fig_transfer.png)\n")
    mt = C["transfer_age_bmi"]
    diag = np.mean([mt[a][a] for a in mt])
    off = np.mean([mt[a][b] for a in mt for b in mt if a != b])
    w(f"Với tuổi + BMI: AUC trung bình cùng dataset **{diag:.3f}**, khác dataset **{off:.3f}**.")
    if "transfer_age_bmi_glucose" in C:
        mg = C["transfer_age_bmi_glucose"]
        w(f" Với thêm đường huyết (chỉ Pima ↔ 100k): Pima→100k = {mg['pima']['kaggle100k']:.3f}, 100k→Pima = {mg['kaggle100k']['pima']:.3f} "
          "(lưu ý: đường huyết ở Pima đo sau nghiệm pháp dung nạp glucose, ở 100k là đường huyết thông thường – cùng đơn vị mg/dL nhưng khác ý nghĩa).")
    w("\n")

    w("### 5.4 Yếu tố nào quyết định dự đoán ở từng dataset\n")
    w("![importance](figures/fig_importance.png)\n")
    for d in dss:
        top = C["top_features"][d][:3]
        w(f"- **{SHORT[d]}**: " + ", ".join(f"`{k}` (−{v['mean']:.3f})" for k, v in top))
    w("")
    w("### 5.5 Kiểm tra loại bỏ đặc trưng (do Agent 2 yêu cầu)\n")
    if has_abl:
        ab = C["ablation"]
        w(md_table(["Dataset", "Đặc trưng bị bỏ", "AUC đầy đủ → sau khi bỏ", "PR-AUC đầy đủ → sau khi bỏ"],
                   [[SHORT[d], ", ".join(f"`{x}`" for x in ab[d]["dropped"]), f"{ab[d]['auc_full']:.3f} → {ab[d]['auc_abl']:.3f}",
                     f"{ab[d]['pr_full']:.3f} → {ab[d]['pr_abl']:.3f}"] for d in dss if d in ab]))
        w("\n![ablation](figures/fig_ablation.png)\n")
    w("### 5.6 BRFSS tự nhiên vs BRFSS cân bằng 50/50\n")
    if "brfss" in summ and "brfss5050" in summ:
        a, b = summ["brfss"], summ["brfss5050"]
        w(md_table(["", "BRFSS (tự nhiên)", "BRFSS 50/50"], [
            ["Tỉ lệ bệnh", pct(a["prevalence"]), pct(b["prevalence"])],
            ["ROC-AUC", f"{a['auc']:.3f}", f"{b['auc']:.3f}"],
            ["Recall / Precision (mô hình cuối)", f"{a['recall']:.2f} / {a['precision']:.2f}", f"{b['recall']:.2f} / {b['precision']:.2f}"],
            ["Recall / Precision ở ngưỡng 0.5", f"{a['recall_at_05']:.2f} / {a['precision_at_05']:.2f}", f"{b['recall_at_05']:.2f} / {b['precision_at_05']:.2f}"],
            ["Accuracy", f"{a['accuracy']:.2f}", f"{b['accuracy']:.2f}"],
        ]))
        w("")

    # ------------------------------------------------------------ 6
    w("## 6. Nhận xét và kết luận\n")
    w("**Điểm chính (rút ra từ số liệu ở trên):**\n")
    w("1. Xếp hạng ROC-AUC: " + " > ".join(f"**{SHORT[d]}** ({summ[d]['auc']:.3f})" for d in ranking) + ".")
    if "kaggle100k" in C["ablation"]:
        ab = C["ablation"]["kaggle100k"]
        w(f"2. **100k** đạt AUC cao nhất vì có xét nghiệm trực tiếp: khi Agent 2 phát hiện mô hình phụ thuộc vào {', '.join(ab['dropped'])} và loại chúng ra, "
          f"AUC giảm từ {ab['auc_full']:.3f} xuống {ab['auc_abl']:.3f}. PR-AUC còn tụt mạnh hơn ({ab['pr_full']:.3f} → {ab['pr_abl']:.3f}). Nghĩa là phần lớn chất lượng dự đoán đến từ hai chỉ số xét nghiệm, chứ không phải từ các yếu tố nguy cơ (tuổi, BMI, bệnh nền) – tức là mô hình \"chẩn đoán\" nhiều hơn là \"dự báo sớm\".")
    if "pima" in C["ablation"]:
        ab = C["ablation"]["pima"]
        w(f"3. **Pima** cũng phụ thuộc vào {', '.join(ab['dropped'])}: bỏ đi thì AUC từ {ab['auc_full']:.3f} còn {ab['auc_abl']:.3f}. "
          f"Tập test chỉ có {fin['pima']['n_test']} mẫu nên khoảng tin cậy AUC rộng ({summ['pima']['auc_lo']:.2f}–{summ['pima']['auc_hi']:.2f}).")
    if "brfss" in summ:
        s = summ["brfss"]
        w(f"4. **BRFSS** không có xét nghiệm nên AUC chỉ {s['auc']:.3f}; vì bệnh hiếm ({pct(s['prevalence'])}) nên ở độ nhạy {s['recall']:.2f} thì precision chỉ {s['precision']:.2f} "
          f"(ngẫu nhiên là {pct(s['prevalence'])}). Ở ngưỡng 0.5 mô hình chỉ bắt được {pct(s['recall_at_05'], 0)} người bệnh – đây chính là lỗi mà Agent 2 phát hiện và sửa bằng cách hạ ngưỡng. "
          "Đổi lại, đây là mô hình duy nhất có thể dùng khi *không có xét nghiệm nào*, vd. sàng lọc qua bảng hỏi.")
    if "brfss5050" in summ and "brfss" in summ:
        w(f"5. Cân bằng lớp (50/50) gần như không đổi ROC-AUC ({summ['brfss']['auc']:.3f} vs {summ['brfss5050']['auc']:.3f}) nhưng làm accuracy/precision \"đẹp\" lên chỉ vì tỉ lệ bệnh nhân tạo – "
          "không được dùng các con số này để ước lượng hiệu quả ngoài đời thật.")
    gap = off - diag
    if abs(gap) < 0.03:
        w(f"6. Chuyển giao chéo chỉ với tuổi + BMI: AUC cùng dataset {diag:.3f} và khác dataset {off:.3f} gần như bằng nhau – mối liên hệ tuổi/BMI → bệnh khá ổn định giữa các nguồn, "
          "và AUC phụ thuộc vào dataset *được kiểm tra* nhiều hơn dataset huấn luyện (vd. 100k dễ hơn vì chênh lệch tuổi giữa hai nhóm rất lớn, Cohen's d = "
          f"{C['shared_features']['kaggle100k']['age']['d']:.2f}). Sự khác biệt thật sự giữa các dataset nằm ở các đặc trưng *riêng* (xét nghiệm vs tự khai). "
          "AUC chỉ đo thứ hạng nên không phát hiện được việc xác suất tuyệt đối bị lệch khi đổi quần thể (tỉ lệ bệnh 8.6% / 15.3% / 34.9%).")
    else:
        w(f"6. Chuyển giao chéo chỉ với tuổi + BMI: AUC cùng dataset {diag:.3f} so với khác dataset {off:.3f} – các dataset khác nhau về quần thể/nhãn nên mô hình không dùng chung được.")

    w("\n**Hạn chế cần nêu trong tiểu luận:**\n")
    w("- Đây là mô hình nghiên cứu, **không phải công cụ chẩn đoán**. Dữ liệu BRFSS là tự khai; Pima chỉ gồm nữ một nhóm dân tộc; 100k có dấu hiệu dữ liệu tổng hợp (các nhóm chủng tộc đều đúng 20%, ~80% bản ghi cùng một năm).")
    w("- Nhãn ở 3 dataset **không cùng nghĩa** (Pima: chẩn đoán trong 5 năm; BRFSS: bác sĩ từng chẩn đoán, tiền đái tháo đường tính là 0), và tập đặc trưng khác nhau, nên AUC giữa các dataset chỉ mang tính tham khảo.")
    w("- Bản ghi trùng lặp bị loại trước khi chia train/test để tránh rò rỉ; điều này làm tỉ lệ bệnh của BRFSS thay đổi nhẹ so với bản gốc. Với dữ liệu khảo sát toàn là biến rời rạc, một số trùng lặp là ngẫu nhiên hợp lệ.")
    w("- Các agent ở đây là agent dựa trên luật + học máy (không dùng LLM). Hướng mở rộng: thêm một agent LLM diễn giải kết quả cho bác sĩ, agent hiệu chuẩn xác suất, agent công bằng (fairness) theo giới/chủng tộc.")

    w("\n## 7. Cách chạy lại\n")
    w("```bash\npip install -r requirements.txt\npython main.py            # chạy đầy đủ (vài phút - chục phút)\npython main.py --fast      # chạy thử nhanh trên mẫu nhỏ\npython make_report.py      # dựng lại báo cáo/biểu đồ từ outputs/results.json\npython predict.py --input examples/patient_example.json\n```\n")

    (out_dir / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    # bảng số liệu dạng CSV
    import csv
    with open(out_dir / "metrics_summary.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(C["summary"][0].keys()))
        wr.writeheader()
        wr.writerows(C["summary"])
    return out_dir / "REPORT.md"
