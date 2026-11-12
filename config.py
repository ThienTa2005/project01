"""Cấu hình trung tâm của hệ đa agent dự đoán bệnh tiểu đường."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    data_dir: Path = ROOT / "data"
    out_dir: Path = ROOT / "outputs"
    model_dir: Path = ROOT / "models"
    seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    max_rounds: int = 3          # số vòng "huấn luyện -> tự đánh giá" tối đa cho mỗi dataset
    fast: bool = False           # chế độ chạy nhanh (lấy mẫu dataset lớn) để thử nghiệm
    fast_n: int = 30000
    n_boot: int = 300            # số lần bootstrap để tính khoảng tin cậy AUC
    # Ngưỡng mà Agent 2 (EvaluatorAgent) dùng để tự đánh giá
    thr: dict = field(default_factory=lambda: dict(
        auc_min=0.75,         # khả năng phân biệt tối thiểu (ROC-AUC trên tập test)
        recall_target=0.70,   # độ nhạy tối thiểu - sàng lọc bệnh ưu tiên không bỏ sót
        specificity_min=0.50, # tránh "báo động" mọi người
        overfit_gap=0.05,     # chênh lệch AUC train - CV tối đa
        cv_std_max=0.03,      # độ dao động AUC giữa các fold
        test_cv_gap=0.03,     # lệch giữa AUC test và AUC CV
        ece_max=0.05,         # sai số hiệu chuẩn xác suất (ECE)
        dependence_drop=0.05, # AUC giảm > ngưỡng khi hoán vị 1 đặc trưng -> kiểm tra loại bỏ
        regression_drop=0.02, # vòng sau tệ hơn vòng trước quá ngưỡng này -> cảnh báo
    ))

    def ensure_dirs(self):
        for d in (self.out_dir, self.model_dir, self.out_dir / "figures"):
            d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Mô tả 3 bộ dữ liệu (dataset 3 = archive.zip chứa 3 file BRFSS2015)
# --------------------------------------------------------------------------- #
DATASETS = {
    "pima": dict(
        label="Pima Indians Diabetes",
        short="Pima",
        file="diabetes.csv",
        target="Outcome",
        zero_as_nan=["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
        drop=[],
        categorical=[],
        role="main",
        population="768 phụ nữ ≥ 21 tuổi thuộc bộ tộc Pima (Mỹ) – dữ liệu lâm sàng của NIDDK",
        label_def="Outcome = 1 là nhãn đái tháo đường của bộ dữ liệu; không suy ra thời điểm chẩn đoán từ CSV",
        collection="Đo lâm sàng (đường huyết, huyết áp, insulin, độ dày da, BMI)",
    ),
    "kaggle100k": dict(
        label="Diabetes Prediction 100k",
        short="100k",
        file="diabetes_dataset.csv",
        target="diabetes",
        zero_as_nan=[],
        drop=["year", "location"],          # cột hành chính, không phải đặc trưng lâm sàng
        categorical=["gender", "smoking_history"],
        role="main",
        population="100.000 bản ghi, nhiều bang/năm/chủng tộc (phân bố race đều 20% -> dấu hiệu dữ liệu tổng hợp/ghép)",
        label_def="diabetes = 1/0 (gắn với HbA1c và đường huyết)",
        collection="Có xét nghiệm HbA1c + đường huyết, tiền sử tăng huyết áp/tim mạch, hút thuốc",
    ),
    "brfss": dict(
        label="CDC BRFSS 2015 (binary)",
        short="BRFSS",
        file="diabetes_binary_health_indicators_BRFSS2015.csv",
        target="Diabetes_binary",
        zero_as_nan=[],
        drop=[],
        categorical=[],
        role="main",
        population="253.680 người trưởng thành Mỹ trả lời khảo sát sức khỏe CDC 2015",
        label_def="Diabetes_binary = 1 nếu tự khai đã được bác sĩ chẩn đoán đái tháo đường (tiền đái tháo đường -> 0)",
        collection="Khảo sát điện thoại, tự khai báo (lối sống, bệnh nền, kinh tế - xã hội), KHÔNG có xét nghiệm",
    ),
    "brfss5050": dict(
        label="CDC BRFSS 2015 (50/50)",
        short="BRFSS-50/50",
        file="diabetes_binary_5050split_health_indicators_BRFSS2015.csv",
        target="Diabetes_binary",
        zero_as_nan=[],
        drop=[],
        categorical=[],
        role="variant",
        population="70.692 bản ghi – tập con của BRFSS đã cân bằng 50% mắc / 50% không mắc",
        label_def="Giống BRFSS binary nhưng cân bằng lớp (50/50)",
        collection="Giống BRFSS binary",
    ),
}

# File chỉ để mô tả (không huấn luyện): nhãn 3 lớp
PROFILE_ONLY = {
    "brfss012": dict(
        label="CDC BRFSS 2015 (3 lớp 0/1/2)",
        file="diabetes_012_health_indicators_BRFSS2015.csv",
        target="Diabetes_012",
    ),
}

# Các cột dùng để hài hòa hóa (harmonize) giữa các dataset khi so sánh
AGE_COL = {"pima": "Age", "kaggle100k": "age", "brfss": "Age", "brfss5050": "Age"}
BMI_COL = {"pima": "BMI", "kaggle100k": "bmi", "brfss": "BMI", "brfss5050": "BMI"}
GLU_COL = {"pima": "Glucose", "kaggle100k": "blood_glucose_level"}
# BRFSS mã hóa tuổi thành 13 nhóm (1 = 18-24 ... 13 = ≥ 80) -> quy về tuổi trung bình nhóm
BRFSS_AGE_MID = {1: 21, 2: 27, 3: 32, 4: 37, 5: 42, 6: 47, 7: 52, 8: 57,
                 9: 62, 10: 67, 11: 72, 12: 77, 13: 82}
