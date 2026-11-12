# Diabetes MLP v1.0 - Model Card

## Intended use

Mô hình phục vụ nghiên cứu và giảng dạy trong hệ đa agent hỗ trợ quyết định về đái tháo đường. Mô hình không được thẩm định lâm sàng, không xác nhận chẩn đoán, không kê thuốc và không thay thế nhân viên y tế.

## Architecture and version

- Model version: `diabetes-mlp-v1.0`
- Artifact: `diabetes_dl_v1.joblib`
- Network: feed-forward MLP, Dense(64, ReLU) → Dense(32, ReLU) → xác suất lớp 1
- Training code: `python -m scripts.train_dl_model`
- Random seed: 42
- Training date: 2026-11-12

## Training data

Nguồn huấn luyện là `data/diabetes_dataset.csv`. Dữ liệu trùng được loại bỏ trước khi tách tập. Các cột `year`, `location` và toàn bộ cột `race:*` không được đưa vào mô hình. Tập test được giữ riêng đến bước đánh giá cuối.

Phân chia thực tế:

- Train: 63.990 bản ghi
- Validation chọn ngưỡng: 15.998 bản ghi
- Test cuối: 19.998 bản ghi

## Feature order

1. `gender`
2. `age`
3. `hypertension`
4. `heart_disease`
5. `smoking_history`
6. `bmi`
7. `hbA1c_level`
8. `blood_glucose_level`

Numeric được điền thiếu bằng median và chuẩn hóa StandardScaler. Categorical được điền mode và one-hot encoding; giá trị chưa thấy khi huấn luyện được bỏ qua an toàn.

## Output and operating threshold

`score` là xác suất lớp 1 theo mô hình, không phải xác suất nguy cơ lâm sàng đã hiệu chuẩn cho một cá nhân. `prediction = 1` khi `score >= 0,163612`. Ngưỡng này tối ưu F2 trên validation, không chọn trên test.

## Held-out test metrics

| Metric | Value |
|---|---:|
| ROC-AUC | 0,9756 |
| PR-AUC | 0,8749 |
| Recall | 0,8141 |
| Specificity | 0,9611 |
| Precision | 0,6603 |
| F2 | 0,7779 |
| Brier score | 0,0234 |
| ECE | 0,0037 |

## Limitations

- Chất lượng test cao không chứng minh hiệu lực lâm sàng hoặc khả năng chuyển giao sang quần thể khác.
- Dataset có dấu hiệu tổng hợp/ghép và nhãn phụ thuộc mạnh vào HbA1c, đường huyết.
- Không sử dụng mô hình khi thiếu cả HbA1c và đường huyết hoặc có dưới 4/8 trường.
- Kết quả mâu thuẫn giữa mô hình và luật minh bạch phải được gắn cờ cho con người xem xét.
