# Hybrid Multiagent Diabetes Decision Support - Implementation Map

Tài liệu này ánh xạ đặc tả kiến trúc sang mã chạy thực tế. Hệ thống là nguyên mẫu giáo dục/nghiên cứu, không phải thiết bị hoặc hệ thống chẩn đoán y khoa.

## Six layers

| Layer | Implementation |
|---|---|
| User interaction | `templates/index.html`, `static/kb.js` |
| API and request processing | Flask `app.py`, endpoint `/api/decision-support` |
| Multiagent coordination | `agents/runtime/coordinator.py` |
| Intelligent agents | Các module độc lập trong `agents/runtime/` |
| Knowledge and model | `models/diabetes_dl_v1.joblib`, `knowledge/`, Neo4j hoặc NetworkX fallback |
| Evaluation and audit | `MonitoringAgent`, `outputs/runtime_audit.jsonl`, dòng `MAS_AUDIT` trong service logs, pytest và model metadata |

## Agent responsibilities

| Agent | Input | Output | Failure behavior |
|---|---|---|---|
| PatientDataAgent | Raw patient JSON | Validated and normalized features | Trả `invalid`; Coordinator dừng workflow |
| DLPredictionAgent | Validated features | Score, threshold, flag, model metadata | `insufficient_data` hoặc lỗi model rõ ràng |
| RuleBasedAgent | Features + DL result | Auditable flags, conflict and review flags | Không che giấu mâu thuẫn |
| KGRAGAgent | Natural-language question | Evidence, sources, snippets, subgraph | Lỗi retrieval có cấu trúc |
| CoordinatorAgent | User request | Aggregated workflow result | Duy trì kết quả từng agent |
| ExplanationAgent | Prediction + rules + evidence | Structured, evidence-grounded report | Luôn nêu giới hạn và trạng thái thiếu |
| MonitoringAgent | Execution metadata | Privacy-minimized audit record | Audit write failure làm workflow `partial` |
| RLSimulationAgent | Synthetic state | Simulated educational/follow-up action | Không nhận hành động điều trị thật |

## Message contract

Mỗi kết quả agent chứa:

```json
{
  "request_id": "req_...",
  "agent": "DLPredictionAgent",
  "status": "ok",
  "payload": {},
  "evidence": [],
  "version": "diabetes-mlp-v1.0",
  "error": null,
  "started_at": "...Z",
  "duration_ms": 12.3
}
```

## Runtime sequence

1. Coordinator tạo hoặc xác thực `request_id`.
2. PatientDataAgent kiểm tra kiểu, miền giá trị, trường bắt buộc và tính nhất quán.
3. Nếu hợp lệ, DLPredictionAgent chạy artifact MLP có phiên bản.
4. RuleBasedAgent áp dụng luật minh bạch và phát hiện mâu thuẫn.
5. KGRAGAgent truy xuất Vector/Graph/Hybrid với nguồn.
6. ExplanationAgent tạo báo cáo gồm prediction, rule flags, evidence và limitations.
7. MonitoringAgent ghi trạng thái, phiên bản, thời gian và SHA-256 của output; không ghi dữ liệu bệnh nhân.
8. RLSimulationAgent chỉ chạy khi `simulate_rl=true` và chỉ trên trạng thái tổng hợp.

## Safety invariants

- Không khẳng định chẩn đoán.
- Không kê thuốc, liều hoặc thay đổi điều trị.
- Luật chưa được xem là luật lâm sàng đã thẩm định.
- KG/RAG phải giữ source attribution.
- Mâu thuẫn giữa mô hình và luật bắt buộc được gắn cờ.
- Audit không lưu đặc trưng bệnh nhân hoặc nội dung câu hỏi; output được kiểm chứng bằng hash thay vì lưu nội dung nhạy cảm.
