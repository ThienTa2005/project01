# Hybrid Multiagent System hỗ trợ quyết định về đái tháo đường

Dự án cho môn Hệ đa agent gồm hai workflow: (1) các agent huấn luyện, tự đánh giá và so sánh mô hình trên ba bộ dữ liệu; (2) MAS runtime hỗ trợ quyết định gồm Patient Data, Deep Learning, Rule-Based, KG/RAG, Coordinator, Explanation và Monitoring Agent. RL chỉ hoạt động trong mô phỏng tổng hợp. Tất cả kết quả chỉ phục vụ giáo dục/nghiên cứu.

| Dataset | File | Ghi chú |
|---|---|---|
| 1. Pima Indians Diabetes | `data/diabetes.csv` | 768 dòng, đo lâm sàng |
| 2. Diabetes Prediction 100k | `data/diabetes_dataset.csv` | 100.000 dòng, có HbA1c + đường huyết |
| 3. CDC BRFSS 2015 | `data/archive.zip` (3 file: `binary`, `5050split`, `012`) | khảo sát tự khai; `binary` là bản chính, `5050split` chạy như biến thể, `012` chỉ để mô tả |

## Chạy

```bash
pip install -r requirements.txt
python main.py                 # chạy đầy đủ (~2-3 phút trên máy 2 nhân)
python main.py --fast          # thử nhanh trên mẫu nhỏ
python make_report.py          # dựng lại báo cáo/biểu đồ từ outputs/results.json
python predict.py --input examples/patient_example.json   # dự đoán cho 1 người bằng "hội đồng" agent
python -m scripts.train_dl_model  # huấn luyện lại artifact MLP có phiên bản
```

Kết quả nằm trong `outputs/`: **`REPORT.md`** (báo cáo tiếng Việt), `figures/`, `metrics_summary.csv`, `results.json`,
`agent_trace.jsonl` (toàn bộ message giữa các agent). Mô hình cuối của mỗi dataset lưu ở `models/`.

## Giao diện web

Web app Flask: nhập thông tin một người -> 3 agent (mỗi agent = 1 mô hình đã huấn luyện ở `models/`) cho xác suất, so với ngưỡng riêng và **bỏ phiếu**. Tab "Kết quả nghiên cứu" hiển thị bảng chỉ số và biểu đồ từ `outputs/`.

```bash
pip install -r requirements-web.txt
python app.py                  # mở http://localhost:8000
```

Web **không cần dữ liệu gốc** (`data/`) và không huấn luyện lại, chỉ đọc `models/*.joblib` (~0,4 MB) nên rất nhẹ.

| Đường dẫn | Mô tả |
|---|---|
| `GET /` | Giao diện |
| `GET /healthz` | Kiểm tra sống + mô hình đã nạp |
| `GET /api/schema` | Danh sách trường nhập của từng agent |
| `POST /api/predict` | Dự đoán, nhận JSON cùng định dạng `examples/patient_example.json` |
| `POST /api/decision-support` | Chạy MAS runtime: Data → DL → Rule → KG/RAG → Explanation → Monitoring |
| `GET /api/report` | Bảng chỉ số + danh sách biểu đồ |

```bash
curl -X POST localhost:8000/api/predict -H "Content-Type: application/json" \
  -d '{"common":{"age":52,"bmi":33.5,"sex":"female"},"kaggle100k":{"hbA1c_level":6.6}}'
```

Chạy toàn bộ workflow theo đặc tả kiến trúc:

```bash
curl -X POST localhost:8000/api/decision-support -H "Content-Type: application/json" \
  -d '{"patient":{"common":{"age":52,"bmi":33.5,"sex":"female"},"kaggle100k":{"hbA1c_level":6.6,"blood_glucose_level":155}},"question":"HbA1c bao nhiêu là đái tháo đường?"}'
```

Quy tắc: bắt buộc có `age`; agent nào có < 50% đặc trưng sẽ bị bỏ qua (cột chủng tộc của dataset 100k không tính). Muốn thêm/bớt ô nhập chỉ cần sửa `SCHEMA` trong `service.py`.

## Deploy

**Quan trọng:** file `models/*.joblib` phụ thuộc phiên bản scikit-learn. `requirements-web.txt` ghim chính xác `scikit-learn==1.9.0`, đúng phiên bản đã tạo các artifact hiện tại. Nếu huấn luyện lại bằng phiên bản khác, phải cập nhật đồng bộ artifact và dependency deploy.

Mọi cách dưới đây đều cần đẩy dự án lên GitHub trước (trừ Hugging Face có thể upload trực tiếp):

```bash
git init && git add . && git commit -m "diabetes mas web"
git branch -M main
git remote add origin https://github.com/<ten-ban>/diabetes-mas.git
git push -u origin main
```

> `data/` nặng ~12 MB (có `archive.zip`). Web không cần nên nếu muốn repo nhẹ, thêm `data/` vào `.gitignore` trước khi `git add`. Cách này vẫn chạy được web, chỉ không chạy lại được `main.py` từ repo.

### Cách 1 - Render (miễn phí, dễ nhất)
1. Vào https://render.com -> **New -> Blueprint** (hoặc **Web Service**) -> chọn repo GitHub.
2. Với Blueprint: Render đọc `render.yaml` và tự cấu hình. Với Web Service nhập tay: *Build* `pip install -r requirements-web.txt`, *Start* `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 60`, *Health check path* `/healthz`.
3. Đợi build xong, nhận link dạng `https://diabetes-mas.onrender.com`.

Gói free sẽ "ngủ" sau ~15 phút không có truy cập, lần mở đầu tiên mất khoảng 30-60 giây. Nên mở trước khi thuyết trình/demo.

### Cách 2 - Hugging Face Spaces (miễn phí, không ngủ lâu, hợp demo học thuật)
1. https://huggingface.co/new-space -> SDK chọn **Docker** (template Blank).
2. Upload toàn bộ dự án (hoặc `git push` sang repo Space). Thêm vào đầu `README.md` của Space khối cấu hình:
   ```
   ---
   title: Diabetes MAS
   sdk: docker
   app_port: 7860
   ---
   ```
3. `Dockerfile` đã sẵn sàng (cổng 7860). Space tự build, link dạng `https://huggingface.co/spaces/<user>/<ten-space>`.

### Cách 3 - Railway / Fly.io / Heroku
- **Railway:** New Project -> Deploy from GitHub; dùng `Procfile` hoặc `Dockerfile` có sẵn.
- **Fly.io:** `fly launch` (nhận `Dockerfile`), rồi `fly deploy`.
- Biến môi trường `PORT` do nền tảng cấp, không cần đặt.

### Cách 4 - Chạy bằng Docker trên máy / VPS
```bash
docker build -t diabetes-mas .
docker run -p 8000:7860 diabetes-mas      # http://localhost:8000
```
Trên VPS có thể đặt Nginx/Caddy phía trước để có HTTPS và tên miền.

### Khắc phục sự cố
- `InconsistentVersionWarning` hoặc lỗi khi nạp `.joblib`: sai phiên bản scikit-learn, xem mục "Quan trọng" ở trên.
- Build lỗi hết bộ nhớ trên gói free: giảm `--workers 2` xuống `1`.
- `/healthz` báo `false` cho một mô hình: thiếu file trong `models/` (chạy `python main.py` để tạo lại).

## Cấu trúc

```
main.py                 điểm vào: dựng agent, gửi message khởi động, chạy bus
config.py               đường dẫn, ngưỡng tự đánh giá, mô tả dataset
agents/
  base.py               Message, Blackboard, MessageBus, Agent
  runtime/              MAS hỗ trợ quyết định theo PDF (7 agent bắt buộc + RL mô phỏng tùy chọn)
  orchestrator.py       điều phối, vòng lặp huấn luyện -> tự đánh giá -> làm lại
  data_agent.py         nạp, làm sạch, bỏ trùng lặp, chia train/test
  model_agent.py        3 mô hình ứng viên + CV 5-fold + chiến lược (ngưỡng, class_weight, regularize, bỏ đặc trưng)
  evaluator_agent.py    AGENT 2 - tự đánh giá (9 tiêu chí, ra hành động khắc phục)
  comparator_agent.py   so sánh dataset: hiệu năng, đặc trưng chung, chuyển giao chéo, ablation
  reporter_agent.py / report_builder.py   biểu đồ + REPORT.md
predict.py              dự đoán một bệnh nhân bằng hội đồng 3 agent (dòng lệnh)
app.py                  web app Flask (giao diện + API)
service.py              nạp mô hình, schema form, hàm predict dùng cho web
scripts/train_dl_model.py  huấn luyện MLP, chọn ngưỡng trên validation, xuất artifact + metadata
models/diabetes_dl_v1.joblib / .metadata.json  mô hình DL và metadata có phiên bản
templates/index.html    giao diện web
requirements-web.txt    thư viện tối thiểu để chạy/deploy web (ghim phiên bản)
Dockerfile, Procfile, render.yaml   cấu hình deploy
```

## Điều chỉnh

- Ngưỡng tự đánh giá (AUC tối thiểu, recall mục tiêu, overfitting…) trong `Config.thr` ở `config.py`.
- Thêm dataset mới: khai báo trong `DATASETS` (file, cột nhãn, cột phân loại, cột bỏ).
- Lưu ý: đây là mô hình nghiên cứu, **không phải công cụ chẩn đoán y khoa**.

---

## BỔ SUNG 06/11/2026 – DEMO KIẾN THỨC GRAPH + RAG + MULTI-AGENT

**Tài liệu giải đáp bảng:** [`t.txt`](t.txt) đối chiếu yêu cầu ban đầu về dữ liệu, Neo4j, nguồn tham khảo và giới hạn OCR. Bản hiện tại giữ pipeline nghiên cứu cũ, đồng thời bổ sung Deep Learning và MAS runtime theo `docs/ARCHITECTURE_COMPLIANCE.md`.

### Mức độ triển khai (minh bạch)

| Thành phần | Thực tế chạy | Phạm vi/giới hạn |
|---|---|---|
| 3 mô hình dự đoán | `models/*.joblib` đã có từ bản gốc | Pima/100k/BRFSS, vote majority; không dùng để chẩn đoán |
| Knowledge Base | **7 nguồn, 46 thực thể, 46 quan hệ chuyên môn**, có 46 liên kết tài liệu `SUPPORTS` | Các mệnh đề biên tập có mã nguồn và URL, không phải full-text đã fine-tune |
| Knowledge Graph | Mặc định chạy bằng NetworkX, xuất GraphML / Cypher | Neo4j database là **tùy chọn**, chưa chạy nếu không đặt credentials. Khi có credentials và đã sync, GraphQueryAgent truy vấn cạnh thực từ Neo4j; khi lỗi, trả nhãn fallback về NetworkX |
| Vector RAG | Kết hợp TF-IDF n-gram từ + ký tự, mở rộng đồng nghĩa/ý định | Lexical retrieval cải tiến, không phải semantic embedding |
| GraphRAG | Tìm thực thể, duyệt cạnh 1–3 chặng | Là retrieval dựa đồ thị, không phải LLM tự viết Cypher |
| Hybrid | `α × vector_score + (1−α) × graph_score` | Chỉ là phương pháp kết hợp điểm minh họa, chưa benchmark trên ground-truth chuyên gia |
| Chatbot | Trả lời trích rút + nguồn + đồ thị con + trace | **Không** phải LLM thật; không được đoán chứng cứ |
| Web/PDF | Web HTTPS WHO/NIDDK/CDC/Vinmec hoặc PDF text, trích entity mentions | Nội dung mới trong web giữ tạm theo phiên; CLI có thể lưu bền vững. Không OCR PDF ảnh |
| Image/OCR | Chưa triển khai | Không nhận ảnh hoặc PDF scan là nguồn text; không được trình bày là đã có OCR |
| Deep Learning | **Đã triển khai** `diabetes-mlp-v1.0` | MLP 2 tầng ẩn 64/32, artifact và metadata có phiên bản; AUC test 0,976 trên 19.998 bản ghi |
| Orchestration | **Coordinator runtime hoàn chỉnh** | Patient Data → DL → Rule-Based → KG/RAG → Explanation → Monitoring; RL mô phỏng là tùy chọn |
| Hợp đồng agent | **Có cấu trúc và phiên bản** | `request_id`, `agent`, `status`, `payload`, `evidence`, `version`, `error`, thời gian thực thi |
| Hỏi đáp linh hoạt | 12 nhóm ý định, câu hỏi nối tiếp, trả lời theo cấu trúc và cảnh báo câu hỏi chẩn đoán cá nhân | Không tạo kiến thức ngoài các bằng chứng đã truy xuất |

### Chạy bản web mới

```bash
# Python 3.12, nên dùng env riêng
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements-web.txt
python -m scripts.build_kb
python app.py
```

Mở `http://localhost:8000`, giao diện có 4 mục: **Dự đoán (3 agent)**, **KB + RAG Chatbot**, **Đồ thị Knowledge Graph**, **Kết quả nghiên cứu**.

### Demo có thể thực hiện offline (không cần Neo4j/LLM/API)

1. Tab Dự đoán: bấm **Điền ví dụ → Dự đoán**, xem 3 xác suất, ngưỡng và kết quả đa số.
2. Tab Đồ thị Knowledge Graph: chọn thực thể `Đái tháo đường`, xem quan hệ và tài liệu nguồn. Ấn **↓ File Neo4j Cypher**, **↓ File GraphML**.
3. Tìm đường nhiều chặng: bắt đầu `Đái tháo đường`, kết thúc `Thận`, nhấn **Tìm đường đi**. Đường mong đợi: `Đái tháo đường → Đường huyết tăng → Thận` (2 chặng).
4. Tab KB + RAG Chatbot: hỏi **HbA1c bao nhiêu là đái tháo đường?**, chọn **Hybrid**, bấm **Truy xuất và trả lời**. Mở dẫn chứng `[NIDDK-01]` và đồ thị con. Thử Vector/Graph để so sánh.
5. Tick **Chạy toàn bộ MAS hỗ trợ quyết định** để chạy Coordinator, Patient Data, Deep Learning, Rule-Based, KG/RAG, Explanation và Monitoring Agent trong cùng request.
6. Tab Đồ thị: nhập một URL web trong danh sách cho phép hoặc tải PDF text, xem entity mentions. Quay về KB Chatbot để hỏi kèm tài liệu tạm. Nếu website nguồn chặn truy cập, dùng 7 nguồn biên tập sẵn để demo.
7. Tab Kết quả nghiên cứu: metrics và biểu đồ gốc không thay đổi.

### Neo4j thật (tùy chọn)

Nếu có Neo4j Aura/DB riêng:

```bash
# Chỉ chạy trên terminal local hoặc CI có secret; KHÔNG đưa mật khẩu vào GitHub.
# Linux/macOS
export NEO4J_URI='neo4j+s://<host>.databases.neo4j.io'
export NEO4J_USER='neo4j'
export NEO4J_PASSWORD='<password>'
python -m scripts.sync_neo4j
```

Trên Windows PowerShell, dùng `$env:NEO4J_URI='...'`, `$env:NEO4J_USERNAME='neo4j'`, `$env:NEO4J_PASSWORD='...'`, `$env:NEO4J_DATABASE='neo4j'`. Dự án cũng tiếp tục chấp nhận tên cũ `NEO4J_USER`. Sau khi sync, xem Neo4j Browser:

```cypher
MATCH (a:Entity)-[r]->(b:Entity)
WHERE a.id = 'Đái tháo đường'
RETURN a,r,b LIMIT 30;
```

Kiểm tra provenance:

```cypher
MATCH (s:Source)-[r:SUPPORTS]->(e:Entity)
RETURN s,r,e LIMIT 20;
```

Có thể cấu hình `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` trong Render Environment để dashboard hiển thị trạng thái **Neo4j kết nối**; runtime web chỉ kiểm tra và thực hiện GraphRAG bằng truy vấn Neo4j READ; không tự sync dữ liệu. Nếu đọc Neo4j lỗi hoặc dữ liệu không tương ứng, chatbot ghi nhãn fallback NetworkX. Vector TF-IDF vẫn đánh chỉ mục nguồn/fact nội bộ. Với cấu hình mặc định, UI ghi rõ **đang dùng NetworkX**. File Cypher có thể dùng bằng `cypher-shell -f knowledge/kb_diabetes.cypher`.

### Thêm nguồn web/PDF vào KB

**Tạm thời trong web:** mở tab Đồ thị Knowledge Graph → Bổ sung web/PDF → tải nội dung → hỏi RAG, dữ liệu chỉ được giữ trong phiên trình duyệt và gửi kèm request. Nguồn mới chỉ sinh quan hệ `MENTIONS` (có nhắc đến), tránh khẳng định quan hệ bệnh học không được duyệt.

**Lưu bền vững trên repo:**

```bash
python -m scripts.ingest_source --url 'https://www.vinmec.com/vie/bai-viet/dau-hieu-som-bao-hieu-benh-tieu-duong-vi'
# Hoặc:
python -m scripts.ingest_source --pdf path/to/source.pdf
python -m scripts.build_kb
```

Sau khi kiểm tra nội dung, commit `knowledge/extra_sources.json` cùng file GraphML/Cypher mới và push. Nội dung sẽ được RAG đánh chỉ mục trên mỗi truy vấn. Không commit tài liệu sức khỏe cá nhân hoặc nội dung không có quyền phân phối. Nguồn web tải từ xa có thể bị từ chối do chống bot/mất kết nối; KB biên tập vẫn hoạt động offline.

### Nguồn, dữ liệu huấn luyện và RAG là ba khái niệm khác nhau

- **Nguồn tri thức:** `knowledge/seed.json` chứa trích yếu tự biên tập và URL WHO / NIDDK / CDC / Vinmec. `knowledge/extra_sources.json` chứa nguồn bổ sung (mặc định rỗng).
- **Huấn luyện ML:** dataset nằm trong `data/`, quy trình `main.py` + `agents/data_agent.py` + `agents/model_agent.py` + `agents/evaluator_agent.py`, kết quả model `.joblib` lưu ở `models/`. Việc hỏi chatbot không làm train lại.
- **Truy xuất RAG:** `knowledge/engine.py` lấy các đoạn và quan hệ phù hợp từ KB để trả lời, không cập nhật tham số mô hình ML.

### API mới

| Endpoint | Mục đích |
|---|---|
| `GET /api/kb` | Thống kê, nguồn và câu hỏi mẫu |
| `GET /api/kb/graph` | Nút/cạnh cho đồ thị |
| `GET /api/kb/path?start=...&end=...&hops=5` | Đường đi đa chặng |
| `GET /api/kb/neo4j` | Kiểm tra kết nối Neo4j (không tiết lộ secrets) |
| `GET /kb-diabetes/kb_diabetes.cypher` | Tải file Cypher |
| `GET /kb-diabetes/kb_diabetes.graphml` | Tải GraphML |
| `POST /api/kb/ingest` | JSON `{ "url": "https://..." }` hoặc `multipart` PDF `file` |
| `POST /api/chat` | `{ "question": "...", "method": "hybrid", "top_k": 5, "hops": 2, "alpha": 0.5, "include_patient": false }` |
| `POST /api/decision-support` | `{ "patient": {...}, "question": "...", "simulate_rl": false }`; trả báo cáo và kết quả từng agent |

### Tests và Deploy Render

```bash
pip install -r requirements-dev.txt
python -m pytest -q tests/test_knowledge.py
python -m compileall -q app.py service.py knowledge scripts
```

**Lưu ý deploy:** `requirements-web.txt` đã thêm NetworkX, requests, BeautifulSoup, pypdf, neo4j; `render.yaml` và `Dockerfile` dùng **1 worker** để giảm RAM. Nếu web cũ trên Render trỏ repo cũ, hãy **thay tất cả file bằng bản ZIP này** (không upload ZIP nguyên dạng), commit & push, rồi theo dõi Render Logs. Có sẵn dataset và biểu đồ, nhưng web deploy chỉ cần `models/`, `knowledge/`, `static/`, `templates/`, `outputs/figures`, `outputs/metrics_summary.csv`, các file Python và requirements.

**Khả năng kiểm chứng ngày 12/11/2026:** `pytest -q -p no:cacheprovider tests` đạt **32 test**, gồm core KB/RAG, Flask, hợp đồng agent, luồng MAS đầy đủ, DL inference, validation dừng sớm và RL chỉ-mô-phỏng. `python -m scripts.train_dl_model`, `python -m scripts.build_kb`, `python predict.py --input examples/patient_example.json` và `python make_report.py` chạy được trên Windows. Neo4j Docker local đã được xác nhận với 46 Entity và 92 relationship tổng cộng; chưa xác nhận Render deploy hoặc Neo4j Aura.

**Cảnh báo:** Hệ thống học thuật, không có giá trị chẩn đoán bệnh. Pima chỉ nghiên cứu trên nữ trong quần thể dữ liệu. Tính xác suất từ các quần thể khác nhau có thể lệch hiệu chuẩn; không coi tỷ lệ bỏ phiếu là phần trăm xác suất bệnh.
