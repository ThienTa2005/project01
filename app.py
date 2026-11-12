"""Web app: python app.py  (dev)  |  gunicorn app:app  (production)"""
from __future__ import annotations

import csv
import os

from flask import Flask, jsonify, render_template, request, send_from_directory

import service
from agents.runtime import DecisionSupportCoordinator
from config import ROOT
from knowledge.engine import all_data, graph_stats, find_path, graph, extract_mentions
from knowledge.gemini import gemini_status
from knowledge.assistant import ask as ask_kb
from knowledge.neo4j_store import neo4j_status
from knowledge.ingest import ingest_pdf, ingest_url

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024  # cho phép PDF nhỏ, không lưu trên máy chủ
OUT = ROOT / "outputs"
decision_coordinator = DecisionSupportCoordinator()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/healthz")
def healthz():
    return jsonify(status="ok",
                   models={d: service.load_model(d) is not None for d in service.AGENTS},
                   dl_model=(ROOT / "models" / "diabetes_dl_v1.joblib").exists(),
                   runtime_mas="decision-coordinator-v1.0",
                   llm=gemini_status(),
                   kb=graph_stats())


@app.get("/api/schema")
def schema():
    return jsonify(schema=service.SCHEMA, agents={d: service.AGENT_LABEL[d] for d in service.AGENTS})


@app.get("/api/llm/gemini")
def gemini_info():
    """Report configuration only; never performs a paid call or exposes secrets."""
    return jsonify(gemini_status())


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Body phải là JSON object."), 400
    try:
        return jsonify(service.predict(payload))
    except ValueError as e:
        return jsonify(error=str(e)), 400


@app.post("/api/decision-support")
def decision_support():
    """Complete PDF-specified MAS workflow for one patient request."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Body phải là JSON object."), 400
    result = decision_coordinator.process(payload)
    return jsonify(result), (400 if result["status"] == "validation_error" else 200)


@app.get("/api/report")
def report():
    rows, f = [], OUT / "metrics_summary.csv"
    if f.exists():
        with f.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append({k: (float(v) if k not in ("ds", "best") and v not in ("", None) else v) for k, v in r.items()})
    figs = sorted(p.name for p in (OUT / "figures").glob("*.png")) if (OUT / "figures").exists() else []
    return jsonify(metrics=rows, figures=figs)


@app.get("/figures/<path:name>")
def figures(name):
    return send_from_directory(OUT / "figures", name)



# ---------------------------------------------------------------------------
# Knowledge Base / RAG / Multi-tool orchestration
# ---------------------------------------------------------------------------
@app.get("/api/kb")
def kb_info():
    d = all_data()
    return jsonify(stats=graph_stats(), sources=d["sources"],
                   description=d["description"],
                   demo_queries=["Bệnh tiểu đường có triệu chứng gì?",
                                 "HbA1c bao nhiêu là đái tháo đường?",
                                 "Vì sao tiểu đường có thể ảnh hưởng thận?",
                                 "Có thể phòng ngừa đái tháo đường típ 2 như thế nào?"])


@app.get("/api/kb/graph")
def kb_graph():
    return jsonify(stats=graph_stats(),
                   nodes=[{"id": n, "kind": attrs.get("kind", "Entity"),
                           "label": attrs.get("name", n)} for n, attrs in graph().nodes(data=True)],
                   edges=[{"source": a, "target": b,
                           "relation": attrs.get("predicate", "RELATED_TO"),
                           "source_id": attrs.get("source_id", "")}
                          for a, b, attrs in graph().edges(data=True)])


@app.get("/api/kb/path")
def kb_path():
    return jsonify(find_path(request.args.get("start", "Đái tháo đường"),
                            request.args.get("end", "Bệnh thận do đái tháo đường"),
                            request.args.get("hops", 3)))


@app.get("/api/kb/neo4j")
def kb_neo4j():
    return jsonify(neo4j_status())


@app.get("/kb-diabetes/<name>")
def download_kb(name):
    if name not in {"kb_diabetes.cypher", "kb_diabetes.graphml"}:
        return jsonify(error="File không tồn tại."), 404
    return send_from_directory(ROOT / "knowledge", name, as_attachment=True)


@app.post("/api/kb/ingest")
def kb_ingest():
    try:
        if "file" in request.files:
            f = request.files["file"]
            if not f.filename or not f.filename.lower().endswith(".pdf"):
                raise ValueError("Chỉ nhận PDF.")
            return jsonify(ingest_pdf(f.read(), title=f.filename))
        payload = request.get_json(silent=True) or {}
        url = str(payload.get("url", ""))
        return jsonify(ingest_url(url))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400


@app.post("/api/chat")
def kb_chat():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Body phải là JSON."), 400
    try:
        if len(str(data.get("question", ""))) > 600:
            raise ValueError("Câu hỏi quá dài.")
        if len(str(data)) > 80000:
            raise ValueError("Giới hạn tài liệu phiên làm việc là 80 KB.")
        if data.get("include_patient"):
            decision = decision_coordinator.process({
                "patient": data.get("patient"),
                "question": data.get("question", ""),
                "method": data.get("method", "hybrid"),
                "top_k": data.get("top_k", 5),
                "hops": data.get("hops", 2),
                "alpha": data.get("alpha", 0.5),
                "extra_documents": data.get("extra_documents", []),
                "history": data.get("history", []),
            })
            if decision["status"] == "validation_error":
                details = decision["report"].get("validation", {}).get("errors", [])
                message = details[0]["message"] if details else "Dữ liệu bệnh nhân không hợp lệ."
                return jsonify(error=message, decision_support=decision), 400
            kg = decision["agent_results"]["KGRAGAgent"]["payload"]
            response = dict(kg)
            response["answer"] = decision["report"]["narrative"]
            response["llm"] = decision["report"].get("generation", {})
            response["answer_mode"] = (
                "gemini-grounded-mas"
                if response["llm"].get("used")
                else "extractive-offline (Gemini fallback)"
            )
            response["decision_support"] = decision
            response["ml_prediction"] = None
            response["multi_agent_note"] = "Luồng MAS runtime qua CoordinatorAgent và hợp đồng message có phiên bản."
            response["trace"] = [
                {"agent": item["agent"], "action": item["status"],
                 "detail": f"v{item['version']} · {item['duration_ms']} ms"}
                for item in decision["agents"]
            ]
            return jsonify(response)
        return jsonify(ask_kb(question=data.get("question", ""),
                              method=data.get("method", "hybrid"),
                              top_k=data.get("top_k", 5), hops=data.get("hops", 2),
                              alpha=data.get("alpha", 0.5),
                              extras=data.get("extra_documents", []),
                              history=data.get("history", [])))
    except (ValueError, TypeError, OverflowError) as exc:
        return jsonify(error=str(exc)), 400


@app.errorhandler(413)
def oversized(_):
    return jsonify(error="Dung lượng vượt 4 MB."), 413

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=os.environ.get("DEBUG") == "1")
