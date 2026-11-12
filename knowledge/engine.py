"""Triển khai KB + Vector RAG + GraphRAG + Hybrid không phụ thuộc LLM.

Các mệnh đề trong seed.json được biên tập từ nguồn có URL. Tài liệu tự tải lên
chỉ được nối với các THỰC THỂ ĐƯỢC NHẮC ĐẾN (MENTIONS), tuyệt đối không tự
suy diễn quan hệ y khoa từ đoạn văn không qua kiểm chứng.
"""
from __future__ import annotations

import io
import math
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
SEED = ROOT / "seed.json"
MAX_EXTRA_DOCUMENTS = 3
MAX_EXTRA_TEXT = 14000


def normalize(text: str) -> str:
    """Gộp chữ thường, bỏ dấu tiếng Việt để tìm kiếm ít phụ thuộc dấu."""
    text = unicodedata.normalize("NFD", str(text).lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


@lru_cache(maxsize=1)
def seed_data():
    return json.loads(SEED.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def all_data():
    """Nguồn biên tập + nguồn web/PDF đã nhập qua CLI (phân biệt độ tin cậy)."""
    base = seed_data()
    path = ROOT / "extra_sources.json"
    extras = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    if not isinstance(extras, list):
        raise ValueError("extra_sources.json phải là một danh sách JSON.")
    return {**base, "sources": base["sources"] + extras}


@lru_cache(maxsize=1)
def graph():
    """Tạo đồ thị NetworkX từ các quan hệ đã biên tập; không cần Neo4j."""
    G = nx.MultiDiGraph()
    data = all_data()
    for fact in data["facts"]:
        s, o = fact["subject"], fact["object"]
        G.add_node(s, name=s, kind="Entity")
        G.add_node(o, name=o, kind="Entity")
        G.add_edge(s, o, key=fact["id"], predicate=fact["predicate"],
                   source_id=fact["source_id"], fact_id=fact["id"],
                   evidence=fact["evidence"])
    for src in data["sources"]:
        node = f"SOURCE:{src['id']}"
        G.add_node(node, name=src["title"], kind="Source", url=src["url"])
    # Mỗi cạnh SUPPORTS liên kết tài liệu nguồn với chủ thể của fact.
    # Quan hệ y khoa vẫn nằm trên cạnh Entity -> Entity, có source_id.
    for fact in data["facts"]:
        G.add_edge(f"SOURCE:{fact['source_id']}", fact["subject"],
                   key=f"SUPPORTS:{fact['id']}", predicate="SUPPORTS",
                   fact_id=fact["id"], source_id=fact["source_id"],
                   evidence=fact["evidence"])
    for src in data["sources"]:
        if src.get("kind") == "curated" or not src.get("body"):
            continue
        for i, entity in enumerate(extract_mentions(src["body"])):
            G.add_edge(f"SOURCE:{src['id']}", entity,
                       key=f"MENTIONS:{src['id']}:{i}", predicate="MENTIONS",
                       fact_id=f"MENTIONS:{i}", source_id=src["id"],
                       evidence="Chỉ phát hiện nhắc đến thực thể, chưa xác thực quan hệ.")
    return G


def graph_stats():
    g = graph()
    return {"entities": sum(d.get("kind") == "Entity" for _, d in g.nodes(data=True)),
            "sources": len(all_data()["sources"]),
            "relations": len(seed_data()["facts"]),
            "graph_nodes": g.number_of_nodes(),
            "storage": "NetworkX (mặc định) / Neo4j tùy chọn"}


def _keywords(text: str):
    words = normalize(text).split()
    # các từ không giúp phân biệt nội dung
    stop = {"la", "gi", "va", "cua", "o", "the", "nao", "cho", "hoi", "ve", "co", "nhung", "cac", "tu", "den", "hay", "toi", "duoc", "voi", "tai", "sao", "nhieu", "benh", "dai", "thao", "duong", "tieu"}
    return {w for w in words if len(w) > 2 and w not in stop}


# Đồng nghĩa thường gặp khi truy vấn; không biến thành khẳng định y khoa.
ALIASES = {"tiểu đường": "Đái tháo đường", "diabetes": "Đái tháo đường",
           "tiểu đường tuýp 2": "Đái tháo đường típ 2", "tiểu đường type 2": "Đái tháo đường típ 2",
           "đường máu": "Đường huyết tăng", "đường trong máu": "Đường huyết tăng",
           "đường huyết": "Đường huyết tăng", "blood sugar": "Đường huyết tăng",
           "bệnh thận": "Bệnh thận do đái tháo đường", "mắt": "Võng mạc",
           "hba1c": "HbA1c", "a1c": "HbA1c", "hoóc môn insulin": "Insulin",
           "type 1": "Đái tháo đường típ 1", "típ 1": "Đái tháo đường típ 1", "tuýp 1": "Đái tháo đường típ 1",
           "type 2": "Đái tháo đường típ 2", "típ 2": "Đái tháo đường típ 2", "tuýp 2": "Đái tháo đường típ 2"}

# Bộ hiểu ý định nhỏ, có thể kiểm thử và không phát sinh kiến thức mới. Tất cả
# chuỗi ở đây đã bỏ dấu vì được so với normalize(question).
INTENT_RULES = [
    ("personal_assessment", ("toi bi", "toi co", "cua toi", "co phai toi", "toi dang")),
    ("diagnosis", ("co phai", "chan doan", "lam sao biet", "kiem tra", "xet nghiem", "phat hien")),
    ("threshold", ("bao nhieu", "nguong", "chi so", "ket qua", "muc nao", "hba1c", "a1c", "mg dl")),
    ("symptoms", ("trieu chung", "dau hieu", "bieu hien", "nhan biet", "canh bao")),
    ("diet", ("an gi", "an uong", "kieng", "thuc pham", "dinh duong", "che do an")),
    ("prevention", ("phong ngua", "phong tranh", "ngan ngua", "giam nguy co", "tranh mac", "han che nguy co")),
    ("risk", ("nguy co", "nguyen nhan", "tai sao mac", "de mac", "ai de bi")),
    ("complications", ("bien chung", "anh huong", "ton thuong", "hau qua", "nguy hiem")),
    ("definition", ("la gi", "nghia la", "dinh nghia", "hoat dong the nao", "vai tro")),
    ("treatment", ("dieu tri", "thuoc", "metformin", "insulin", "kiem soat", "quan ly benh")),
    ("cure", ("chua khoi", "khoi han", "het benh", "tri dut diem")),
    ("types", ("may loai", "phan loai", "type 1", "type 2", "tip 1", "tip 2", "typ 1", "typ 2", "tuyp 1", "tuyp 2")),
]

INTENT_PREDICATES = {
    "diagnosis": {"ASSESSED_WITH", "HAS_DIABETES_THRESHOLD", "HAS_PREDIABETES_RANGE", "NOT_SUFFICIENT_FOR_DIAGNOSIS"},
    "threshold": {"HAS_DIABETES_THRESHOLD", "HAS_PREDIABETES_RANGE", "REFLECTS", "ASSESSED_WITH"},
    "symptoms": {"HAS_SYMPTOM"},
    "diet": {"PREVENTION", "CAN_BE_TREATED_WITH"},
    "prevention": {"PREVENTION"},
    "risk": {"HAS_RISK_FACTOR"},
    "complications": {"HAS_COMPLICATION", "CAN_DAMAGE"},
    "treatment": {"CAN_BE_TREATED_WITH", "REQUIRES", "PREVENTION"},
    "cure": {"CAN_BE_TREATED_WITH", "REQUIRES"},
    "types": {"HAS_TYPE"},
    "definition": {"IS_A", "PRODUCED_BY", "REGULATES", "RELATED_TO", "REFLECTS", "HAS_TYPE"},
}

INTENT_EXPANSIONS = {
    "diagnosis": "xét nghiệm chẩn đoán xác nhận đường huyết HbA1c",
    "threshold": "ngưỡng xét nghiệm HbA1c glucose đường huyết",
    "symptoms": "triệu chứng dấu hiệu biểu hiện",
    "diet": "dinh dưỡng chế độ ăn lành mạnh đường chất béo",
    "prevention": "phòng ngừa giảm nguy cơ vận động cân nặng",
    "risk": "yếu tố nguy cơ nguyên nhân",
    "complications": "biến chứng tổn thương thận mắt thần kinh tim",
    "treatment": "điều trị quản lý thuốc insulin metformin lối sống",
    "cure": "điều trị quản lý bệnh",
    "types": "phân loại típ 1 típ 2 thai kỳ",
    "definition": "định nghĩa vai trò là gì",
}

DOMAIN_TERMS = (
    "dai thao", "tieu duong", "diabetes", "glucose", "insulin", "metformin",
    "duong huyet", "duong mau", "hba1c", "a1c", "tien dai thao",
)
FOLLOWUP_MARKERS = ("con ", "the con", "vay ", "truong hop nay", "benh nay", "no ", "nhu the nao")
SMALLTALK_EXACT = {
    "hi", "hello", "hey", "alo", "chao", "xin chao", "chao ban",
    "cam on", "cam on ban", "thanks", "thank you",
    "ban la ai", "ban lam duoc gi", "giup toi voi", "help",
}
SMALLTALK_PHRASES = (
    "xin chao", "chao diabot", "cam on diabot", "ban co the lam gi",
    "chatbot nay lam gi", "huong dan toi",
)


def selected_entities(question: str):
    q = normalize(question)
    found = []
    for entity, attrs in graph().nodes(data=True):
        if attrs.get("kind") != "Entity":
            continue
        n = normalize(entity)
        if n in q and len(n) >= 4:
            found.append((len(n), entity))
    for alias, canonical in ALIASES.items():
        if normalize(alias) in q:
            found.append((len(normalize(alias)), canonical))
    found.sort(reverse=True)
    seen = set()
    return [entity for _, entity in found if not (entity in seen or seen.add(entity))]


def detect_intents(question: str):
    q = normalize(question)
    if (q in SMALLTALK_EXACT
            or (len(q.split()) <= 10 and any(phrase in q for phrase in SMALLTALK_PHRASES))):
        return ["smalltalk"]
    intents = [name for name, phrases in INTENT_RULES if any(p in q for p in phrases)]
    return intents or ["general"]


def _history_questions(history):
    """Chỉ nhận tối đa 6 câu người dùng ngắn; không tin dữ liệu lịch sử từ client."""
    if not isinstance(history, list):
        return []
    result = []
    for turn in history[-12:]:
        if not isinstance(turn, dict) or turn.get("role") != "user":
            continue
        text = str(turn.get("content") or "").strip()[:600]
        if text:
            result.append(text)
    return result[-6:]


def question_profile(question: str, history=None):
    """Hiểu miền, ý định và câu hỏi nối tiếp mà không gọi LLM."""
    q = normalize(question)
    intents = detect_intents(question)
    entities = selected_entities(question)
    explicit_domain = bool(entities) or any(term in q for term in DOMAIN_TERMS)
    content_intents = [x for x in intents if x not in {"general", "personal_assessment"}]
    previous = _history_questions(history)
    context_used = False
    effective = question.strip()

    # Câu ngắn như "Còn cách phòng ngừa?" được gắn với câu tiểu đường gần nhất.
    if not explicit_domain and previous and (content_intents or any(x in q for x in FOLLOWUP_MARKERS)):
        for old in reversed(previous):
            old_q = normalize(old)
            if selected_entities(old) or any(term in old_q for term in DOMAIN_TERMS):
                effective = old + " | Câu hỏi nối tiếp: " + question.strip()
                context_used = True
                break

    effective_entities = selected_entities(effective)
    in_scope = bool(effective_entities) or any(term in normalize(effective) for term in DOMAIN_TERMS)
    # Đây là chatbot chuyên biệt về diabetes: một ý định y khoa rõ ràng có thể
    # được hỏi ngắn mà không cần lặp lại "tiểu đường" ở mọi câu.
    implicit_domain_intents = {"symptoms", "diet", "prevention", "risk", "complications",
                               "treatment", "threshold", "cure", "types"}
    if (not in_scope and set(content_intents) & implicit_domain_intents
            and len(q.split()) <= 12):
        effective = effective + " | trong bệnh đái tháo đường"
        in_scope = True
        context_used = context_used or bool(previous)
        effective_entities = selected_entities(effective)

    expansion = " ".join(INTENT_EXPANSIONS[x] for x in content_intents if x in INTENT_EXPANSIONS)
    canonical = " ".join(effective_entities)
    expanded = " ".join(x for x in (effective, canonical, expansion) if x)
    return {
        "intents": intents,
        "primary_intent": content_intents[0] if content_intents else "general",
        "entities": effective_entities,
        "in_scope": in_scope,
        "context_used": context_used,
        "effective_question": effective,
        "expanded_query": expanded,
        "personal": "personal_assessment" in intents,
    }


def validate_extra_docs(extras):
    """Giới hạn dữ liệu qua API; không lưu tài liệu cá nhân trên server."""
    clean = []
    if not isinstance(extras, list):
        return clean
    for i, d in enumerate(extras[:MAX_EXTRA_DOCUMENTS]):
        if not isinstance(d, dict):
            continue
        body = str(d.get("text") or "")[:MAX_EXTRA_TEXT].strip()
        if not body:
            continue
        name = str(d.get("title") or f"Nguồn phiên làm việc {i+1}")[:160]
        url = str(d.get("url") or "")[:500]
        if url and (urlparse(url).scheme != "https" or not urlparse(url).netloc):
            url = ""
        clean.append({"id": f"SESSION-{i+1}", "title": name, "url": url,
                      "publisher": "Người dùng cung cấp (chưa thẩm định)", "kind": "session",
                      "summary": body[:400], "body": body})
    return clean


def extract_mentions(text: str):
    """Trích xuất entity mention theo ontology hiện có (không tự suy luận quan hệ bệnh học)."""
    n = normalize(text)
    mentions = []
    entities = {f[k] for f in seed_data()["facts"] for k in ("subject", "object")}
    for entity in sorted(entities):
        if len(normalize(entity)) > 3 and normalize(entity) in n:
            mentions.append(entity)
    for term, canonical in ALIASES.items():
        if normalize(term) in n and canonical not in mentions:
            mentions.append(canonical)
    return mentions[:24]


def _vector_candidates(question, facts, docs, limit, profile=None):
    items = [(f["evidence"] + " " + f["subject"] + " " + f["object"], "fact", f) for f in facts]
    items += [(d.get("body") or d["summary"], "document", d) for d in docs]
    if not items:
        return []
    corpus = [x[0] for x in items]
    # Trộn n-gram ký tự (chịu lỗi chính tả nhẹ) và n-gram từ (giữ nghĩa cụm từ).
    # Đây vẫn là lexical retrieval có mở rộng truy vấn, không giả danh embedding.
    char_tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1,
                              max_features=24000, preprocessor=normalize)
    word_tf = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1,
                              sublinear_tf=True, max_features=16000, preprocessor=normalize)
    char_mat = char_tf.fit_transform(corpus + [question])
    word_mat = word_tf.fit_transform(corpus + [question])
    scores = (0.58 * cosine_similarity(char_mat[-1:], char_mat[:-1]).ravel()
              + 0.42 * cosine_similarity(word_mat[-1:], word_mat[:-1]).ravel())
    if profile:
        preferred = set().union(*(INTENT_PREDICATES.get(x, set()) for x in profile["intents"]))
        for i, (_, kind, item) in enumerate(items):
            if kind == "fact" and item.get("predicate") in preferred:
                scores[i] += 0.16
            elif kind == "document" and item.get("id", "").startswith("SESSION-"):
                # Tài liệu người dùng vừa chọn phải có cơ hội xuất hiện trong
                # top-k Hybrid thay vì bị toàn bộ cạnh graph lấn át.
                scores[i] += 0.32
    order = sorted(range(len(items)), key=lambda i: float(scores[i]), reverse=True)
    return [{"kind": items[i][1], "item": items[i][2], "score": round(float(scores[i]), 5)}
            for i in order[:limit] if scores[i] >= 0.03]


def intent_bonus(question: str, fact: dict) -> float:
    """Rerank quan hệ theo ý định câu hỏi (không tạo sự thật mới)."""
    q = normalize(question)
    pred = fact["predicate"]
    bonus = 0.0
    rules = [
        (("trieu chung", "dau hieu", "bieu hien"), {"HAS_SYMPTOM"}, 0.5),
        (("bien chung", "anh huong", "ton thuong", "than", "than kinh", "mat"), {"HAS_COMPLICATION", "CAN_DAMAGE"}, 0.26),
        (("bao nhieu", "nguong", "ket qua xet nghiem", "chan doan"), {"HAS_DIABETES_THRESHOLD", "HAS_PREDIABETES_RANGE"}, 0.5),
        (("hba1c",), {"HAS_DIABETES_THRESHOLD", "HAS_PREDIABETES_RANGE", "REFLECTS", "ASSESSED_WITH"}, 0.26),
        (("phong ngua", "giam nguy co", "an gi", "tap the duc", "dinh duong"), {"PREVENTION"}, 0.55),
        (("nguy co", "tai sao mac", "nguyen nhan"), {"HAS_RISK_FACTOR"}, 0.24),
        (("thuoc", "dieu tri", "metformin", "insulin"), {"CAN_BE_TREATED_WITH", "REQUIRES"}, 0.35),
        (("la gi", "dinh nghia", "vai tro"), {"IS_A", "PRODUCED_BY", "REGULATES", "REFLECTS"}, 0.45),
        (("kiem tra", "co phai", "lam sao biet"), {"ASSESSED_WITH", "NOT_SUFFICIENT_FOR_DIAGNOSIS"}, 0.42),
    ]
    for keywords, predicates, weight in rules:
        if any(k in q for k in keywords) and pred in predicates:
            bonus += weight
    # Nếu hỏi đích cụ thể như "thận", ưu tiên cạnh chạm trực tiếp đích.
    if "than" in q and "than" in normalize(fact["object"]):
        bonus += 0.21
    if "hba1c" in q and ("hba1c" in normalize(fact["subject"]) or "hba1c" in normalize(fact["object"])):
        bonus += 0.2
    return bonus


def _graph_candidates(question, facts, hops, limit):
    ents = selected_entities(question)
    if not ents:
        return [], "NetworkX"
    # Neo4j được dùng trực tiếp khi đã cấu hình. Nếu lỗi/missing data,
    # fallback NetworkX phải ghi nhãn rõ, không tuyên bố đang dùng Neo4j.
    graph_backend = "NetworkX"
    try:
        from .neo4j_store import fetch_neighbor_facts
        neo_facts = fetch_neighbor_facts(ents, hops=hops, limit=150)
        if neo_facts:
            allowed = {f["id"] for f in facts}
            validated = [f for f in neo_facts if f["id"] in allowed]
            if validated:
                facts = validated
                graph_backend = "Neo4j"
    except Exception:
        graph_backend = "NetworkX (Neo4j lỗi/kết nối thất bại)"
    g = graph()
    # Lấy vùng lân cận 1..hops; điểm cộng nếu thực thể chính/quan hệ nhắc trong câu.
    g_simple = nx.Graph()
    for f in facts:
        g_simple.add_edge(f["subject"], f["object"])
    distances = {}
    for ent in ents[:4]:
        if ent in g_simple:
            for n, dist in nx.single_source_shortest_path_length(g_simple, ent, cutoff=hops).items():
                distances[n] = min(distances.get(n, 99), dist)
    tokens = _keywords(question)
    ranked = []
    for f in facts:
        if f["subject"] not in distances and f["object"] not in distances:
            continue
        d = min(distances.get(f["subject"], 99), distances.get(f["object"], 99))
        # Hạn chế các cạnh không liên quan khi từ khóa đích được nhắc trực tiếp.
        overlap = len(tokens & _keywords(" ".join([f["subject"], f["object"], f["evidence"]])))
        entity_hit = int(f["subject"] in ents or f["object"] in ents)
        score = 0.12 + 0.14 * entity_hit + 0.09 * overlap - 0.055 * d + intent_bonus(question, f)
        ranked.append({"kind": "fact", "item": f, "score": max(0.0, round(score, 5))})
    return sorted(ranked, key=lambda x: x["score"], reverse=True)[:limit], graph_backend


def _id(candidate):
    return (candidate["kind"], candidate["item"]["id"])


def _compose_answer(profile, snippets):
    """Tạo câu trả lời có cấu trúc từ đúng bằng chứng đã truy xuất."""
    if not snippets:
        return "Chưa tìm được bằng chứng phù hợp trong kho tri thức. Hãy hỏi cụ thể hơn hoặc bổ sung nguồn web/PDF."

    preferred = set().union(*(INTENT_PREDICATES.get(x, set()) for x in profile["intents"]))
    direct = [x for x in snippets if x.get("predicate") in preferred]
    ordered = direct + [x for x in snippets if x not in direct]
    chosen, seen = [], set()
    for item in ordered:
        key = normalize(item["text"])
        if key in seen:
            continue
        seen.add(key)
        chosen.append(item)
        if len(chosen) == 5:
            break

    intent = profile["primary_intent"]
    if profile["personal"] or intent == "diagnosis":
        lead = ("Không thể xác định một người có mắc đái tháo đường chỉ từ câu hỏi hoặc một triệu chứng. "
                "Chẩn đoán cần được nhân viên y tế đánh giá bằng xét nghiệm phù hợp. Bằng chứng liên quan trong KB:")
    elif intent == "cure":
        lead = ("Kho tri thức không có bằng chứng cho phép khẳng định một thuốc có thể chữa khỏi hoàn toàn đái tháo đường. "
                "Các nguồn hiện mô tả điều trị và quản lý bệnh như sau:")
    elif intent == "definition" and not any(x.get("predicate") in {"IS_A", "PRODUCED_BY", "REGULATES"} for x in chosen):
        lead = "KB chưa có định nghĩa trực tiếp đầy đủ; các quan hệ gần nhất có nguồn là:"
    else:
        leads = {
            "symptoms": "Các dấu hiệu được nguồn trong KB đề cập gồm:",
            "diet": "Về ăn uống và lối sống, bằng chứng hiện có cho biết:",
            "prevention": "Các biện pháp có bằng chứng về phòng ngừa hoặc trì hoãn típ 2 gồm:",
            "risk": "Các yếu tố nguy cơ được nguồn trong KB đề cập gồm:",
            "complications": "Các biến chứng hoặc cơ quan có thể bị ảnh hưởng gồm:",
            "treatment": "Thông tin điều trị và quản lý có trong KB:",
            "threshold": "Các ngưỡng và thông tin xét nghiệm liên quan:",
            "types": "Các thể bệnh được mô tả trong KB:",
            "definition": "Theo các nguồn trong KB:",
        }
        lead = leads.get(intent, "Dựa trên các bằng chứng phù hợp nhất trong kho tri thức:")
    bullets = [f"• {x['text']} [{x['source_id']}]" for x in chosen]
    return lead + "\n" + "\n".join(bullets)


def retrieve(question, method="hybrid", top_k=5, hops=2, alpha=0.5, extra_documents=None, history=None):
    """Hybrid = alpha*vector + (1-alpha)*graph; alpha trong [0,1]."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Bạn cần nhập câu hỏi.")
    if len(question) > 600:
        raise ValueError("Câu hỏi tối đa 600 ký tự.")
    if method not in {"vector", "graph", "hybrid"}:
        raise ValueError("Phương pháp phải là vector, graph hoặc hybrid.")
    top_k = max(1, min(int(top_k), 12))
    hops = max(1, min(int(hops), 3))
    alpha = float(alpha)
    if not math.isfinite(alpha):
        raise ValueError("Trọng số alpha phải là số hữu hạn.")
    alpha = max(0.0, min(alpha, 1.0))
    data = all_data()
    facts = data["facts"]
    validated_extras = validate_extra_docs(extra_documents)
    docs = data["sources"] + validated_extras
    profile = question_profile(question, history)
    if profile["primary_intent"] == "smalltalk":
        q = normalize(question)
        if "cam on" in q or "thank" in q:
            answer = "Rất vui vì đã hỗ trợ bạn. Bạn có thể hỏi tiếp về đái tháo đường, xét nghiệm, nguy cơ, phòng ngừa hoặc biến chứng."
        elif "ban la ai" in q or "lam gi" in q or "giup" in q or "help" in q:
            answer = ("Mình là DiaBot, trợ lý của hệ đa tác tử về đái tháo đường. "
                      "Mình có thể giải thích kiến thức có nguồn, truy xuất Knowledge Graph "
                      "và hỗ trợ diễn giải kết quả nghiên cứu; mình không thay thế bác sĩ.")
        else:
            answer = ("Chào bạn! Mình là DiaBot. Bạn muốn tìm hiểu điều gì về đái tháo đường? "
                      "Ví dụ: ngưỡng HbA1c, yếu tố nguy cơ, cách phòng ngừa hoặc biến chứng.")
        return {
            "question": question, "method": method, "top_k": top_k, "hops": hops, "alpha": alpha,
            "answer": answer, "answer_mode": "smalltalk-safe", "graph_backend": "Không truy vấn (hội thoại)",
            "snippets": [], "sources": [], "subgraph": {"nodes": [], "edges": []},
            "entity_matches": [], "intent": "smalltalk", "context_used": False,
            "effective_question": question.strip(),
            "trace": [{
                "agent": "ConversationRouter",
                "action": "Nhận diện hội thoại xã giao",
                "detail": "Không cần truy vấn dữ kiện y khoa",
            }],
        }
    # Không trả lời câu ngoài phạm vi bằng những gram ký tự trùng ngẫu nhiên.
    if not profile["in_scope"] and not validated_extras:
        return {"question": question, "method": method, "top_k": top_k, "hops": hops, "alpha": alpha,
                "answer": "Câu hỏi nằm ngoài phạm vi kho tri thức về đái tháo đường; không đủ nguồn để trả lời.",
                "answer_mode": "extractive-offline (không gọi LLM)", "graph_backend": "NetworkX", "snippets": [], "sources": [],
                "subgraph": {"nodes": [], "edges": []}, "entity_matches": [],
                "intent": profile["primary_intent"], "context_used": profile["context_used"],
                "effective_question": profile["effective_question"],
                "trace": [{"agent":"QueryRouter", "action":"Từ chối ngoài phạm vi", "detail":"Chỉ số Diabetes"}]}
    query = profile["expanded_query"]
    v = _vector_candidates(query, facts, docs, 30, profile)
    g, graph_backend = _graph_candidates(query, facts, hops, 48) if method != "vector" else ([], "không truy vấn KG (Vector RAG)")
    items = {}
    for r in v:
        entry = items.setdefault(_id(r), {"kind": r["kind"], "item": r["item"], "vector_score": 0, "graph_score": 0})
        entry["vector_score"] = r["score"]
    for r in g:
        entry = items.setdefault(_id(r), {"kind": r["kind"], "item": r["item"], "vector_score": 0, "graph_score": 0})
        entry["graph_score"] = r["score"]
    for entry in items.values():
        if method == "vector":
            entry["score"] = entry["vector_score"]
        elif method == "graph":
            entry["score"] = entry["graph_score"]
        else:
            entry["score"] = round(alpha * entry["vector_score"] + (1-alpha) * entry["graph_score"], 5)
    selected = [e for e in sorted(items.values(), key=lambda d: d["score"], reverse=True)
                if e["score"] > 0][:top_k]
    source_by_id = {s["id"]: s for s in docs}
    snippets, sources, edges = [], {}, []
    for e in selected:
        item = e["item"]
        if e["kind"] == "fact":
            sr = source_by_id[item["source_id"]]
            snippet = item["evidence"]
            edges.append({"source": item["subject"], "target": item["object"],
                          "relation": item["predicate"], "source_id": item["source_id"]})
        else:
            sr = item
            snippet = (item.get("summary") if item.get("kind") == "web" else item.get("body", "")[:450]) or ""
            if sr["id"].startswith("SESSION"):
                for mention in extract_mentions(item.get("body", ""))[:8]:
                    edges.append({"source": sr["title"], "target": mention,
                                  "relation": "MENTIONS (tự động)", "source_id": sr["id"]})
        sources[sr["id"]] = {k: sr.get(k) for k in ("id", "title", "publisher", "url", "kind")}
        snippets.append({"id": item["id"], "kind": e["kind"], "text": snippet,
                         "source_id": sr["id"], "score": round(e["score"], 4),
                         "vector_score": e["vector_score"], "graph_score": e["graph_score"],
                         "subject": item.get("subject") if e["kind"] == "fact" else None,
                         "predicate": item.get("predicate") if e["kind"] == "fact" else None,
                         "object": item.get("object") if e["kind"] == "fact" else None})
    # Không sử dụng LLM; câu trả lời được diễn đạt theo ý định nhưng từng ý vẫn
    # phải trỏ tới snippet và source_id đã truy xuất.
    answer = _compose_answer(profile, snippets)
    nodes = sorted({x["source"] for x in edges} | {x["target"] for x in edges})
    return {"question": question, "method": method, "top_k": top_k, "hops": hops, "alpha": alpha,
            "answer": answer, "answer_mode": "extractive-offline (không gọi LLM)",
            "graph_backend": graph_backend,
            "snippets": snippets, "sources": list(sources.values()),
            "subgraph": {"nodes": [{"id": x} for x in nodes], "edges": edges},
            "entity_matches": profile["entities"], "intent": profile["primary_intent"],
            "context_used": profile["context_used"], "effective_question": profile["effective_question"],
            "trace": [
                {"agent": "QueryRouter", "action": "Nhận câu hỏi; nhận diện ý định/ngữ cảnh", "detail": f"{method} · {profile['primary_intent']}"},
                {"agent": "LanguageUnderstandingAgent", "action": "Mở rộng đồng nghĩa và câu hỏi nối tiếp", "detail": "Có dùng ngữ cảnh" if profile["context_used"] else "Câu hỏi độc lập"},
                {"agent": "VectorSearchAgent", "action": "Word + character TF-IDF retrieval", "detail": f"{len(v)} ứng viên", "used": method != "graph"},
                {"agent": "GraphQueryAgent", "action": "KG query entity/hops", "detail": f"{len(g)} quan hệ · backend {graph_backend}", "used": method != "vector"},
                {"agent": "FusionAgent", "action": "Xếp hạng/ngắt top-k", "detail": f"{len(snippets)} bằng chứng"},
                {"agent": "AnswerAgent", "action": "Trích rút và gắn nguồn", "detail": "Không tự sinh kiến thức không có nguồn"},
            ]}


def find_path(start, end, hops=3):
    """Đường đi nhiều chặng trong KG đã được biên tập (không duyệt document shortcut)."""
    g = nx.Graph()
    for fact in seed_data()["facts"]:
        g.add_edge(fact["subject"], fact["object"], predicate=fact["predicate"],
                   source_id=fact["source_id"])
    aliases = {normalize(node): node for node in g.nodes}
    for a, b in ALIASES.items():
        if b in g:
            aliases[normalize(a)] = b
    start, end = aliases.get(normalize(start)), aliases.get(normalize(end))
    if not start or not end:
        return {"found": False, "reason": "Không tìm thấy một trong hai thực thể trong KB."}
    try:
        path = nx.shortest_path(g, start, end)
    except nx.NetworkXNoPath:
        return {"found": False, "reason": "Không có đường nối hai thực thể."}
    if len(path) - 1 > max(1, min(int(hops), 5)):
        return {"found": False, "reason": f"Đường ngắn nhất dài {len(path)-1} chặng, vượt giới hạn đang đặt."}
    edges = [{"source": a, "target": b, **g.edges[a, b]} for a, b in zip(path, path[1:])]
    return {"found": True, "path": path, "hops": len(path)-1, "edges": edges}


def escape_cypher(s):
    return "'" + str(s).replace('\\', '\\\\').replace("'", "\\'").replace("\n", " ") + "'"


def cypher_export(data=None):
    """Script MERGE chạy được trên Neo4j (đủ mệnh đề + provenance)."""
    data = data or all_data()
    lines = ["// Sinh từ knowledge/seed.json; dữ kiện có provenance.",
             "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE;",
             "CREATE CONSTRAINT source_id_unique IF NOT EXISTS FOR (n:Source) REQUIRE n.id IS UNIQUE;"]
    ent = sorted({f["subject"] for f in data["facts"]} | {f["object"] for f in data["facts"]})
    for x in ent:
        lines.append(f"MERGE (n:Entity {{id:{escape_cypher(x)}}}) SET n.name={escape_cypher(x)};")
    for src in data["sources"]:
        lines.append("MERGE (s:Source {id:" + escape_cypher(src["id"]) + "}) SET s.title=" + escape_cypher(src["title"]) + ", s.url=" + escape_cypher(src["url"]) + ", s.publisher=" + escape_cypher(src["publisher"]) + ";")
    for f in data["facts"]:
        p = f["predicate"]
        assert re.fullmatch(r"[A-Z_]+", p)
        lines.append(f"MATCH (a:Entity {{id:{escape_cypher(f['subject'])}}}), (b:Entity {{id:{escape_cypher(f['object'])}}}) "
                     f"MERGE (a)-[r:{p} {{fact_id:{escape_cypher(f['id'])}}}]->(b) "
                     f"SET r.source_id={escape_cypher(f['source_id'])}, r.evidence={escape_cypher(f['evidence'])};")
    for f in data["facts"]:
        lines.append("MATCH (s:Source {id:" + escape_cypher(f["source_id"]) + "}), "
                     "(e:Entity {id:" + escape_cypher(f["subject"]) + "}) "
                     "MERGE (s)-[r:SUPPORTS {fact_id:" + escape_cypher(f["id"]) + "}]->(e);")
    for src in data["sources"]:
        if not src.get("body"):
            continue
        for i, ent in enumerate(extract_mentions(src["body"])):
            lines.append("MATCH (s:Source {id:" + escape_cypher(src["id"]) + "}), "
                         "(e:Entity {id:" + escape_cypher(ent) + "}) "
                         "MERGE (s)-[r:MENTIONS {mention_id:" + escape_cypher(f"{src['id']}:{i}") + "}]->(e);")
    return "\n".join(lines)+"\n"


def graphml_export(path):
    nx.write_graphml(graph(), str(path))
