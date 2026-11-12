"""Neo4j là backend tùy chọn. Chỉ đọc ở runtime web, chỉ đồng bộ bằng CLI."""
import os


def _connection_settings():
    """Accept both the project variable and Aura credential-file naming."""
    return {
        "uri": os.environ.get("NEO4J_URI", ""),
        "user": os.environ.get("NEO4J_USER") or os.environ.get("NEO4J_USERNAME", "neo4j"),
        "password": os.environ.get("NEO4J_PASSWORD", ""),
        "database": os.environ.get("NEO4J_DATABASE", "neo4j"),
    }


def neo4j_status():
    settings = _connection_settings()
    if not settings["uri"]:
        return {"configured": False, "connected": False, "backend": "NetworkX", "note": "Chưa đặt NEO4J_URI; KB vẫn chạy nội bộ."}
    try:
        from neo4j import GraphDatabase
        with GraphDatabase.driver(settings["uri"], auth=(settings["user"], settings["password"]),
                                  connection_timeout=2, max_connection_lifetime=15) as driver:
            driver.verify_connectivity()
            with driver.session(database=settings["database"]) as session:
                n = session.run("MATCH (n:Entity) RETURN count(n) AS n").single()["n"]
                e = session.run("MATCH ()-[r]->() RETURN count(r) AS n").single()["n"]
        return {"configured": True, "connected": True, "backend": "Neo4j",
                "database": settings["database"], "entity_nodes": n, "relationships": e}
    except Exception as exc:
        return {"configured": True, "connected": False, "backend": "NetworkX", "note": f"Neo4j chưa kết nối ({exc.__class__.__name__}); xem Render Environment."}


def fetch_neighbor_facts(entities, hops=2, limit=100):
    """Truy vấn CẠNH THỰC TẾ từ Neo4j khi đã cấu hình.

    Luồng read-only, chỉ MATCH; mở rộng từng chặng bằng tên Entity;
    không ghép văn bản người dùng vào Cypher, tránh injection.
    Trả None nếu không thiết lập Neo4j, ném lỗi nếu có thiết lập nhưng kết nối lỗi
    để lớp RAG ghi nhãn fallback NetworkX.
    """
    settings = _connection_settings()
    if not settings["uri"]:
        return None
    if not entities:
        return []
    from neo4j import GraphDatabase
    frontier = set(str(x) for x in entities[:6])
    seen_nodes = set(frontier)
    found = {}
    statement = """
    MATCH (a:Entity)-[r]->(b:Entity)
    WHERE (a.id IN $frontier OR b.id IN $frontier)
      AND r.fact_id IS NOT NULL AND r.source_id IS NOT NULL
    RETURN a.id AS subject, type(r) AS predicate, b.id AS object,
           r.fact_id AS id, r.source_id AS source_id, r.evidence AS evidence
    LIMIT $limit
    """
    with GraphDatabase.driver(
        settings["uri"],
        auth=(settings["user"], settings["password"]),
        connection_timeout=2, connection_acquisition_timeout=3,
        max_connection_lifetime=30,
    ) as driver:
        with driver.session(database=settings["database"], default_access_mode="READ") as session:
            for _ in range(max(1, min(int(hops), 3))):
                if not frontier:
                    break
                records = session.run(statement, frontier=list(frontier), limit=int(min(limit, 200)))
                next_frontier = set()
                for rec in records:
                    f = dict(rec)
                    if not f.get("id") or not f.get("evidence"):
                        continue
                    found[f["id"]] = f
                    for n in (f["subject"], f["object"]):
                        if n not in seen_nodes:
                            next_frontier.add(n)
                seen_nodes |= next_frontier
                frontier = next_frontier
    return list(found.values())
