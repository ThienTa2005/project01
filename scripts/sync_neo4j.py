"""Chỉ chạy CLI trên máy có quyền ghi:
NEO4J_URI=neo4j+s://... NEO4J_USER=neo4j NEO4J_PASSWORD=... python -m scripts.sync_neo4j
"""
import os
import re
import sys

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from knowledge.engine import all_data, extract_mentions


def main():
    from neo4j import GraphDatabase
    uri, password = os.getenv("NEO4J_URI"), os.getenv("NEO4J_PASSWORD")
    user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME", "neo4j")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    if not uri or not password:
        raise SystemExit("Thiếu NEO4J_URI hoặc NEO4J_PASSWORD; tuyệt đối không commit credentials.")
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            session.run("CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE").consume()
            session.run("CREATE CONSTRAINT source_id_unique IF NOT EXISTS FOR (n:Source) REQUIRE n.id IS UNIQUE").consume()
            d = all_data()
            for f in d["facts"]:
                for k in ("subject", "object"):
                    session.run("MERGE (e:Entity {id:$id}) SET e.name=$id", id=f[k]).consume()
            for src in d["sources"]:
                session.run("MERGE (s:Source {id:$id}) SET s.title=$title,s.url=$url,s.publisher=$publisher", **{k:src[k] for k in ("id", "title", "url", "publisher")}).consume()
            for f in d["facts"]:
                pred = f["predicate"]
                if not re.fullmatch("[A-Z_]+", pred):
                    raise ValueError("Predicate không hợp lệ")
                q = f"MATCH (a:Entity {{id:$subject}}),(b:Entity {{id:$object}}) MERGE (a)-[r:{pred} {{fact_id:$fact_id}}]->(b) SET r.source_id=$source_id,r.evidence=$evidence"
                session.run(q, subject=f["subject"], object=f["object"], fact_id=f["id"],
                            source_id=f["source_id"], evidence=f["evidence"]).consume()
                session.run("MATCH (s:Source {id:$source}),(e:Entity {id:$subject}) "
                            "MERGE (s)-[:SUPPORTS {fact_id:$fact_id}]->(e)",
                            source=f["source_id"], subject=f["subject"], fact_id=f["id"]).consume()
            for src in d["sources"]:
                if not src.get("body"):
                    continue
                for i, ent in enumerate(extract_mentions(src["body"])):
                    session.run("MATCH (s:Source {id:$source}),(e:Entity {id:$entity}) "
                                "MERGE (s)-[:MENTIONS {mention_id:$mention_id}]->(e)",
                                source=src["id"], entity=ent, mention_id=f"{src['id']}:{i}").consume()
            print(f"Đã đồng bộ {len(d['sources'])} nguồn, {len(d['facts'])} quan hệ y khoa + provenance/mentions vào Neo4j database '{database}'.")


if __name__ == '__main__':
    main()
