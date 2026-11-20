// Test fixture for multi-file Cypher KB loading.
MERGE (s:Source {id:'SRC-1'}) SET s.title='Source 1', s.url='https://example.org/1', s.publisher='Test';
MATCH (a:Entity {id:'A1'}), (b:Entity {id:'B1'}) MERGE (a)-[r:RELATED_TO {fact_id:'F-1'}]->(b) SET r.source_id='SRC-1', r.evidence='Evidence 1';
