// Test fixture for multi-file Cypher KB loading.
MERGE (s:Source {id:'SRC-2'}) SET s.title='Source 2', s.url='https://example.org/2', s.publisher='Test';
MATCH (a:Entity {id:'A2'}), (b:Entity {id:'B2'}) MERGE (a)-[r:RELATED_TO {fact_id:'F-2'}]->(b) SET r.source_id='SRC-2', r.evidence='Evidence 2';
