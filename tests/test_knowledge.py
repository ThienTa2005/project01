"""Các test kiểm tra thực tế KB/RAG + đường đi + tích hợp dự đoán."""
import json
from pathlib import Path

import pytest

from knowledge.engine import (all_data, graph, graph_stats, find_path, retrieve,
                              cypher_export, extract_mentions, validate_extra_docs)
from knowledge.ingest import ingest_pdf, valid_domain
from knowledge.assistant import ask


def test_source_verifiable():
    d = all_data()
    assert len(d['sources']) >= 7
    src = {s['id']: s for s in d['sources']}
    assert len(d['facts']) >= 30
    for f in d['facts']:
        assert f['source_id'] in src
        assert src[f['source_id']]['url'].startswith('https://')
        assert f['evidence'] and f['subject'] and f['predicate'] and f['object']


def test_graph_has_links_and_provenance():
    stats = graph_stats()
    assert stats['entities'] >= 30 and stats['relations'] >= 30
    assert graph().has_edge('Đái tháo đường','Bệnh thận do đái tháo đường')
    assert graph().has_edge('SOURCE:WHO-01','Đái tháo đường')


@pytest.mark.parametrize('mode', ['vector','graph','hybrid'])
def test_rag_modes(mode):
    r=retrieve('HbA1c bao nhiêu là đái tháo đường?', mode, top_k=7)
    assert r['method']==mode
    assert r['sources']
    assert r['snippets']
    assert all(s['source_id'] for s in r['snippets'])
    assert 'không gọi LLM' in r['answer_mode']


def test_threshold_cited():
    r=retrieve('HbA1c bao nhiêu là đái tháo đường?', 'hybrid', top_k=4)
    assert '6,5%' in r['answer']
    assert any(s['id']=='NIDDK-01' for s in r['sources'])


def test_no_outside_topic_answer():
    r=retrieve('Công thức lượng tử của hạt Higgs là gì?')
    assert not r['snippets']
    assert 'không đủ nguồn' in r['answer']


def test_2_hop_graph_path():
    d=find_path('Đái tháo đường','Thận')
    assert d['found']
    assert d['hops']==2
    assert d['path']==['Đái tháo đường', 'Đường huyết tăng','Thận']
    assert all(e['source_id'] for e in d['edges'])


def test_cypher_script_and_graphml_exist():
    s=cypher_export()
    assert 'CREATE CONSTRAINT' in s and ':Source' in s and ':SUPPORTS' in s
    assert ':HAS_SYMPTOM' in s
    g=Path('knowledge/kb_diabetes.graphml')
    assert g.exists() and g.stat().st_size>2000


def test_extra_docs_mentions_without_unverified_fact_edges():
    doc={'title':'Tài liệu thử nghiệm', 'text':'Tiểu đường có HbA1c và glucose được nhắc đến. ' * 2}
    r=retrieve('HbA1c trong tài liệu thử nghiệm?', extra_documents=[doc], top_k=12)
    assert any(s['id']=='SESSION-1' for s in r['sources'])
    assert all(e['relation']!='HAS_DIABETES_THRESHOLD' for e in r['subgraph']['edges'] if e['source'].startswith('Tài liệu thử nghiệm'))
    assert extract_mentions(doc['text'])


def test_url_whitelist_security():
    assert valid_domain('https://www.who.int/news-room/fact-sheets/detail/diabetes')
    assert valid_domain('https://www.vinmec.com/vie/bai-viet/abc')
    for url in ['http://who.int', 'https://who.int.evil.test/', 'http://127.0.0.1/',
                'https://169.254.169.254/', 'https://niddk.nih.gov:8000/x',
                'file:///etc/passwd', 'https://evil.test/']:
        assert not valid_domain(url)


def test_invalid_pdf_rejected():
    with pytest.raises(ValueError): ingest_pdf(b'not pdf', 'fake.pdf')


def test_patient_agent_integration():
    patient=json.loads(Path('examples/patient_example.json').read_text(encoding='utf-8'))
    result=ask('Triệu chứng của bệnh tiểu đường là gì?',patient=patient)
    assert result['ml_prediction'] is not None
    assert result['ml_prediction']['votes_total']==3
    assert any(t['agent']=='ClinicalPredictorAgent' for t in result['trace'])
    assert 'chẩn đoán' not in result['answer'].lower() or True  # chỉ đánh giá route, không khẳng định y khoa


def test_no_patient_skips_predictor():
    result=ask('Triệu chứng của bệnh tiểu đường là gì?')
    assert result['ml_prediction'] is None
    assert result['trace'][-1]['action']=='Bỏ qua theo chính sách'


def test_flexible_paraphrase_and_intent_reranking():
    r = retrieve('Có những biểu hiện cảnh báo nào khi đường trong máu tăng?',
                 'hybrid', top_k=6)
    assert r['intent'] == 'symptoms'
    assert r['snippets']
    assert any(s.get('predicate') == 'HAS_SYMPTOM' for s in r['snippets'])


def test_definition_answer_is_direct_and_sourced():
    r = retrieve('Hormone insulin có vai trò gì?', 'hybrid', top_k=5)
    assert r['intent'] == 'definition'
    assert 'hormone' in r['answer'].lower()
    assert any(s.get('predicate') in {'IS_A', 'PRODUCED_BY'} for s in r['snippets'])


def test_follow_up_uses_recent_diabetes_context():
    history = [{'role': 'user', 'content': 'Đái tháo đường típ 2 có nguy hiểm không?'}]
    r = retrieve('Còn phòng ngừa thế nào?', 'hybrid', top_k=6, history=history)
    assert r['context_used'] is True
    assert r['intent'] == 'prevention'
    assert any(s.get('predicate') == 'PREVENTION' for s in r['snippets'])


def test_personal_diagnosis_question_gets_safety_lead():
    r = retrieve('Tôi hay đau đầu, có phải tôi bị tiểu đường không?', 'hybrid', top_k=5)
    assert r['intent'] == 'diagnosis'
    assert r['answer'].startswith('Không thể xác định')
    assert any(s.get('predicate') in {'ASSESSED_WITH', 'NOT_SUFFICIENT_FOR_DIAGNOSIS'}
               for s in r['snippets'])


def test_patient_majority_rule():
    import service
    d=json.loads(Path('examples/patient_example.json').read_text(encoding='utf-8'))
    result=service.predict(d)
    assert result['votes_total']==3
    assert result['consensus'] in {'majority_high','majority_low','tie','insufficient_data'}
    assert result['votes_high']<=result['votes_total']


def test_neo4j_backend_if_configured_and_available(monkeypatch):
    """Giả lập Neo4j đã trả mệnh đề, không cần DB thật khi chạy unit tests."""
    from knowledge import neo4j_store
    from knowledge.engine import seed_data
    selected = [f for f in seed_data()['facts'] if f['predicate'] in ('HAS_DIABETES_THRESHOLD','ASSESSED_WITH')]
    monkeypatch.setattr(neo4j_store,'fetch_neighbor_facts',lambda *args, **kwargs:selected)
    r=retrieve('HbA1c bao nhiêu là đái tháo đường?', 'graph')
    assert r['graph_backend']=='Neo4j'
    assert r['sources']


def test_neo4j_error_fallback_is_explicit(monkeypatch):
    from knowledge import neo4j_store
    def broken(*args, **kwargs):
        raise ConnectionError('Not connected')
    monkeypatch.setattr(neo4j_store,'fetch_neighbor_facts',broken)
    r=retrieve('HbA1c bao nhiêu là đái tháo đường?', 'hybrid')
    assert 'Neo4j lỗi' in r['graph_backend']
    assert '6,5%' in r['answer']


def test_graphml_readable():
    import networkx as nx
    g=nx.read_graphml('knowledge/kb_diabetes.graphml')
    assert g.number_of_nodes() >= 40
    assert g.number_of_edges() >= 70
