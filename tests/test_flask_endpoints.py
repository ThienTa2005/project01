"""Tích hợp Flask. pytest sẽ skip nếu chưa cài Flask trên môi trường kiểm thử."""
import pytest

pytest.importorskip('flask')

from app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


def test_health_and_kb(client):
    r = client.get('/healthz')
    assert r.status_code == 200 and r.json['kb']['relations'] >= 30
    r = client.get('/api/kb')
    assert r.status_code == 200 and len(r.json['sources']) >= 7


def test_graph_path_and_sources(client):
    g = client.get('/api/kb/graph')
    assert g.status_code == 200 and len(g.json['edges']) >= 36
    r = client.get('/api/kb/path?start=%C4%90%C3%A1i%20th%C3%A1o%20%C4%91%C6%B0%E1%BB%9Dng&end=Th%E1%BA%ADn')
    assert r.status_code == 200 and r.json['hops']==2
    assert client.get('/kb-diabetes/kb_diabetes.cypher').status_code == 200


def test_rag_endpoint(client):
    r=client.post('/api/chat',json={'question':'HbA1c bao nhiêu là đái tháo đường?', 'method':'hybrid'})
    assert r.status_code==200 and '6,5%' in r.json['answer']
    assert r.json['sources']


def test_rag_follow_up_history(client):
    r=client.post('/api/chat',json={
        'question':'Còn cách phòng ngừa?',
        'method':'hybrid',
        'history':[{'role':'user','content':'Tiểu đường típ 2 có yếu tố nguy cơ nào?'}]
    })
    assert r.status_code==200
    assert r.json['context_used'] is True
    assert r.json['intent']=='prevention'
    assert any(s.get('predicate')=='PREVENTION' for s in r.json['snippets'])


def test_patient_and_missing_source(client):
    pred=client.post('/api/predict',json={'common':{'age':50}})
    assert pred.status_code==200
    forbidden=client.post('/api/kb/ingest',json={'url':'http://127.0.0.1:5000/'})
    assert forbidden.status_code==400
