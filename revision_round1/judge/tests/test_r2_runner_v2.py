import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import r2_runner_v2 as r

def test_packet_and_pilot():
    rows=r.packet(); ids=r.pilot_ids(rows)
    assert len(ids)==10 and sum(x['answerable'] for x in rows if x['query_id'] in ids)==8
    for x in rows:
        a=r.payload(x,'B3',r.MODELS[0]); b=r.payload(x,'B3',r.MODELS[1])
        assert a['messages']==b['messages']
        assert x['judge_inputs']['B3']['code_baseline']=='B4'
        assert x['judge_inputs']['B5']['code_baseline']=='B6'

def test_offline(tmp_path):
    def fail(*a,**k): raise AssertionError('network called')
    result=r.run(tmp_path,send=fail)
    assert result['evaluations']==40 and not list(tmp_path.iterdir())

def test_score_validation():
    assert r.parse('{"status":"ok","semantic_accuracy":0.5,"faithfulness":1,"rationale":"evidence"}',True)['faithfulness']==1
    for content in ['{}','{"status":"ok","semantic_accuracy":2,"faithfulness":1,"rationale":"e"}','{"status":"ok","semantic_accuracy":true,"faithfulness":1,"rationale":"e"}']:
        with pytest.raises(ValueError): r.parse(content,True)
    with pytest.raises(ValueError): r.parse('{"status":"ok","semantic_accuracy":0,"faithfulness":1,"rationale":"e"}',False)

def test_exact_model_preflight(tmp_path,monkeypatch):
    monkeypatch.setenv('OMNIROUTE_API_KEY','test-key-not-in-packet')
    def send(path,key,body=None):
        assert path=='/models'
        return {'data':[{'id':'cx/not-the-requested-model'},{'id':r.MODELS[1]}]}
    with pytest.raises(RuntimeError,match='exact model absent'): r.run(tmp_path,authorize=True,send=send)
    assert not list(tmp_path.iterdir())

def test_identity_mismatch_is_preserved_and_stops(tmp_path,monkeypatch):
    monkeypatch.setenv('OMNIROUTE_API_KEY','test-key-not-in-packet')
    def send(path,key,body=None):
        if path=='/models': return {'data':[{'id':m,'root':('deepseek-v4-flash-low' if m==r.MODELS[1] else 'gpt-6-luna-low')} for m in r.MODELS]}
        return {'model':'different-model','usage':{'prompt_tokens':12,'completion_tokens':2},'choices':[{'message':{'content':'{}'}}]}
    with pytest.raises(RuntimeError,match='returned model'): r.run(tmp_path,authorize=True,send=send)
    attempts=list(tmp_path.glob('attempt-*.json'))
    assert len(attempts)==1
    record=json.loads(attempts[0].read_text())
    assert record['state']=='error' and record['response']['model']=='different-model'
    with pytest.raises(RuntimeError,match='manual reconciliation'): r.run(tmp_path,authorize=True,send=send)
    assert len(list(tmp_path.glob('attempt-*.json')))==1

def test_timeout_is_uncertain_and_not_retried(tmp_path,monkeypatch):
    monkeypatch.setenv('OMNIROUTE_API_KEY','test-key-not-in-packet')
    def send(path,key,body=None):
        if path=='/models': return {'data':[{'id':m,'root':('deepseek-v4-flash-low' if m==r.MODELS[1] else 'gpt-6-luna-low')} for m in r.MODELS]}
        raise TimeoutError('deadline')
    with pytest.raises(RuntimeError,match='deadline'): r.run(tmp_path,authorize=True,send=send)
    record=json.loads(next(tmp_path.glob('attempt-*.json')).read_text())
    assert record['state']=='uncertain'
    with pytest.raises(RuntimeError,match='manual reconciliation'): r.run(tmp_path,authorize=True,send=send)

def test_approved_normalization_and_deepseek_root():
    assert r.RETURNED_MODELS[r.MODELS[0]] == 'gpt-6-luna'
    assert r.RETURNED_MODELS[r.MODELS[1]] == 'deepseek-v4-flash'
    assert r.RETURNED_MODELS[r.MODELS[0]] != 'unrelated'

def test_deepseek_root_not_inferred_from_route(tmp_path,monkeypatch):
    monkeypatch.setenv('OMNIROUTE_API_KEY','test-key-not-in-packet')
    def send(path,key,body=None):
        assert path=='/models'
        return {'data':[{'id':r.MODELS[0]}, {'id':r.MODELS[1],'root':'unrelated'}]}
    with pytest.raises(RuntimeError,match='catalog root'): r.run(tmp_path,authorize=True,send=send)

def test_valid_success_status_is_preserved():
    content='{"status":"valid","semantic_accuracy":0,"faithfulness":0.5,"rationale":"Evidence is insufficient."}'
    parsed=r.parse(content,True)
    assert parsed['status']=='valid' and parsed['semantic_accuracy']==0 and parsed['faithfulness']==0.5
