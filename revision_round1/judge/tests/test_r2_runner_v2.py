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
        return {'data':[{'id':'cx/gpt-6-luna-low'},{'id':r.MODELS[1]}]}
    with pytest.raises(RuntimeError,match='exact model absent'): r.run(tmp_path,authorize=True,send=send)
    assert not list(tmp_path.iterdir())
