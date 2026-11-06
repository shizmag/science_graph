import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('runner',Path(__file__).resolve().parents[1]/'run_openrouter_judge.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

def test_existing_session_reused_without_writing(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'WORKSPACE',tmp_path)
    p=tmp_path/'sessions'/'resume.json';p.parent.mkdir();p.write_text(json.dumps({'session_id':'resume','model':r.MODEL}))
    monkeypatch.setattr(r,'bridge',lambda *x: pytest.fail('session should be reused'))
    r.ensure_session('resume')

def test_session_mismatch_is_error(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'WORKSPACE',tmp_path)
    p=tmp_path/'sessions'/'resume.json';p.parent.mkdir();p.write_text(json.dumps({'session_id':'resume','model':'other'}))
    with pytest.raises(ValueError,match='different'):r.ensure_session('resume')

def test_missing_session_registered_once(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'WORKSPACE',tmp_path);calls=[];monkeypatch.setattr(r,'bridge',lambda *x:calls.append(x))
    r.ensure_session('new-session');assert calls==[('session','--id','new-session','--model',r.MODEL)]

def test_no_fabricated_judge_data():
    task={'evaluation_id':'x','input_sha256':'y','answerable':False}
    with pytest.raises(KeyError):r.normalize(task,'session',{'status':'scored','semantic_accuracy':0,'faithfulness':1})
    o={'status':'scored','semantic_accuracy':0,'faithfulness':1,'semantic_accuracy_rationale':'x','faithfulness_rationale':'y','evidence_identifiers':['source'],'unsupported_claims':[],'unresolved_citation_mapping':False,'error_reason':None}
    assert r.normalize(task,'session',o)['semantic_accuracy']==0

def test_key_environment_precedes_dotenv(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'ROOT',tmp_path)
    (tmp_path/'.env').write_text('OPENROUTER_API_KEY="local-key"\n')
    monkeypatch.setenv('OPENROUTER_API_KEY','env-key')
    assert r.load_api_key()=='env-key'
    monkeypatch.delenv('OPENROUTER_API_KEY')
    assert r.load_api_key()=='local-key'
