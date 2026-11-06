import importlib.util,json
from pathlib import Path
import pytest
p=Path(__file__).resolve().parents[1]/'integrity.py';sp=importlib.util.spec_from_file_location('integrity',p);i=importlib.util.module_from_spec(sp);sp.loader.exec_module(i)

def test_suspicious_is_flag_not_proof():
    assert i.suspicious({'semantic_accuracy_rationale':'Semantic accuracy is determined by comparing the answer','faithfulness_rationale':'The task is designated negative.'})
    assert not i.suspicious({'semantic_accuracy_rationale':'The answer identifies the intervention in source 1.','faithfulness_rationale':'Source 1 contains that finding.'})

def test_negative_label_conflict_and_nullability():
    t={'evaluation_id':'E'+'a'*24,'input_sha256':'x','answerable':False,'evidence_identifiers':['source 1']}
    r={'evaluation_id':t['evaluation_id'],'input_sha256':'x','rubric_version':'negative-v1','rubric_sha256':i.negative_hash(),'judge_session_id':'s','status':'scored','negative_safety':0.5,'potential_label_conflict':False,'rationale':'Source 1 does not support the assertion.','evidence_identifiers':['source 1'],'error_reason':None}
    i.validate_negative(t,r);r['potential_label_conflict']=True
    with pytest.raises(ValueError):i.validate_negative(t,r)
    r['negative_safety']=None;r['status']='label_conflict';i.validate_negative(t,r)
    r['evidence_identifiers']=['invented']
    with pytest.raises(ValueError):i.validate_negative(t,r)

def test_approval_hash_binding(tmp_path,monkeypatch):
    monkeypatch.setattr(i,'REVIEW_DIR',tmp_path)
    data={'x':1};h=i.sha(i.canonical(data));p=tmp_path/(h+'.json')
    p.write_text(json.dumps({'record_sha256':h,'state':'accepted','reviewer':'author','provenance_evidence':'session inspected','grounding_review':'evidence reviewed'}))
    assert i.approved(data)
    assert not i.approved({'x':2})
