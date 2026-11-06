import importlib.util, json
from pathlib import Path
import pytest
p=Path(__file__).resolve().parents[1]/'bridge.py';spec=importlib.util.spec_from_file_location('bridge',p);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

def test_determinism_and_accounting(tmp_path):
    a=tmp_path/'a';z=tmp_path/'z';b.prepare(a,tmp_path/'m1.json');b.prepare(z,tmp_path/'m2.json')
    m=json.loads((a/'manifest.json').read_text()); assert len(m['batches'])==20
    assert sum(len(x['tasks']) for x in m['batches'])==200
    assert (a/'manifest.json').read_bytes()==(z/'manifest.json').read_bytes()
    for f in (a/'tasks').glob('*.json'):
        t=json.loads(f.read_text());assert set(t)==b.TASK_KEYS; assert b.task_hash(t)==t['input_sha256']
        assert 'paper_baseline' not in f.read_text() and 'historical_scores' not in f.read_text()
    private=json.loads((tmp_path/'m1.json').read_text());assert len(private['mapping'])==200
    assert {x['baseline'] for x in private['mapping'].values()}=={'B3','B5'}

def test_validators():
    t={'evaluation_id':'abc','answerable':False,'input_sha256':'x','evidence_identifiers':['source 1']}
    r={'evaluation_id':'abc','input_sha256':'x','rubric_version':'v2','judge_session_id':'session','status':'scored','semantic_accuracy':None,'faithfulness':0.5,'semantic_accuracy_rationale':'Not applicable.','faithfulness_rationale':'Some support.','evidence_identifiers':['source 1'],'unsupported_claims':['claim'],'unresolved_citation_mapping':False,'error_reason':None}
    b.validate_result(t,r);r['faithfulness']=3
    with pytest.raises(ValueError): b.validate_result(t,r)
    r['faithfulness']=1;r['semantic_accuracy']=0
    with pytest.raises(ValueError): b.validate_result(t,r)
    r['semantic_accuracy']=None;r['evidence_identifiers']=['fabricated']
    with pytest.raises(ValueError): b.validate_result(t,r)

def example(t,session='test'):
    return {'evaluation_id':t['evaluation_id'],'input_sha256':t['input_sha256'],'rubric_version':'v2','judge_session_id':session,'status':'scored','semantic_accuracy':1 if t['answerable'] else None,'faithfulness':0.5,'semantic_accuracy_rationale':'synthetic fixture','faithfulness_rationale':'synthetic fixture','evidence_identifiers':t['evidence_identifiers'][:1],'unsupported_claims':[],'unresolved_citation_mapping':False,'error_reason':None}

def test_resume_duplicate_and_invalid(tmp_path):
    w=tmp_path/'w';b.prepare(w,tmp_path/'private.json');b.save_session(w,'test','synthetic fixture - not a real judge')
    n=b.next_task(w);t=b.read(n['task_path']);draft=tmp_path/'draft.json';b.write(draft,example(t));path=b.submit(w,draft)
    assert b.next_task(w)['evaluation_id']!=t['evaluation_id']
    with pytest.raises(ValueError):b.submit(w,draft)
    other=Path(path).with_name(t['evaluation_id']+'.v1.json');other.write_text(Path(path).read_text())
    assert b.status(w)['invalid']
    other.unlink(); Path(path).write_text('{broken')
    assert b.status(w)['invalid']

def test_hash_and_evidence_isolation(tmp_path):
    w=tmp_path/'w';private=tmp_path/'private.json';b.prepare(w,private);p=b.read(private)
    import sys
    rows={r['query_id']:r for r in b.source.packet()}
    for eid,meta in p['mapping'].items():
        t=b.read(w/'tasks'/f'{eid}.json');assert t['generator_visible_evidence']==rows[meta['query_id']]['judge_inputs'][meta['baseline']]['generator_visible_evidence']
    n=b.next_task(w);f=Path(n['task_path']);t=b.read(f);t['generated_answer']+=' altered';b.write(f,t)
    with pytest.raises(ValueError):b.status(w)

def test_export_paired_fixture(tmp_path):
    w=tmp_path/'w';private=tmp_path/'private.json';b.prepare(w,private);b.save_session(w,'test','synthetic fixture')
    mapping=b.read(private)['mapping'];query=next(v['query_id'] for v in mapping.values());ids=[i for i,v in mapping.items() if v['query_id']==query]
    for eid in ids:
        t=b.read(w/'tasks'/f'{eid}.json');draft=tmp_path/'draft.json';b.write(draft,example(t));b.submit(w,draft)
    s=b.export(w,private,tmp_path/'out');assert s['metrics']['faithfulness']['paired_n']==1
    assert (tmp_path/'out'/'query_pairs.csv').is_file()

def test_correction_keeps_original_and_requires_reason(tmp_path):
    w=tmp_path/'w';b.prepare(w,tmp_path/'private.json');b.save_session(w,'test','fixture')
    t=b.read(b.next_task(w)['task_path']);d=tmp_path/'d.json';b.write(d,example(t));original=Path(b.submit(w,d));before=original.read_bytes()
    replacement=example(t);replacement['faithfulness']=1;b.write(d,replacement)
    with pytest.raises(ValueError):b.submit(w,d)
    corrected=Path(b.submit(w,d,'fixture correction'))
    assert original.read_bytes()==before and corrected.exists() and not b.status(w)['invalid']
    assert b.check(w)[2][t['evaluation_id']]['faithfulness']==1

def test_partial_metric_and_unevaluable():
    t={'evaluation_id':'E'+'a'*24,'answerable':True,'input_sha256':'x','evidence_identifiers':['source']}
    r=example(t);r.update(status='partial',faithfulness=None,error_reason={'code':'evidence','detail':'unusable'})
    b.validate_result(t,r)
    r.update(status='unevaluable',semantic_accuracy=None,faithfulness=None)
    b.validate_result(t,r)
