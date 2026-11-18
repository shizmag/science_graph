"""Offline edge cases and archive regression checks. No network allowed."""
import copy
import importlib.util
import json
import sys
import urllib.error
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runner'))
from common import HOME,ROOT,encode,sha,rows,verify_freeze
from scoring import validate,scores
from run import execute,body_for

@pytest.fixture
def task():
    return dict(task_id='mock-abc',query_id='qid',paper_configuration='B3',anonymized_configuration='answer_abc',judge='omniroute',status='ready',
        visible=dict(evaluation_id='abc',question='Q',candidate_answer='A helps. [Block 1]',generator_visible_evidence='<|source_id|>1 Block 1: A helps.',visible_identifiers=['<|source_id|>1']))

@pytest.fixture
def result():
    return dict(evaluation_id='abc',claims=[dict(id='c1',quote='A helps.',support='full',evidence_quote='A helps.',explanation='explicit')],
        citations=[dict(quote='[Block 1]',claim_id='c1',target='Block 1',resolution='resolved',source_quote='A helps.',support='full',explanation='explicit numbered span')],notes='')


def test_full(task,result):
    validate(result,task);s=scores(result);assert s['F']==s['CF']==s['coverage']==s['resolvability']==1


def test_partial(task,result):
    result['claims'][0]['support']='partial';result['citations'][0]['support']='partial'
    validate(result,task);assert scores(result)['F']==scores(result)['CF']==.5


def test_missing_citations(task,result):
    result['citations']=[];validate(result,task);s=scores(result)
    assert s['CF']==0 and s['CF_no_citation_na'] is None and s['CF_status']=='missing_required_citations' and s['compliance']==0


def test_wrong_source(task,result):
    result['citations'][0]['support']='unsupported';validate(result,task)
    assert scores(result)['F']==1 and scores(result)['CF']==0 and scores(result)['n_incorrect']==1

@pytest.mark.parametrize('resolution',['unresolved','ambiguous'])
def test_unresolvable(task,result,resolution):
    result['citations'][0].update(resolution=resolution,support='unassessable',source_quote='')
    validate(result,task);s=scores(result)
    assert s['CF']==0 and s['CF_resolved_only'] is None and s['n_incorrect']==0


def test_claimless(task,result):
    result.update(claims=[],citations=[]);validate(result,task)
    assert scores(result)['F'] is None and scores(result)['CF'] is None
    assert scores(result,empty=True)['F_status']=='empty_output'


def test_invented_block(task,result):
    task['visible']['generator_visible_evidence']='[Chunk] A helps.'
    with pytest.raises(ValueError,match='Block mapping'):validate(result,task)


def test_bad_identity(task,result):
    result['evaluation_id']='other'
    with pytest.raises(ValueError):validate(result,task)


def test_invalid_schema(task,result):
    result['extra']='no'
    with pytest.raises(Exception):validate(result,task)


def test_quote_validation(task,result):
    result['claims'][0]['evidence_quote']='invented'
    with pytest.raises(ValueError):validate(result,task)


def response(r,model='gpt-6-luna'):
    return encode(dict(model=model,choices=[dict(finish_reason='stop',message={'content':json.dumps(r)})],usage={'total_tokens':30})),{}

CFG=dict(model='cx/gpt-6-luna-low',aliases={'gpt-6-luna','cx/gpt-6-luna-low'})

def test_retry_resume_capture(task,result,tmp_path):
    calls=[]
    def send(url,key,body):
        calls.append(body)
        if len(calls)==1:return b'{"model":"gpt-6-luna","choices":[{"finish_reason":"stop","message":{"content":"bad json"}}]}',{}
        return response(result)
    assert execute(task,CFG,'http://mock','secretvalue',tmp_path,'hash',send,lambda n:None)=='accepted'
    assert len(calls)==2
    assert execute(task,CFG,'http://mock','secretvalue',tmp_path,'hash',send,lambda n:None)=='accepted'
    assert len(calls)==2
    attempts=list((tmp_path/'attempts').glob('*.json'));assert len(attempts)==2
    for p in attempts:
        a=json.loads(p.read_text());assert a['request_sha256']==sha(a['request_utf8'].encode())
        assert a['response_sha256']==sha(a['response_body'].encode())
        assert 'secretvalue' not in p.read_text()


def test_network_retry(task,result,tmp_path):
    calls=[]
    def send(*a):
        calls.append(1)
        if len(calls)==1:raise urllib.error.URLError('transient')
        return response(result)
    assert execute(task,CFG,'http://mock','secret',tmp_path,'h',send,lambda n:None)=='accepted'
    assert len(calls)==2


def test_schema_retry(task,result,tmp_path):
    calls=[]
    def send(*a):
        calls.append(1)
        return response(dict(result,unknown=1) if len(calls)==1 else result)
    assert execute(task,CFG,'http://mock','secret',tmp_path,'h',send,lambda n:None)=='accepted'
    assert len(calls)==2


def test_retry_bound(task,result,tmp_path):
    calls=[]
    def send(*a):
        calls.append(1);return response({})
    assert execute(task,CFG,'http://mock','secret',tmp_path,'h',send,lambda n:None)=='retry_exhausted'
    assert execute(task,CFG,'http://mock','secret',tmp_path,'h',send,lambda n:None)=='retry_exhausted'
    assert len(calls)==4


def test_empty_no_call(task,tmp_path):
    task['status']='empty_output'
    def send(*a):raise AssertionError('empty should not call')
    assert execute(task,CFG,'http://mock','secret',tmp_path,'h',send)=='empty_output'


def test_model_mismatch_no_retry(task,result,tmp_path):
    calls=[]
    def send(*a):calls.append(1);return response(result,'other-model')
    assert execute(task,CFG,'http://mock','secret',tmp_path,'h',send,lambda n:None)=='fatal'
    assert execute(task,CFG,'http://mock','secret',tmp_path,'h',send,lambda n:None)=='fatal'
    assert len(calls)==1


def test_all_inputs_and_blinding():
    records=rows(HOME/'evidence_manifest.jsonl');tasks=rows(HOME/'judge_tasks.jsonl')
    assert len(records)==160 and len(tasks)==320
    assert len({t['task_id'] for t in tasks})==320
    byhash={r['record_sha256']:r for r in records}
    for r in records:
        for k,d in r['sha256'].items():
            value=r[k];assert sha(value.encode() if isinstance(value,str) else encode(value))==d
        unhashed={k:v for k,v in r.items() if k!='record_sha256'}
        assert sha(encode(unhashed))==r['record_sha256']
        assert r['code_configuration']=={'B3':'B4','B5':'B6'}[r['paper_configuration']]
    for t in tasks:
        r=byhash[t['input_record_sha256']]
        assert t['query_id']==r['query_id'] and t['paper_configuration']==r['paper_configuration']
        assert t['visible']['candidate_answer']==r['cleaned_answer']
        assert set(t['visible'])=={'evaluation_id','question','candidate_answer','generator_visible_evidence','source_mapping_policy','visible_identifiers'}
        body=body_for(t,CFG);sent=body['messages'][1]['content']
        assert 'reference_answer' not in sent and 'paper_configuration' not in sent and 'historical_scores' not in sent
        assert sha(encode(t['visible']))==t['visible_sha256']
    for q in {t['query_id'] for t in tasks}:
        for p in ('B3','B5'):
            pair=[t for t in tasks if t['query_id']==q and t['paper_configuration']==p]
            assert pair[0]['visible']==pair[1]['visible']


def test_statistics_synthetic():
    spec=importlib.util.spec_from_file_location('stats',HOME/'statistical_analysis.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert m.icc([0,.5,1],[0,.5,1])==pytest.approx(1)
    assert m.icc([0,.5,1],[.2,.7,1.2])<1
    assert m.agreement(__import__('numpy').array([[0,0],[.5,.5],[1,1]]))['Spearman']==pytest.approx(1)


def test_protocol_freeze():
    assert len(verify_freeze())==64


def test_checkpoint_tamper(task,result,tmp_path):
    def send(*a):return response(result)
    execute(task,CFG,'http://mock','secret',tmp_path,'h',send,lambda n:None)
    path=next((tmp_path/'attempts').glob('*.json'));d=json.loads(path.read_text());d['response_body']='changed';path.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='corrupted'):execute(task,CFG,'http://mock','secret',tmp_path,'h',send,lambda n:None)


def test_started_uncertainty(task,result,tmp_path):
    from common import write_json
    write_json(tmp_path/'started'/'mock-abc-1.json',{})
    def send(*a):raise AssertionError('must not retry uncertain attempt')
    assert execute(task,CFG,'http://mock','secret',tmp_path,'h',send)=='uncertain_attempt_requires_reconciliation'


def test_crash_recovery_no_call(task,result,tmp_path):
    def send(*a):return response(result)
    execute(task,CFG,'http://mock','secret',tmp_path,'h',send,lambda n:None)
    next((tmp_path/'accepted').glob('*.json')).unlink()
    def forbidden(*a):raise AssertionError('must recover, not rejudge')
    assert execute(task,CFG,'http://mock','secret',tmp_path,'h',forbidden)=='accepted'


def test_http429_retry(task,result,tmp_path):
    import io
    calls=[]
    def send(*a):
        calls.append(1)
        if len(calls)==1:raise urllib.error.HTTPError('http://mock',429,'limited',{},io.BytesIO(b'limited'))
        return response(result)
    assert execute(task,CFG,'http://mock','secret',tmp_path,'h',send,lambda n:None)=='accepted'
    assert len(calls)==2


def test_prompt_fallback():
    import prepare
    from historical_evidence_recovery import evidence_for_baseline
    templates={'ask_expander':'E={{enrichment_block}} Q={{query}}','ask_no_expander':'T={{context_text}} G={{context_graph}} Q={{query}}'}
    for enrichment in ('','No essential knowledge graph enrichment found.'):
        case={'query':'raw Q','baselines':{'B6':dict(status='success',trimmed_text='T',trimmed_graph='G',enrichment_block=enrichment)}}
        branch,prompt=prepare.render(case,'B5',templates)
        assert branch=='ask_no_expander' and prompt=='T=T G=G Q=raw Q'
        assert evidence_for_baseline(case,'B6')[1]==['trimmed_text','trimmed_graph']
    case['baselines']['B6']['enrichment_block']='only E'
    assert prepare.render(case,'B5',templates)==('ask_expander','E=only E Q=raw Q')


def test_mixed_citations(task,result):
    other=copy.deepcopy(result['citations'][0]);other['support']='unsupported';result['citations'].append(other)
    validate(result,task);assert scores(result)['CF']==.5 and scores(result)['coverage']==1


def test_hash_serialization_stable():
    data=rows(HOME/'evidence_manifest.jsonl')
    serialized=b''.join(encode(r)+b'\n' for r in data)
    for _ in range(3):assert serialized==(HOME/'evidence_manifest.jsonl').read_bytes()


def test_same_id_analysis_and_cluster_bootstrap(tmp_path,monkeypatch):
    """Synthetic, isolated data, never written to scientific results directory."""
    import numpy as np
    spec=importlib.util.spec_from_file_location('stats_mock',HOME/'statistical_analysis.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    tasks=[]
    for q in ('q1','q2','q3'):
        for p in ('B3','B5'):
            for j in ('omniroute','openrouter'):
                t=dict(task_id=q+p+j,query_id=q,paper_configuration=p,judge=j,status='ready',visible={})
                tasks.append(t)
                if (q,p,j)==('q3','B5','openrouter'):continue
                sc=dict(F=.8 if p=='B3' else .3,CF=1.,CF_no_citation_na=1.,CF_resolved_only=1.,coverage=1.,resolvability=1.,compliance=1.,n_citations=1,n_resolved=1,n_incorrect=0,n_unresolved=0,n_ambiguous=0,F_status='scored',CF_status='scored')
                path=tmp_path/'accepted'/(t['task_id']+'.json');path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(dict(task=t,parsed={},scores=sc)))
    (tmp_path/'judge_tasks.jsonl').write_bytes(b''.join(encode(t)+b'\n' for t in tasks))
    (tmp_path/'evidence_manifest.jsonl').write_text('{"generation_status":"success"}\n')
    monkeypatch.setattr(m,'verify_freeze',lambda h:'hash');monkeypatch.setattr(m,'validate_checkpoint',lambda *a:None)
    monkeypatch.setattr(m,'body_for',lambda *a:{});monkeypatch.setattr(m,'scores',lambda p:{})
    # Retain score assertion by returning parsed numeric aggregate from local fake.
    for path in (tmp_path/'accepted').glob('*.json'):
        d=json.loads(path.read_text());d['parsed']=d['scores'];path.write_text(json.dumps(d))
    monkeypatch.setattr(m,'scores',lambda p:p)
    result=m.analyze(tmp_path,nboot=30)
    metric=result['metrics']['F']
    assert metric['common_ids']==['q1','q2']
    assert metric['judge_available_ids']['omniroute']==['q1','q2','q3']
    assert metric['contrasts']['primary_common']['omniroute']['difference']==pytest.approx(.5)
    assert metric['contrasts']['primary_common']['openrouter']['n']==2
    assert result['accepted']==11 and result['planned']==12
    assert result['status']=='PENDING RESULTS'
