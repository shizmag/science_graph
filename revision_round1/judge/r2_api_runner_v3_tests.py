import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent));import r2_api_runner_v3 as r

def test_task_types_and_pilot():
 ts=r.tasks_for(r.cohort(),True);assert len(ts)==20;assert sum(x['question_type']=='designated_negative' for x in ts)==4;assert all(x['evidence_ids'] for x in ts)
def test_answerable_abstention_is_still_scoreable():
 t={'evaluation_id':'x','question_type':'answerable','evidence_ids':['s']};o={'evaluation_id':'x','question_type':'answerable','generation_status':'success','response_behavior':'abstained','semantic_accuracy':0,'negative_safety':None,'faithfulness':None,'faithfulness_status':'not_applicable_no_factual_claims','answer_relevance':0,'potential_label_conflict':False,'semantic_accuracy_rationale':'wrong refusal','faithfulness_rationale':'no claims','negative_safety_rationale':None,'evidence_ids':[]};r.validate(o,t)
def test_negative_score_and_conflict():
 t={'evaluation_id':'x','question_type':'designated_negative','evidence_ids':['s']}
 o={'evaluation_id':'x','question_type':'designated_negative','generation_status':'success','response_behavior':'abstained','semantic_accuracy':None,'negative_safety':1,'faithfulness':None,'faithfulness_status':'not_applicable_no_factual_claims','answer_relevance':1,'potential_label_conflict':False,'semantic_accuracy_rationale':None,'faithfulness_rationale':'none','negative_safety_rationale':'bounded','evidence_ids':[]};r.validate(o,t)
 o['potential_label_conflict']=True;o['negative_safety']=None;r.validate(o,t)
def test_dry_run():assert r.run()['dry_run']

def test_wrong_negative_semantic_rejected():
 import pytest
 t={'evaluation_id':'x','question_type':'designated_negative','evidence_ids':[]}
 o={'evaluation_id':'x','question_type':'designated_negative','generation_status':'success','response_behavior':'abstained','semantic_accuracy':0,'negative_safety':1,'faithfulness':None,'faithfulness_status':'not_applicable_no_factual_claims','answer_relevance':1,'potential_label_conflict':False,'semantic_accuracy_rationale':'wrong','faithfulness_rationale':'no claims','negative_safety_rationale':'bounded','evidence_ids':[]}
 with pytest.raises(ValueError):r.validate(o,t)

def test_no_network_in_dry_run(monkeypatch):
 def fail(*args,**kwargs):raise AssertionError('network accessed')
 monkeypatch.setattr(r.urllib.request,'urlopen',fail)
 assert r.run(pilot_only=True)['tasks']==40

def test_provider_routes_share_instructions():
 t=r.tasks_for(r.cohort(),True)[0]
 a=r.body_for(dict(r.MODELS['omniroute'],provider='omniroute'),t)
 b=r.body_for(dict(r.MODELS['openrouter'],provider='openrouter'),t)
 assert a['messages']==b['messages'] and b['provider']['allow_fallbacks'] is False
 assert t['query_id'] not in a['messages'][1]['content'] or t['query_id'] in t['generator_visible_evidence']

def test_incorrect_scores_rejected():
 import pytest
 t={'evaluation_id':'x','question_type':'answerable','evidence_ids':[]}
 o={'evaluation_id':'x','question_type':'answerable','generation_status':'success','response_behavior':'answered','semantic_accuracy':2,'negative_safety':None,'faithfulness':1,'faithfulness_status':'scored','answer_relevance':1,'potential_label_conflict':False,'semantic_accuracy_rationale':'x','faithfulness_rationale':'x','negative_safety_rationale':None,'evidence_ids':[]}
 with pytest.raises(ValueError):r.validate(o,t)

def test_fake_pilot_resume_and_gate(tmp_path,monkeypatch):
 import pytest
 monkeypatch.setattr(r,'OUT',tmp_path);monkeypatch.setattr(r.time,'sleep',lambda _:None)
 monkeypatch.setattr(r,'envkey',lambda k:'http://127.0.0.1:20128/v1' if k=='REVISION_JUDGE_BASE_URL' else 'fake-key')
 calls=[]
 def fake(cfg,key,t):
  calls.append((cfg['provider'],t['evaluation_id']));ans=t['question_type']=='answerable'
  o={'evaluation_id':t['evaluation_id'],'question_type':t['question_type'],'generation_status':'success','response_behavior':'answered','semantic_accuracy':1 if ans else None,'negative_safety':None if ans else 0,'faithfulness':1,'faithfulness_status':'scored','answer_relevance':1,'potential_label_conflict':False,'semantic_accuracy_rationale':'fixture' if ans else None,'faithfulness_rationale':'fixture','negative_safety_rationale':None if ans else 'fixture','evidence_ids':[]}
  return {'model':'gpt-6-luna' if cfg['provider']=='omniroute' else cfg['model'],'choices':[{'message':{'content':json.dumps(o)},'finish_reason':'stop'}],'usage':{'prompt_tokens':1,'completion_tokens':1}}
 monkeypatch.setattr(r,'request',fake)
 result=r.run(True,True);assert result['completed']==40 and len(calls)==40
 r.run(True,True);assert len(calls)==40
 r.run(True,False)
 assert len(calls)==400

def test_live_transport_request_is_exact_and_mocked(monkeypatch):
 import io
 seen=[]
 class Response:
  headers={'x-request-id':'test'}
  def __enter__(self): return self
  def __exit__(self,*args): pass
  def read(self): return b'{"model":"gpt-6-luna","choices":[]}'
 def urlopen(req, timeout):
  seen.append((req.full_url, json.loads(req.data), timeout));return Response()
 monkeypatch.setattr(r.urllib.request,'urlopen',urlopen)
 t=r.tasks_for(r.cohort(),True)[0]
 for name in ('omniroute','openrouter'):
  cfg=dict(r.MODELS[name],provider=name,base_url='http://localhost:20128/v1' if name=='omniroute' else 'https://openrouter.ai/api/v1')
  r.request(cfg,'fake-key',t)
  assert seen[-1][1]['model']==cfg['model']
  assert seen[-1][0]==cfg['base_url']+'/chat/completions'
  assert seen[-1][2]==180
 assert seen[1][1]['provider']['allow_fallbacks'] is False
 assert seen[0][1]['messages']==seen[1][1]['messages']

def test_conflict_score_still_requires_scale():
 import pytest
 t={'evaluation_id':'x','question_type':'designated_negative','evidence_ids':[]}
 o={'evaluation_id':'x','question_type':'designated_negative','generation_status':'success','response_behavior':'answered','semantic_accuracy':None,'negative_safety':0.3,'faithfulness':None,'faithfulness_status':'not_applicable_no_factual_claims','answer_relevance':1,'potential_label_conflict':True,'semantic_accuracy_rationale':None,'faithfulness_rationale':'none','negative_safety_rationale':'conflict','evidence_ids':[]}
 with pytest.raises(ValueError): r.validate(o,t)
