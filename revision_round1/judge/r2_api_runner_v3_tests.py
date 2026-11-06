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
