"""Offline query-paired analysis of v3 checkpoints; never uses earlier rubrics."""
from __future__ import annotations
import csv,json,math,sys
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
import analyze_agreement as stats
import r2_api_runner_v3 as run

def finite(x):return float(x) if math.isfinite(float(x)) else None

def weighted_kappa(x,y):
 if len(x)<2:return None
 a=np.array(x);b=np.array(y);levels=np.array([0,.5,1]);w=((levels[:,None]-levels[None,:])**2)
 obs=np.array([[np.mean((a==i)&(b==j)) for j in levels] for i in levels]);exp=np.outer([np.mean(a==i) for i in levels],[np.mean(b==i) for i in levels]);den=float((w*exp).sum())
 return 1-float((w*obs).sum())/den if den else None

def analyze(out=run.OUT):
 out=Path(out);manifest=json.loads((out/'manifest.json').read_text());rh=run.sha(run.RUBRIC.read_bytes())
 if manifest['rubric_sha256']!=rh:raise ValueError('rubric mismatch')
 items=run.tasks_for(run.cohort());expected={t['evaluation_id']:t for t in items};results={};rows=[]
 for provider,cfg in run.MODELS.items():
  for f in (out/'results'/provider).glob('*.json'):
   rec=json.loads(f.read_text());t=expected.get(f.stem)
   if not t or rec['task']['input_sha256']!=t['input_sha256'] or rec['rubric_sha256']!=rh or rec['requested_model']!=cfg['model'] or rec['returned_model'] not in cfg['aliases']:raise ValueError('mismatched checkpoint')
   run.validate(rec['result'],t);results[(provider,f.stem)]=rec['result']
   baseline=next(b for b in ('B3','B5') if run.sha(run.canon([t['query_id'],b]))[:24]==f.stem)
   rows.append({'provider':provider,'query_id':t['query_id'],'baseline':baseline,'evaluation_id':f.stem,'input_sha256':t['input_sha256'],**rec['result']})
 summary={'rubric_sha256':rh,'expected_per_judge':len(items),'judges':{},'agreement':{}}
 for p in run.MODELS:
  pr=[x for x in rows if x['provider']==p];by=defaultdict(dict)
  for x in pr:by[x['query_id']][x['baseline']]=x
  info={'completed':len(pr),'missing':len(items)-len(pr),'potential_label_conflicts':sum(x['potential_label_conflict'] for x in pr),'claimless_abstentions':sum(x['faithfulness_status']=='not_applicable_no_factual_claims' for x in pr),'contrasts':{}}
  for typ in ('answerable','designated_negative'):
   for metric in (('semantic_accuracy','faithfulness','answer_relevance') if typ=='answerable' else ('negative_safety','faithfulness','answer_relevance')):
    pairs=[(v['B3'][metric],v['B5'][metric]) for v in by.values() if 'B3'in v and 'B5'in v and v['B3']['question_type']==typ and v['B3'][metric] is not None and v['B5'][metric] is not None]
    diff=[a-b for a,b in pairs];ci=stats.bootstrap_ci(diff,np.mean,20261106)
    info['contrasts'][typ+'_'+metric]={'paired_n':len(pairs),'b3_mean':float(np.mean([x[0] for x in pairs])) if pairs else None,'b5_mean':float(np.mean([x[1] for x in pairs])) if pairs else None,'difference':float(np.mean(diff)) if diff else None,'ci95':[finite(c) for c in ci]}
  summary['judges'][p]=info
 for typ in ('answerable','designated_negative'):
  for metric in (('semantic_accuracy','faithfulness','answer_relevance') if typ=='answerable' else ('negative_safety','faithfulness','answer_relevance')):
   pairs=[(results[('omniroute',eid)][metric],results[('openrouter',eid)][metric],eid) for eid,t in expected.items() if t['question_type']==typ and ('omniroute',eid) in results and ('openrouter',eid) in results and results[('omniroute',eid)][metric] is not None and results[('openrouter',eid)][metric] is not None]
   x=np.array([a for a,b,e in pairs]);y=np.array([b for a,b,e in pairs]);sp=stats.safe_spearman(x,y) if len(x) else math.nan;icc=stats.icc_a1(x,y) if len(x) else math.nan
   summary['agreement'][typ+'_'+metric]={'matched_n':len(pairs),'matched_queries':len({expected[e]['query_id'] for a,b,e in pairs}),'mean_absolute_difference':float(np.mean(abs(x-y))) if len(x) else None,'spearman':finite(sp),'icc_absolute':finite(icc),'weighted_kappa':weighted_kappa(x,y) if metric=='negative_safety' else None,'largest_disagreements':[{'evaluation_id':e,'query_id':expected[e]['query_id'],'luna':a,'deepseek':b,'absolute_difference':abs(a-b)} for a,b,e in sorted(pairs,key=lambda z:abs(z[0]-z[1]),reverse=True)[:10]]}
 out.joinpath('analysis').mkdir(exist_ok=True);run.atomic(out/'analysis'/'summary.json',summary)
 if rows:
  with (out/'analysis'/'scores.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 return summary
if __name__=='__main__':print(json.dumps(analyze(),indent=2))
