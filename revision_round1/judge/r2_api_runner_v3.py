#!/usr/bin/env python3
"""Minimal paired R2 v3 API runner for OmniRoute and OpenRouter."""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, tempfile, time, urllib.error, urllib.request
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE)); import r2_runner_v2 as frozen
MODELS={'omniroute':{'model':'cx/gpt-6-luna-low','aliases':{'cx/gpt-6-luna-low','gpt-6-luna'},'rpm':20,'base_env':'REVISION_JUDGE_BASE_URL','key_env':'REVISION_JUDGE_API_KEY'},'openrouter':{'model':'deepseek/deepseek-v4.1-flash-1017','aliases':{'deepseek/deepseek-v4.1-flash-1017'},'rpm':15,'base_url':'https://openrouter.ai/api/v1','key_env':'OPENROUTER_API_KEY'}}
RUBRIC=HERE/'r2_shared_rubric_v3.md'; OUT=ROOT/'revision_round1/experiments/r2_api_v3';SEED=20261106

def canon(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def sha(x):return hashlib.sha256(x).hexdigest()
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fd,n=tempfile.mkstemp(dir=p.parent,prefix='.tmp-');
 try:
  with os.fdopen(fd,'wb') as f:f.write(canon(x)+b'\n');f.flush();os.fsync(f.fileno())
  os.replace(n,p)
 finally:
  if os.path.exists(n):os.unlink(n)
def envkey(name):
 if os.environ.get(name):return os.environ[name]
 ep=ROOT/'.env'
 if ep.exists():
  for line in ep.read_text().splitlines():
   if '=' in line and not line.lstrip().startswith('#'):
    k,v=line.removeprefix('export ').split('=',1)
    if k.strip()==name:return v.strip().strip('"\'')
 return None
def cohort():
 rows=frozen.packet();return rows

def task(row,b):
 x=row['judge_inputs'][b];eid=sha(canon([row['query_id'],b]))[:24]
 return {'evaluation_id':eid,'query_id':row['query_id'],'configuration_label':'evaluation_'+eid,'question_type':'answerable' if row['answerable'] else 'designated_negative','question':row['query'],'reference_answer':row['reference_answer'],'generated_answer':x['candidate_answer'],'generator_visible_evidence':x['generator_visible_evidence'],'evidence_ids':sorted({s for s in x['citation_metadata'] if s and s in x['generator_visible_evidence']}),'input_sha256':sha(canon([row['query_id'],b,row['query'],row['reference_answer'],x['candidate_answer'],x['generator_visible_evidence']])), 'evidence_sha256':x['evidence_sha256']}
def tasks_for(rows,pilot=False):
 ids=[r['query_id'] for r in rows if r['answerable']];neg=[r['query_id'] for r in rows if not r['answerable']];rng=random.Random(SEED);rng.shuffle(ids);rng.shuffle(neg);chosen=set(ids[:8]+neg[:2]) if pilot else None
 return [task(r,b) for r in rows for b in ('B3','B5') if chosen is None or r['query_id'] in chosen]
def validate(o,t):
 required={'evaluation_id','question_type','generation_status','response_behavior','semantic_accuracy','negative_safety','faithfulness','faithfulness_status','answer_relevance','potential_label_conflict','semantic_accuracy_rationale','faithfulness_rationale','negative_safety_rationale','evidence_ids'}
 if set(o)!=required:raise ValueError('normalized result fields mismatch')
 if o['evaluation_id']!=t['evaluation_id'] or o['question_type']!=t['question_type'] or o['generation_status']!='success':raise ValueError('identity/status mismatch')
 for k in ('semantic_accuracy','negative_safety','faithfulness','answer_relevance'):
  v=o[k]
  if v is not None and (isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1):raise ValueError('score out of range: '+k)
 if t['question_type']=='answerable' and o['negative_safety'] is not None:raise ValueError('negative score on answerable')
 if t['question_type']=='designated_negative' and o['semantic_accuracy'] is not None:raise ValueError('semantic score on negative')
 if type(o['potential_label_conflict']) is not bool:raise ValueError('invalid label conflict flag')
 if o['potential_label_conflict'] and t['question_type']!='designated_negative':raise ValueError('label conflict on answerable')
 if o['response_behavior'] not in ('answered','abstained','mixed'):raise ValueError('invalid response behavior')
 if t['question_type']=='answerable' and o['semantic_accuracy'] is None:raise ValueError('missing semantic score')
 if t['question_type']=='designated_negative' and not o['potential_label_conflict'] and o['negative_safety'] not in (0,0.5,1):raise ValueError('invalid negative safety score')
 if o['faithfulness_status']=='scored' and o['faithfulness'] is None:raise ValueError('missing faithfulness score')
 if o['faithfulness_status']=='not_applicable_no_factual_claims' and o['faithfulness'] is not None:raise ValueError('claimless score must be null')
 if o['faithfulness_status'] not in ('scored','not_applicable_no_factual_claims'):raise ValueError('invalid faithfulness status')
 for score,reason in [('semantic_accuracy','semantic_accuracy_rationale'),('negative_safety','negative_safety_rationale'),('faithfulness','faithfulness_rationale')]:
  if o[score] is not None and (not isinstance(o[reason],str) or not o[reason].strip()):raise ValueError('missing rationale for '+score)
 if not isinstance(o['evidence_ids'],list) or not set(o['evidence_ids'])<=set(t['evidence_ids']):raise ValueError('invalid evidence IDs')
 for k in ('semantic_accuracy_rationale','faithfulness_rationale','negative_safety_rationale'):
  if o[k] is not None and not isinstance(o[k],str):raise ValueError('invalid rationale')

def messages(t):
 visible={k:v for k,v in t.items() if k not in ('query_id','configuration_label','input_sha256','evidence_sha256')}
 return [{'role':'system','content':RUBRIC.read_text()},{'role':'user','content':json.dumps(visible,ensure_ascii=False,sort_keys=True)+"\nReturn ONLY the required normalized JSON object."}]
def request(cfg,key,t):
 base=cfg.get('base_url') or cfg['base_url'];body={'model':cfg['model'],'temperature':0,'max_tokens':1400,'response_format':{'type':'json_object'},'messages':messages(t)}
 req=urllib.request.Request(base.rstrip('/')+'/chat/completions',data=canon(body),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','X-Title':'KNOSYS R2 v3'})
 start=time.monotonic()
 with urllib.request.urlopen(req,timeout=180) as r:
  payload=json.loads(r.read());payload['_meta']={'requested_model':cfg['model'],'returned_model':payload.get('model'),'provider':cfg.get('provider'),'latency_seconds':time.monotonic()-start,'headers':{k:v for k,v in r.headers.items() if k.lower().startswith(('x-','ratelimit','request'))}}
 return payload

def run(authorize=False,pilot_only=False,attempt_limit=4,token_budget=10000000):
 rows=cohort();rubric_hash=sha(RUBRIC.read_bytes());all_tasks=tasks_for(rows,pilot_only);manifest={'schema':'r2-api-v3','rubric_sha256':rubric_hash,'cohort_ids_sha256':sha(canon(sorted(r['query_id'] for r in rows))),'task_count':len(tasks_for(rows,False)),'pilot_task_count':len(tasks_for(rows,True)),'seed':SEED,'models':{k:v['model'] for k,v in MODELS.items()}}
 if not authorize:return {'dry_run':True,'manifest':manifest,'tasks':len(all_tasks)*2}
 # All credentials must be available before the first potentially paid call.
 missing=[v['key_env'] for v in MODELS.values() if not envkey(v['key_env'])]
 if missing:raise RuntimeError('missing credentials: '+', '.join(missing))
 if not envkey('REVISION_JUDGE_BASE_URL'):raise RuntimeError('OmniRoute base URL missing')
 # Execution remains disabled until token/call budgets and pilot QC are implemented.
 raise RuntimeError('v3 runner is incomplete: budget and pilot quality gates are not implemented; no requests sent')
 out=OUT;out.mkdir(parents=True,exist_ok=True);mp=out/'manifest.json'
 if mp.exists() and json.loads(mp.read_text())!=manifest:raise RuntimeError('existing manifest mismatch; refusing to mix')
 if not mp.exists():atomic(mp,manifest)
 if not pilot_only:raise RuntimeError('full run requires implemented and reviewed pilot gating; run pilot-only first')
 total=0
 for provider,cfg0 in MODELS.items():
  cfg=dict(cfg0);cfg['provider']=provider;key=envkey(cfg['key_env']);
  if not key:raise RuntimeError(cfg['key_env']+' missing')
  base=cfg.get('base_url') or envkey(cfg['base_env']);
  if not base:raise RuntimeError('OmniRoute base URL missing')
  cfg['base_url']=base; d=out/provider; last=0
  for t in all_tasks:
   p=d/(t['evaluation_id']+'.json');raw=d/'raw'/(t['evaluation_id']+'.json')
   if p.exists():
    saved=json.loads(p.read_text())
    if saved.get('rubric_sha256')!=rubric_hash or saved.get('task',{}).get('input_sha256')!=t['input_sha256'] or saved.get('requested_model')!=cfg['model']:raise RuntimeError('checkpoint mismatch')
    continue
   uncertain=d/'uncertain'/(t['evaluation_id']+'.json')
   if uncertain.exists():raise RuntimeError('previous request requires reconciliation: '+t['evaluation_id'])
   for attempt in range(attempt_limit):
    wait=60/cfg['rpm']-(time.time()-last)
    if wait>0:time.sleep(wait)
    last=time.time();total+=1
    try:
     atomic(uncertain,{'state':'started','attempt':attempt+1,'time':time.time()})
     payload=request(cfg,key,t);atomic(raw,{'attempt':attempt+1,'task':t,'response':payload})
     ret=payload.get('model')
     if ret not in cfg['aliases']:raise ValueError('unexpected returned model '+str(ret))
     content=payload['choices'][0]['message']['content'];o=json.loads(content);validate(o,t)
     atomic(p,{'provider':provider,'requested_model':cfg['model'],'returned_model':ret,'task':t,'result':o,'usage':payload.get('usage'),'meta':payload.get('_meta'),'rubric_sha256':rubric_hash});uncertain.unlink();break
    except urllib.error.HTTPError as e:
     detail=e.read().decode('utf8','replace')[:1000]
     if e.code==429 and attempt<attempt_limit-1:uncertain.unlink();time.sleep(min(120,2**attempt*30));continue
     atomic(d/'errors'/f'{t["evaluation_id"]}.json',{'status':e.code,'detail':detail});raise
    except (json.JSONDecodeError,KeyError,ValueError) as e:
     # A completed but invalid response needs manual review; never silently retry it.
     if False:continue
     atomic(d/'errors'/f'{t["evaluation_id"]}.json',{'error':str(e)});raise
   else:raise RuntimeError('retry limit exhausted')
 return {'completed_requests':total,'task_slots':len(all_tasks)*2,'output':str(OUT)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--authorize-judge-execution',action='store_true');p.add_argument('--pilot-only',action='store_true');p.add_argument('--attempt-limit',type=int,default=4);a=p.parse_args();print(json.dumps(run(a.authorize_judge_execution,a.pilot_only,a.attempt_limit),indent=2))
if __name__=='__main__':main()
