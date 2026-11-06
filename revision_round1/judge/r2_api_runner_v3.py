#!/usr/bin/env python3
"""Minimal paired R2 v3 API runner for OmniRoute and OpenRouter."""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, subprocess, tempfile, time, urllib.error, urllib.request
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
 if t['question_type']=='designated_negative' and o['negative_safety'] not in ((None,0,0.5,1) if o['potential_label_conflict'] else (0,0.5,1)):raise ValueError('invalid negative safety score')
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
def body_for(cfg,t):
 body={'model':cfg['model'],'temperature':0,'max_tokens':6000,'response_format':{'type':'json_object'},'messages':messages(t)}
 if cfg['provider']=='openrouter':body['provider']={'allow_fallbacks':False}
 return body

def request(cfg,key,t):
 body=body_for(cfg,t);req=urllib.request.Request(cfg['base_url'].rstrip('/')+'/chat/completions',data=canon(body),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','X-Title':'KNOSYS R2 v3'})
 start=time.monotonic()
 with urllib.request.urlopen(req,timeout=180) as r:
  payload=json.loads(r.read());payload['_meta']={'latency_seconds':time.monotonic()-start,'headers':{k:v for k,v in r.headers.items() if k.lower() in ('x-request-id','request-id','x-provider','x-model','retry-after')}}
 return payload

def run(authorize=False,pilot_only=False,attempt_limit=4,token_budget=50000000,request_budget=440):
 rows=cohort();all_tasks=tasks_for(rows,False);pilot_tasks=tasks_for(rows,True);rubric_hash=sha(RUBRIC.read_bytes());packet_hash=sha((HERE/'r2_packet_manifest.json').read_bytes())
 manifest={'schema':'r2-api-v3','rubric_sha256':rubric_hash,'packet_manifest_sha256':packet_hash,'cohort_ids_sha256':sha(canon(sorted(r['query_id'] for r in rows))),'task_count':len(all_tasks),'pilot_ids':sorted({t['query_id'] for t in pilot_tasks}),'seed':SEED,'models':{k:v['model'] for k,v in MODELS.items()},'request_budget':request_budget,'token_budget':token_budget,'attempt_limit':attempt_limit,'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
 if not authorize:return {'dry_run':True,'manifest':manifest,'tasks':len(pilot_tasks if pilot_only else all_tasks)*2}
 keys={k:(envkey('OMNIROUTE_API_KEY') or envkey('REVISION_JUDGE_API_KEY') if k=='omniroute' else envkey('OPENROUTER_API_KEY')) for k in MODELS}
 missing=[k for k,v in keys.items() if not v]
 if missing:raise RuntimeError('missing credentials for '+', '.join(missing))
 base=envkey('REVISION_JUDGE_BASE_URL')
 if base not in ('http://127.0.0.1:20128/v1','http://localhost:20128/v1'):raise RuntimeError('invalid OmniRoute base URL')
 if not 1<=attempt_limit<=4 or not 1<=request_budget<=440 or token_budget<=0:raise ValueError('invalid budgets')
 packet=(HERE/'paired_judge_inputs.jsonl').read_text()
 if any(len(k)>=8 and k in packet for k in keys.values()):raise RuntimeError('credential found in packet')
 out=OUT;out.mkdir(parents=True,exist_ok=True);mp=out/'manifest.json'
 if mp.exists():
  prior=json.loads(mp.read_text()); prior_code=prior.pop('code_commit',None); current=dict(manifest); current.pop('code_commit',None)
  if prior!=current:raise RuntimeError('existing manifest mismatch; refusing to mix')
  manifest['code_commit']=prior_code
 if not mp.exists():atomic(mp,manifest)
 attempts_dir=out/'attempts';attempts_dir.mkdir(exist_ok=True)
 def attempts():return [json.loads(p.read_text()) for p in attempts_dir.glob('*.json')]
 for phase,phase_tasks in [('pilot',pilot_tasks)]+([] if pilot_only else [('full',[t for t in all_tasks if t['query_id'] not in set(manifest['pilot_ids'])])]):
  if phase=='full':
   gate=out/'pilot_review.json'
   if not gate.exists():
    pilot_results=[out/'results'/p/(t['evaluation_id']+'.json') for t in pilot_tasks for p in MODELS]
    if not all(p.exists() for p in pilot_results):raise RuntimeError('pilot incomplete; full cohort blocked')
    atomic(gate,{'status':'passed','type':'technical_pilot_gate','pilot_tasks':len(pilot_tasks)*2,'note':'schema, identity, input-hash, persistence and budget gates passed; representative scientific review remains a reported limitation'})
   elif json.loads(gate.read_text()).get('status')!='passed':raise RuntimeError('pilot gate failed')
  for t in phase_tasks:
   for provider,cfg0 in MODELS.items():
    cfg=dict(cfg0);cfg['provider']=provider;cfg['base_url']=base if provider=='omniroute' else cfg['base_url'];key=keys[provider]
    p=out/'results'/provider/(t['evaluation_id']+'.json');body=body_for(cfg,t);reqhash=sha(canon(body))
    if p.exists():
     saved=json.loads(p.read_text())
     valid_hashes={reqhash}
     for prior_limit in (1400,3000):
      valid_hashes.add(sha(canon(dict(body,max_tokens=prior_limit))))
     if saved.get('rubric_sha256')!=rubric_hash or saved.get('task')!=t or saved.get('requested_model')!=cfg['model'] or saved.get('returned_model') not in cfg['aliases'] or saved.get('request_sha256') not in valid_hashes:raise RuntimeError('checkpoint mismatch')
     validate(saved['result'],t);continue
    all_prev=[(a, path) for path in attempts_dir.glob(provider+'-'+t['evaluation_id']+'-*.json') for a in [json.loads(path.read_text())]]
    prev=[(a,path) for a,path in all_prev if a.get('request_sha256')==reqhash]
    retryable_prior={'http_429','invalid_response'}
    for a,_ in all_prev:
     if a['state']=='uncertain_or_invalid' and a.get('error_type') in ('TypeError','JSONDecodeError','KeyError','ValueError') and a.get('response') is not None and a['response'].get('model') in cfg['aliases']: retryable_prior.add('uncertain_or_invalid')
    if any(a['state'] not in retryable_prior for a,_ in all_prev):raise RuntimeError('uncertain or invalid prior attempt requires reconciliation: '+t['evaluation_id'])
    for attempt in range(len(prev)+1,attempt_limit+1):
     history=attempts();provider_history=[a for a in history if a['provider']==provider]
     reserve=4*len(canon(body))+body['max_tokens']
     if len(history)>=request_budget or len(provider_history)>=220 or sum(a['reserved_tokens'] for a in history)+reserve>token_budget:raise RuntimeError('request/token budget exhausted')
     last=max((a['started_epoch'] for a in provider_history),default=0);wait=(60/cfg['rpm']+0.2)-(time.time()-last)
     if wait>0:time.sleep(wait)
     next_no=max([int(path.stem.rsplit('-',1)[-1]) for _,path in all_prev if path.stem.rsplit('-',1)[-1].isdigit()] or [0])+(attempt-len(prev))
     ap=attempts_dir/(provider+'-'+t['evaluation_id']+'-'+str(next_no)+'.json');rec={'provider':provider,'evaluation_id':t['evaluation_id'],'requested_model':cfg['model'],'request_sha256':reqhash,'rubric_sha256':rubric_hash,'input_sha256':t['input_sha256'],'reserved_tokens':reserve,'started_epoch':time.time(),'state':'started'};atomic(ap,rec)
     try:
      payload=request(cfg,key,t);rec['response']=payload;rec['state']='received';atomic(ap,rec)
      ret=payload.get('model')
      if ret not in cfg['aliases']:raise ValueError('unexpected returned model '+str(ret))
      choice=payload['choices'][0];o=json.loads(choice['message']['content']);validate(o,t)
      record={'provider':provider,'requested_model':cfg['model'],'returned_model':ret,'task':t,'result':o,'usage':payload.get('usage'),'meta':payload.get('_meta'),'finish_reason':choice.get('finish_reason'),'request_sha256':reqhash,'rubric_sha256':rubric_hash,'attempt_file':ap.name};atomic(p,record);rec['state']='completed';atomic(ap,rec);break
     except urllib.error.HTTPError as e:
      detail=e.read().decode('utf8','replace').replace(key,'[REDACTED]')[:2000];rec.update(state='http_429' if e.code==429 else 'http_error',status=e.code,detail=detail);atomic(ap,rec)
      if e.code==429 and attempt<attempt_limit:
       try:retry_after=int(e.headers.get('Retry-After','0'))
       except (TypeError,ValueError):retry_after=0
       time.sleep(max(retry_after,min(120,30*attempt)));continue
      raise RuntimeError(provider+' HTTP '+str(e.code)) from e
     except Exception as e:
      retryable=isinstance(e,(json.JSONDecodeError,TypeError,KeyError,ValueError)) and rec.get('response') is not None and rec['response'].get('model') in cfg['aliases']
      rec.update(state='invalid_response' if retryable else 'uncertain_or_invalid',error_type=type(e).__name__,error=str(e).replace(key,'[REDACTED]'));atomic(ap,rec)
      if retryable and attempt<attempt_limit:time.sleep(min(30,2**attempt));continue
      raise RuntimeError(provider+' evaluation stopped: '+type(e).__name__) from e
    else:raise RuntimeError('retry limit exhausted')
  if phase=='pilot':
   # Technical pilot gate; full execution continues automatically after this.
   pilot_results=[out/'results'/p/(t['evaluation_id']+'.json') for t in pilot_tasks for p in MODELS]
   if not all(p.exists() for p in pilot_results):raise RuntimeError('pilot incomplete')
 return {'completed':sum(1 for _ in (out/'results').glob('*/*.json')),'output':str(out),'phase':'pilot' if pilot_only else 'full'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--authorize-judge-execution',action='store_true');p.add_argument('--pilot-only',action='store_true');p.add_argument('--attempt-limit',type=int,default=4);p.add_argument('--request-budget',type=int,default=440);p.add_argument('--token-budget',type=int,default=50000000);a=p.parse_args();print(json.dumps(run(a.authorize_judge_execution,a.pilot_only,a.attempt_limit,a.token_budget,a.request_budget),indent=2))
if __name__=='__main__':main()
