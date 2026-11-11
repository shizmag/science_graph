"""Frozen, resumable R1 dual-judge evaluation boundary."""
from __future__ import annotations
import hashlib,json,os,tempfile,time,urllib.request
from pathlib import Path
from .checkpoint import DurableResultStore
from .prepared import read
RUBRIC=Path(__file__).resolve().parents[2]/"revision_round1/judge/r2_shared_rubric_v3.md"
SCHEMA="r1-judge-input-v1"; RESPONSE="r1-judge-response-v1"
MODELS={"luna":{"provider":"omniroute","requested_model":"cx/gpt-6-luna-low","base_url_env":"REVISION_JUDGE_BASE_URL","key_env":"REVISION_JUDGE_API_KEY"},"deepseek":{"provider":"openrouter","requested_model":"deepseek/deepseek-v4.1-flash-1017","base_url":"https://openrouter.ai/api/v1","key_env":"OPENROUTER_API_KEY"}}
def canon(x): return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()
def sha(x): return hashlib.sha256(x if isinstance(x,bytes) else canon(x)).hexdigest()
def atomic(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=p.parent,prefix='.tmp-')
 with os.fdopen(fd,'w') as f: json.dump(obj,f,sort_keys=True,ensure_ascii=False);f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def _load(p,default): return json.loads(p.read_text()) if p.exists() else default
def prepare(root,manifest_hash):
 root=Path(root); manifest=_load(root/'experiment_manifest.json',{})
 if manifest.get('manifest_hash')!=manifest_hash: raise ValueError('manifest hash mismatch')
 cohort=_load(root/'cohort_manifest.json',{}); rows={str(x['query_id']):x for x in cohort.get('rows',[])}
 store=DurableResultStore(root,root/'experiment_manifest.json',manifest_hash); accepted=store.accepted_rows(); out=[]; private={}
 for row in accepted:
  q,a=row['query_id'],row['arm_id']; p=root/'prepared_inputs'/f'{q}__{a}.json'; item=read(p)
  if item.manifest_hash!=manifest_hash or item.input_hash!=row['input_identity']: raise ValueError(f'prepared identity mismatch: {q}/{a}')
  cr=rows.get(q)
  if cr is None: raise ValueError(f'accepted query is outside cohort: {q}')
  eid=sha([manifest_hash,q,a])[:24]; result=row['result']
  rec={'evaluation_id':eid,'question':cr.get('question',result.get('question','')),'reference_answer':cr.get('reference_answer'),'generated_answer':result.get('answer',result.get('generated_answer','')),'generator_visible_evidence':item.prompt,'evidence_ids':list(item.evidence_ids),'answerable':bool(cr.get('answerable',True)),'rubric_sha256':sha(RUBRIC.read_bytes()),'generation_result_sha256':sha(result),'prepared_input_sha256':item.input_hash,'manifest_hash':manifest_hash,'schema':SCHEMA}
  rec['input_sha256']=sha(rec);out.append(rec);private[eid]={'query_id':q,'arm_id':a}
 out.sort(key=lambda x:x['evaluation_id']); atomic(root/'judge_inputs.json',{'schema':SCHEMA,'manifest_hash':manifest_hash,'rubric_sha256':sha(RUBRIC.read_bytes()),'inputs':out}); atomic(root/'judge_private_mapping.json',private); return len(out)
def _validate(o,task):
 if o.get('evaluation_id')!=task['evaluation_id']: raise ValueError('evaluation identity mismatch')
 for k in ('semantic_accuracy','faithfulness','answer_relevance','negative_safety'):
  if o.get(k) is not None and (not isinstance(o[k],(int,float)) or not 0<=o[k]<=1): raise ValueError('invalid score '+k)
 if not task['answerable'] and o.get('semantic_accuracy') is not None: raise ValueError('negative semantic accuracy must be null')
 return o
def run(root,judge,max_attempts=2,stop_after=None):
 root=Path(root); spec=MODELS[judge]; packet=_load(root/'judge_inputs.json',{}); tasks=packet.get('inputs',[]); ledger=root/f'judge_{judge}.json'; data=_load(ledger,{'schema':RESPONSE,'judge':judge,'accepted':{},'attempts':[]}); done=0
 for task in tasks:
  if task['evaluation_id'] in data['accepted']: continue
  if stop_after is not None and done>=stop_after: break
  for attempt in range(1,max_attempts+1):
   try:
    base=os.environ.get(spec.get('base_url_env',''),spec.get('base_url',''))
    if not base: raise RuntimeError('judge base URL is not configured')
    req={'model':spec['requested_model'],'messages':[{'role':'system','content':'Evaluate only the supplied materials using the frozen rubric.'},{'role':'user','content':json.dumps(task,ensure_ascii=False)}]}
    r=urllib.request.urlopen(urllib.request.Request(base.rstrip('/')+'/chat/completions',data=canon(req),headers={'Content-Type':'application/json','Authorization':'Bearer '+os.environ.get(spec['key_env'],'')}),timeout=60); raw=json.loads(r.read()); content=raw['choices'][0]['message']['content']; parsed=json.loads(content) if isinstance(content,str) else content; parsed['evaluation_id']=task['evaluation_id']; parsed['returned_model']=raw.get('model'); parsed['provider']=spec['provider']; _validate(parsed,task); data['accepted'][task['evaluation_id']]={'task':task,'response':parsed,'raw_response':raw,'completed_at':time.time()};data['attempts'].append({'evaluation_id':task['evaluation_id'],'attempt':attempt,'status':'accepted'});atomic(ledger,data);done+=1; print(f"R1 | LLM-AS-A-JUDGE | {judge} | Accepted: {len(data['accepted'])}/{len(tasks)} | Last evaluation: {task['evaluation_id']} | Checkpoint: saved",flush=True);break
   except Exception as exc:
    data['attempts'].append({'evaluation_id':task['evaluation_id'],'attempt':attempt,'status':'retryable' if attempt<max_attempts else 'terminal','error':str(exc)});atomic(ledger,data)
    if attempt==max_attempts: break
 return len(data['accepted'])
def status(root,judge):
 d=_load(Path(root)/f'judge_{judge}.json',{'accepted':{},'attempts':[]});return {'judge':judge,'accepted':len(d['accepted']),'attempts':len(d['attempts'])}
def export(root,judge,out):
 d=_load(Path(root)/f'judge_{judge}.json',{}); rows=[x['response'] for x in d.get('accepted',{}).values()]; atomic(out,rows);return len(rows)
