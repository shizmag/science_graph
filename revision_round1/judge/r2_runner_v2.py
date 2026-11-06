"""Narrow, fail-closed runner for the recovered R2 v2 packet (not a general evaluator)."""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, re, tempfile, time, urllib.error, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = ('cx/gpt-6-luna-low', 'dva/deepseek-v4-flash-low')
BASE_URL = 'http://127.0.0.1:20128/v1'
RUBRIC = HERE / 'r2_shared_rubric_v2.md'
RETURNED_MODELS = {MODELS[0]: 'gpt-6-luna', MODELS[1]: 'deepseek-v4-flash'}

def digest(b): return hashlib.sha256(b).hexdigest()
def encode(x): return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
def atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.tmp-')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(encode(obj)); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def packet():
    manifest = json.loads((HERE/'r2_packet_manifest.json').read_text())
    for name, item in manifest['files'].items():
        p = HERE/name; b = p.read_bytes()
        if len(b) != item['size_bytes'] or digest(b) != item['sha256']: raise ValueError('packet checksum mismatch: '+name)
    frozen = json.loads((HERE/'paired_judge_query_ids.json').read_text())['query_ids']
    rows = [json.loads(s) for s in (HERE/'paired_judge_inputs.jsonl').read_text().splitlines()]
    ids = [r['query_id'] for r in rows]
    if len(ids) != 100 or len(set(ids)) != 100 or len(set(frozen)) != 100 or set(ids) != set(frozen): raise ValueError('cohort mismatch')
    if sum(r['answerable'] is True for r in rows) != 80: raise ValueError('cohort composition mismatch')
    for r in rows:
        if r['answerable'] and (r['semantic_accuracy_status'] != 'eligible_with_reference' or not r['reference_answer'].strip()): raise ValueError('invalid semantic reference')
        for b, code in [('B3','B4'),('B5','B6')]:
            x = r['judge_inputs'][b]; ev = x['generator_visible_evidence']
            if x['code_baseline'] != code or r['evidence_status'][b] not in ('exact_archived','verified_reconstructed') or not ev.strip() or digest(ev.encode()) != x['evidence_sha256'] or not x['candidate_answer'].strip(): raise ValueError('invalid evidence or answer: '+r['query_id']+' '+b)
    return rows

def pilot_ids(rows):
    rng = random.Random(20261106)
    return sorted(rng.sample(sorted(r['query_id'] for r in rows if r['answerable']),8) + rng.sample(sorted(r['query_id'] for r in rows if not r['answerable']),2))

def payload(row, baseline, model):
    x = row['judge_inputs'][baseline]
    data = {'query_id':row['query_id'], 'question':row['query'], 'reference_answer':row['reference_answer'], 'answerable':row['answerable'], 'candidate_answer':x['candidate_answer'], 'generator_visible_evidence':x['generator_visible_evidence']}
    return {'model':model,'temperature':0,'max_tokens':1024,'response_format':{'type':'json_object'},'messages':[{'role':'system','content':RUBRIC.read_text()},{'role':'user','content':json.dumps(data,ensure_ascii=False,sort_keys=True)}]}

def parse(content, answerable):
    o=json.loads(content)
    if not isinstance(o,dict) or o.get('status') not in ('ok','valid') or not isinstance(o.get('rationale'),str) or not o['rationale'].strip(): raise ValueError('invalid status or rationale')
    for name in ('semantic_accuracy','faithfulness'):
        v=o.get(name)
        if name=='semantic_accuracy' and not answerable:
            if v is not None: raise ValueError('negative semantic score must be null')
        elif isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1: raise ValueError('invalid '+name)
    return o

def transport(path, key, body=None):
    req=urllib.request.Request(BASE_URL+path,data=encode(body) if body is not None else None,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=120) as r:
        response=json.loads(r.read())
        if body is not None:
            # Keep only routing/request metadata; never persist cookies or authorization headers.
            response['_gateway_headers']={k:v for k,v in r.headers.items() if k.lower() in ('x-request-id','request-id','x-model','x-provider','x-omniroute-provider','x-omniroute-model','x-fallback','x-omniroute-fallback')}
        return response

def run(out, authorize=False, max_requests=44, token_budget=2000000, models=MODELS, send=transport, luna_only_after_deepseek_failure=False):
    rows=packet(); ids=pilot_ids(rows); selected=[r for r in rows if r['query_id'] in ids]
    meta={'schema_version':'r2-v2-pilot-1','packet_manifest_sha256':digest((HERE/'r2_packet_manifest.json').read_bytes()),'rubric_sha256':digest(RUBRIC.read_bytes()),'models':list(models),'pilot_ids':ids,'request_max_output_tokens':1024,'max_requests':max_requests,'token_budget':token_budget}
    if not authorize: return {'mode':'offline-dry-run','manifest':meta,'evaluations':len(selected)*2*len(models)}
    from run_revision_pilot import load_local_env
    load_local_env(); key=os.environ.get('OMNIROUTE_API_KEY')
    if not key: raise RuntimeError('OMNIROUTE_API_KEY missing')
    # Avoid sending any configured secret embedded in input evidence or answers.
    blob=(HERE/'paired_judge_inputs.jsonl').read_text()
    sensitive = [v for k,v in os.environ.items() if len(v)>=12 and any(w in k.upper() for w in ('API_KEY','SECRET','PASSWORD','TOKEN'))]
    if any(v in blob for v in sensitive) or re.search(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bsk-[A-Za-z0-9_-]{20,}',blob): raise RuntimeError('possible credential found in packet')
    catalog=send('/models',key); catalog_ids={m['id'] for m in catalog['data']}
    missing=[m for m in models if m not in catalog_ids]
    if missing: raise RuntimeError('exact model absent from catalog: '+', '.join(missing))
    catalog_map={m['id']:m for m in catalog['data']}
    deep=catalog_map.get(MODELS[1],{})
    if MODELS[1] in models and deep.get('root') != 'deepseek-v4-flash-low': raise RuntimeError('DeepSeek catalog root does not support normalization')
    out=Path(out); out.mkdir(parents=True,exist_ok=True); manifest=out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != meta: raise RuntimeError('manifest changed: choose a new directory')
    if not manifest.exists(): atomic(manifest,meta)
    records=list(out.glob('attempt-*.json')); used=sum(json.loads(p.read_text()).get('reserved_tokens',0) for p in records)
    # Wall-clock attempt timestamps survive restart. Never retry an uncertain attempt.
    last={}
    for p in records:
        prior=json.loads(p.read_text()); m=prior['model_requested']
        last[m]=max(last.get(m,0),prior.get('started_epoch',0))
    if luna_only_after_deepseek_failure:
        if not any(json.loads(p.read_text()).get('model_requested')==MODELS[1] and json.loads(p.read_text()).get('http_status')==500 and 'DEVIN_AGENTIC_HOME' in json.loads(p.read_text()).get('error_body','') for p in records):
            raise RuntimeError('DeepSeek gateway configuration failure not recorded')
    for row in selected:
      for baseline in ('B3','B5'):
       for model in models:
        if luna_only_after_deepseek_failure and model==MODELS[1]: continue
        request=payload(row,baseline,model); request_hash=digest(encode(request)); ident=digest(encode([model,row['query_id'],baseline,request_hash]))
        if (out/('result-'+ident+'.json')).exists(): continue
        prior=sorted(out.glob('attempt-'+ident+'-*.json'))
        if any(json.loads(p.read_text())['state'] not in ('rate_limited',) for p in prior):
            raise RuntimeError('previous incomplete/failed attempt requires manual reconciliation: '+ident)
        if len(prior)>4: raise RuntimeError('retry limit exhausted')
        while True:
            reserve=len(encode(request))+1024
            retry_count=sum(1 for p in records if json.loads(p.read_text()).get('state')=='rate_limited')
            if len(records)>=max_requests or used+reserve>token_budget or retry_count>4: raise RuntimeError('request, retry, or conservative token budget exhausted')
            delay=4 if model.startswith('cx/') else 7
            wait=delay-(time.time()-last.get(model,0))
            if wait>0: time.sleep(wait)
            attempt=out/('attempt-'+ident+'-'+str(len(prior)+1)+'.json')
            rec={'id':ident,'model_requested':model,'query_id':row['query_id'],'baseline':baseline,'request_sha256':request_hash,'evidence_sha256':row['judge_inputs'][baseline]['evidence_sha256'],'rubric_sha256':meta['rubric_sha256'],'reserved_tokens':reserve,'state':'started','started_epoch':time.time()}
            atomic(attempt,rec); records.append(attempt); prior.append(attempt); used+=reserve; start=time.monotonic(); last[model]=rec['started_epoch']
            try:
                response=send('/chat/completions',key,request); rec['latency_seconds']=time.monotonic()-start; rec['response']=response
                if response.get('model') not in (model, RETURNED_MODELS[model]): raise ValueError('returned model does not match approved route mapping')
                parsed=parse(response['choices'][0]['message']['content'],row['answerable']); rec['state']='valid';rec['parsed']=parsed
            except urllib.error.HTTPError as e:
                rec['latency_seconds']=time.monotonic()-start;rec['http_status']=e.code
                rec['error_body']=e.read().decode('utf-8','replace').replace(key,'[REDACTED]')
                rec['state']='rate_limited' if e.code==429 else 'error';atomic(attempt,rec)
                if e.code==429 and retry_count<4 and len(records)<max_requests:
                    time.sleep(min(60,2**len(prior)));continue
                raise RuntimeError('pilot stopped: HTTP '+str(e.code)) from e
            except Exception as e:
                rec['latency_seconds']=time.monotonic()-start;rec['state']='uncertain' if isinstance(e,(TimeoutError,urllib.error.URLError)) else 'error';rec['error_type']=type(e).__name__;rec['error']=str(e).replace(key,'[REDACTED]');atomic(attempt,rec);raise RuntimeError('pilot stopped: '+rec['error']) from e
            atomic(attempt,rec);atomic(out/('result-'+ident+'.json'),rec);break
    return {'mode':'partial' if luna_only_after_deepseek_failure else 'complete','evaluations':len(list(out.glob('result-*.json')))}

def reconcile_luna(out):
    """Accept only the previously preserved response, without modifying its attempt."""
    out=Path(out); manifest=json.loads((out/'manifest.json').read_text()); rows=packet()
    if manifest['rubric_sha256'] != digest(RUBRIC.read_bytes()) or manifest['packet_manifest_sha256'] != digest((HERE/'r2_packet_manifest.json').read_bytes()): raise ValueError('manifest changed')
    attempts=list(out.glob('attempt-8da02944044d4713dabb1439858c04996c6e3c4906094c58c303da080479364e-1.json'))
    if len(attempts)!=1: raise ValueError('expected original attempt missing')
    f=attempts[0]; original=json.loads(f.read_text()); row=next(r for r in rows if r['query_id']==original['query_id'])
    model=MODELS[0]; b=original['baseline']; request_hash=digest(encode(payload(row,b,model)))
    expected=digest(encode([model,row['query_id'],b,request_hash]))
    if (original['id']!=expected or original['request_sha256']!=request_hash or original['rubric_sha256']!=manifest['rubric_sha256'] or original['evidence_sha256']!=row['judge_inputs'][b]['evidence_sha256'] or original['model_requested']!=model or original['response']['model']!=RETURNED_MODELS[model]): raise ValueError('reconciliation input/identity mismatch')
    parsed=parse(original['response']['choices'][0]['message']['content'],row['answerable'])
    rec={'original_attempt':f.name,'original_attempt_sha256':digest(f.read_bytes()),'reason':'explicitly approved route-to-upstream model normalization','requested_route':model,'returned_model':RETURNED_MODELS[model],'request_sha256':request_hash,'rubric_sha256':manifest['rubric_sha256']}
    path=out/('reconciliation-'+expected+'.json'); result=out/('result-'+expected+'.json')
    if path.exists() and json.loads(path.read_text())!=rec: raise ValueError('existing reconciliation mismatch')
    if not path.exists(): atomic(path,rec)
    accepted=dict(original); accepted['state']='valid'; accepted['parsed']=parsed;accepted['reconciliation']=path.name
    if result.exists() and json.loads(result.read_text())!=accepted: raise ValueError('existing result mismatch')
    if not result.exists(): atomic(result,accepted)
    return rec

def reconcile_valid_status(out):
    out=Path(out); rows=packet(); manifest=json.loads((out/'manifest.json').read_text())
    if manifest['rubric_sha256']!=digest(RUBRIC.read_bytes()) or manifest['packet_manifest_sha256']!=digest((HERE/'r2_packet_manifest.json').read_bytes()): raise ValueError('manifest changed')
    for f in out.glob('attempt-*.json'):
        rec=json.loads(f.read_text())
        if rec.get('error') != 'invalid status or rationale' or rec.get('model_requested')!=MODELS[0]: continue
        row=next(r for r in rows if r['query_id']==rec['query_id']); model=rec['model_requested'];b=rec['baseline']
        req=digest(encode(payload(row,b,model))); ident=digest(encode([model,row['query_id'],b,req]));response=rec['response']
        if rec['id']!=ident or rec['request_sha256']!=req or rec['evidence_sha256']!=row['judge_inputs'][b]['evidence_sha256'] or rec['rubric_sha256']!=manifest['rubric_sha256'] or response['model']!=RETURNED_MODELS[model]: raise ValueError('attempt mismatch')
        parsed=parse(response['choices'][0]['message']['content'],row['answerable'])
        note={'original_attempt':f.name,'original_attempt_sha256':digest(f.read_bytes()),'reason':'v2 rubric does not prescribe ok status; valid is a valid success status','request_sha256':req,'rubric_sha256':manifest['rubric_sha256']}
        path=out/('reconciliation-'+ident+'.json'); result=out/('result-'+ident+'.json')
        if not path.exists(): atomic(path,note)
        accepted=dict(rec);accepted['state']='valid';accepted['parsed']=parsed;accepted['reconciliation']=path.name
        if not result.exists(): atomic(result,accepted)

def main():
    p=argparse.ArgumentParser();p.add_argument('--authorize-judge-execution',action='store_true');p.add_argument('--reconcile-existing-luna',action='store_true');p.add_argument('--reconcile-valid-status',action='store_true');p.add_argument('--luna-only-after-deepseek-failure',action='store_true');p.add_argument('--output',default='revision_round1/experiments/r2_v2_pilot');p.add_argument('--max-requests',type=int,default=44);p.add_argument('--token-budget',type=int,default=2000000);a=p.parse_args()
    if a.reconcile_valid_status: reconcile_valid_status(a.output)
    elif a.reconcile_existing_luna: print(json.dumps(reconcile_luna(a.output),indent=2))
    else: print(json.dumps(run(a.output,a.authorize_judge_execution,a.max_requests,a.token_budget,luna_only_after_deepseek_failure=a.luna_only_after_deepseek_failure),indent=2))
if __name__=='__main__': main()
