"""Narrow, fail-closed runner for the recovered R2 v2 packet (not a general evaluator)."""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, tempfile, time, urllib.error, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = ('cx/chatgpt-6-luna-low', 'dva/deepseek-v4-flash-low')
BASE_URL = 'http://127.0.0.1:20128/v1'
RUBRIC = HERE / 'r2_shared_rubric_v2.md'

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
    if not isinstance(o,dict) or o.get('status') != 'ok' or not isinstance(o.get('rationale'),str) or not o['rationale'].strip(): raise ValueError('invalid status or rationale')
    for name in ('semantic_accuracy','faithfulness'):
        v=o.get(name)
        if name=='semantic_accuracy' and not answerable:
            if v is not None: raise ValueError('negative semantic score must be null')
        elif isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1: raise ValueError('invalid '+name)
    return o

def transport(path, key, body=None):
    req=urllib.request.Request(BASE_URL+path,data=encode(body) if body is not None else None,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=120) as r: return json.loads(r.read())

def run(out, authorize=False, max_requests=40, token_budget=2000000, models=MODELS, send=transport):
    rows=packet(); ids=pilot_ids(rows); selected=[r for r in rows if r['query_id'] in ids]
    meta={'schema_version':'r2-v2-pilot-1','packet_manifest_sha256':digest((HERE/'r2_packet_manifest.json').read_bytes()),'rubric_sha256':digest(RUBRIC.read_bytes()),'models':list(models),'pilot_ids':ids,'request_max_output_tokens':1024,'max_requests':max_requests,'token_budget':token_budget}
    if not authorize: return {'mode':'offline-dry-run','manifest':meta,'evaluations':len(selected)*2*len(models)}
    from run_revision_pilot import load_local_env
    load_local_env(); key=os.environ.get('OMNIROUTE_API_KEY')
    if not key: raise RuntimeError('OMNIROUTE_API_KEY missing')
    # Avoid sending any configured secret embedded in input evidence or answers.
    blob=(HERE/'paired_judge_inputs.jsonl').read_text()
    if len(key)>=8 and key in blob: raise RuntimeError('credential found in packet')
    catalog=send('/models',key); catalog_ids={m['id'] for m in catalog['data']}
    missing=[m for m in models if m not in catalog_ids]
    if missing: raise RuntimeError('exact model absent from catalog: '+', '.join(missing))
    out=Path(out); out.mkdir(parents=True,exist_ok=True); manifest=out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != meta: raise RuntimeError('manifest changed: choose a new directory')
    if not manifest.exists(): atomic(manifest,meta)
    last={}; records=list(out.glob('attempt-*.json')); used=sum(json.loads(p.read_text()).get('reserved_tokens',0) for p in records)
    for row in selected:
      for baseline in ('B3','B5'):
       for model in models:
        request=payload(row,baseline,model); request_hash=digest(encode(request)); ident=digest(encode([model,row['query_id'],baseline,request_hash]))
        if (out/('result-'+ident+'.json')).exists(): continue
        if list(out.glob('attempt-'+ident+'-*.json')): raise RuntimeError('previous incomplete/failed attempt requires manual reconciliation: '+ident)
        reserve=len(encode(request))+1024 # byte-length upper bound for ordinary tokenizers, not a verified provider guarantee
        if len(records)>=max_requests or used+reserve>token_budget: raise RuntimeError('request or conservative token budget exhausted')
        delay=4 if model.startswith('cx/') else 7
        wait=delay-(time.monotonic()-last.get(model,0))
        if wait>0: time.sleep(wait)
        attempt=out/('attempt-'+ident+'-1.json'); rec={'id':ident,'model_requested':model,'query_id':row['query_id'],'baseline':baseline,'request_sha256':request_hash,'evidence_sha256':row['judge_inputs'][baseline]['evidence_sha256'],'rubric_sha256':meta['rubric_sha256'],'reserved_tokens':reserve,'state':'started'}
        atomic(attempt,rec); records.append(attempt); used+=reserve; start=time.monotonic(); last[model]=start
        try:
            response=send('/chat/completions',key,request); rec['latency_seconds']=time.monotonic()-start; rec['response']=response
            if response.get('model') != model: raise ValueError('returned model does not exactly match requested route')
            parsed=parse(response['choices'][0]['message']['content'],row['answerable']); rec['state']='valid';rec['parsed']=parsed
        except Exception as e:
            rec['latency_seconds']=time.monotonic()-start;rec['state']='error';rec['error_type']=type(e).__name__;rec['error']=str(e).replace(key,'[REDACTED]');atomic(attempt,rec);raise RuntimeError('pilot stopped: '+rec['error']) from e
        atomic(attempt,rec);atomic(out/('result-'+ident+'.json'),rec)
    return {'mode':'complete','evaluations':len(list(out.glob('result-*.json')))}

def main():
    p=argparse.ArgumentParser();p.add_argument('--authorize-judge-execution',action='store_true');p.add_argument('--output',default='revision_round1/experiments/r2_v2_pilot');p.add_argument('--max-requests',type=int,default=40);p.add_argument('--token-budget',type=int,default=2000000);a=p.parse_args()
    print(json.dumps(run(a.output,a.authorize_judge_execution,a.max_requests,a.token_budget),indent=2))
if __name__=='__main__': main()
