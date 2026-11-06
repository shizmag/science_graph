#!/usr/bin/env python3
"""Offline, blinded file bridge; all grading is performed by the interactive judge."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, random, re, sys, tempfile
from pathlib import Path
from collections import Counter, defaultdict

HERE=Path(__file__).resolve().parent
JUDGE=HERE.parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(JUDGE))
import r2_runner_v2 as source
WORKSPACE=ROOT/'agy_r2_workspace'
PRIVATE=HERE/'private'/'mapping.json'
RUBRIC_VERSION='v2'
EXPECTED_PACKET_SHA256='19b87696e4c2dc82d8214b3a47b1647bf2a5c0886b8d4921bc8dd5272f85ea90'
EXPECTED_RUBRIC_SHA256='f50a8f42fd4d8e6c8f2c4972e52dcdde12cb81fce4aa3344dd07b6bf8e0047b9'
TASK_KEYS={'evaluation_id','question','reference_answer','answerable','generated_answer','generator_visible_evidence','evidence_identifiers','rubric_version','rubric','input_sha256'}
RESULT_KEYS={'evaluation_id','input_sha256','rubric_version','judge_session_id','status','semantic_accuracy','faithfulness','semantic_accuracy_rationale','faithfulness_rationale','evidence_identifiers','unsupported_claims','unresolved_citation_mapping','error_reason'}

def canonical(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def sha(x): return hashlib.sha256(x).hexdigest()
def task_hash(task): return sha(canonical({k:v for k,v in task.items() if k!='input_sha256'}))
def read(path): return json.loads(Path(path).read_text())
def write(path,value,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);data=canonical(value)+b'\n'
    if exclusive:
        # Hard-linking the completed tempfile is atomic and never overwrites a result.
        fd,name=tempfile.mkstemp(dir=path.parent,prefix='.tmp-')
        try:
            with os.fdopen(fd,'wb') as f: f.write(data);f.flush();os.fsync(f.fileno())
            os.link(name,path)
        finally: os.unlink(name)
    else:
        fd,name=tempfile.mkstemp(dir=path.parent,prefix='.tmp-')
        try:
            with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
            os.replace(name,path)
        finally:
            if os.path.exists(name):os.unlink(name)

def screen(rows):
    text=json.dumps(rows,ensure_ascii=False)
    secrets=[v for k,v in os.environ.items() if len(v)>=12 and any(x in k.upper() for x in ('API_KEY','SECRET','PASSWORD','TOKEN'))]
    envfile=ROOT/'.env'
    if envfile.is_file():
        for line in envfile.read_text().splitlines():
            if '=' not in line or line.lstrip().startswith('#'):continue
            key,value=line.removeprefix('export ').split('=',1);value=value.strip().strip("\"'")
            if len(value)>=12 and any(x in key.upper() for x in ('API_KEY','SECRET','PASSWORD','TOKEN')):secrets.append(value)
    if any(v in text for v in secrets) or re.search(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bsk-[A-Za-z0-9_-]{20,}',text): raise ValueError('possible credential in packet')

def prepare(workspace=WORKSPACE, private=PRIVATE):
    if sha((JUDGE/'r2_packet_manifest.json').read_bytes())!=EXPECTED_PACKET_SHA256:raise ValueError('source manifest hash mismatch')
    rows=source.packet(); screen(rows); rubric=source.RUBRIC.read_text(); rh=sha(rubric.encode())
    if rh!=EXPECTED_RUBRIC_SHA256:raise ValueError('frozen rubric hash mismatch')
    byid={r['query_id']:r for r in rows};qids=sorted(byid);rng=random.Random(20261106);rng.shuffle(qids)
    tasks={}; mapping={}; batches=[]; used=set()
    for i in range(20):
        batch=[]
        for q in qids[i*5:i*5+5]:
            r=byid[q]
            for baseline in ('B3','B5'):
                x=r['judge_inputs'][baseline]; ev=x['generator_visible_evidence']
                while True:
                    eid='E'+rng.randbytes(12).hex()
                    if eid not in used:used.add(eid);break
                identifiers=sorted({s for s in x['citation_metadata'] if s and s in ev})
                if not identifiers: raise ValueError('no visible evidence identifiers')
                task={'evaluation_id':eid,'question':r['query'],'reference_answer':r['reference_answer'] if r['answerable'] else None,'answerable':r['answerable'],'generated_answer':x['candidate_answer'],'generator_visible_evidence':ev,'evidence_identifiers':identifiers,'rubric_version':RUBRIC_VERSION,'rubric':rubric}
                task['input_sha256']=task_hash(task);tasks[eid]=task;mapping[eid]={'query_id':q,'baseline':baseline,'evidence_sha256':x['evidence_sha256'],'input_sha256':task['input_sha256']};batch.append(eid)
        rng.shuffle(batch);batches.append({'batch':i+1,'tasks':batch})
    public={'schema':'agy-r2-v2-1','rubric_version':RUBRIC_VERSION,'rubric_sha256':rh,'batches':batches,'inputs':{eid:t['input_sha256'] for eid,t in sorted(tasks.items())}}
    hidden={'schema':'agy-r2-private-1','packet_manifest_sha256':sha((JUDGE/'r2_packet_manifest.json').read_bytes()),'rubric_sha256':rh,'public_manifest_sha256':sha(canonical(public)),'mapping':mapping}
    workspace=Path(workspace);private=Path(private)
    # Preparation is repeatable but will not replace altered existing input files.
    for eid,t in tasks.items():
        p=workspace/'tasks'/f'{eid}.json'
        if p.exists() and read(p)!=t:raise ValueError('existing task differs: '+eid)
    for p,data in [(workspace/'manifest.json',public),(private,hidden)]:
        if p.exists() and read(p)!=data:raise ValueError('existing manifest differs: '+str(p))
    for eid,t in tasks.items():
        p=workspace/'tasks'/f'{eid}.json'
        if not p.exists():write(p,t,True)
    for p,data in [(workspace/'manifest.json',public),(private,hidden)]:
        if not p.exists():write(p,data,True)
    return {'batches':20,'tasks':len(tasks),'rubric_sha256':rh}

def validate_task(task,manifest):
    if set(task)!=TASK_KEYS or task.get('rubric_version')!=RUBRIC_VERSION or task_hash(task)!=task.get('input_sha256') or manifest['inputs'].get(task.get('evaluation_id'))!=task['input_sha256']: raise ValueError('task format or hash mismatch')
    if sha(task['rubric'].encode())!=manifest['rubric_sha256']:raise ValueError('rubric mismatch')
    if not isinstance(task['answerable'],bool) or (task['answerable'] and not task['reference_answer']) or (not task['answerable'] and task['reference_answer'] is not None):raise ValueError('reference eligibility mismatch')
    if not task['evidence_identifiers'] or any(not isinstance(x,str) or x not in task['generator_visible_evidence'] for x in task['evidence_identifiers']):raise ValueError('evidence identifiers not in evidence')

def validate_result(t,r):
    if not isinstance(r,dict) or set(r)!=RESULT_KEYS:raise ValueError('result keys mismatch')
    if r['evaluation_id']!=t['evaluation_id'] or r['input_sha256']!=t['input_sha256'] or r['rubric_version']!=RUBRIC_VERSION:raise ValueError('result input/rubric mismatch')
    if not isinstance(r['judge_session_id'],str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',r['judge_session_id']):raise ValueError('invalid session')
    if r['status'] not in ('scored','partial','unevaluable'):raise ValueError('invalid status')
    if not isinstance(r['unresolved_citation_mapping'],bool):raise ValueError('invalid citation flag')
    for key in ('semantic_accuracy_rationale','faithfulness_rationale'):
        if not isinstance(r[key],str) or not r[key].strip():raise ValueError('missing rationale')
    for key in ('evidence_identifiers','unsupported_claims'):
        if not isinstance(r[key],list) or any(not isinstance(x,str) or not x.strip() for x in r[key]):raise ValueError('invalid list')
    if len(set(r['evidence_identifiers']))!=len(r['evidence_identifiers']) or not set(r['evidence_identifiers'])<=set(t['evidence_identifiers']):raise ValueError('unknown or duplicate evidence identifiers')
    if r['status']=='unevaluable':
        if r['semantic_accuracy'] is not None or r['faithfulness'] is not None or not isinstance(r['error_reason'],dict) or set(r['error_reason'])!={'code','detail'} or not all(isinstance(v,str) and v.strip() for v in r['error_reason'].values()):raise ValueError('unevaluable requires null scores and structured reason')
        return
    if r['status']=='scored' and r['error_reason'] is not None:raise ValueError('scored record has error')
    if r['status']=='partial':
        reason=r['error_reason']
        if not isinstance(reason,dict) or set(reason)!={'code','detail'} or not all(isinstance(v,str) and v.strip() for v in reason.values()):raise ValueError('partial result requires structured reason')
        if r['faithfulness'] is not None and (not t['answerable'] or r['semantic_accuracy'] is not None):raise ValueError('partial result has no missing applicable metric')
    for metric in ('semantic_accuracy','faithfulness'):
        v=r[metric]
        if metric=='semantic_accuracy' and not t['answerable']:
            if v is not None:raise ValueError('negative semantic accuracy must be null')
        elif v is None and r['status']=='partial': pass
        elif isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1:raise ValueError('invalid '+metric)
    if not r['evidence_identifiers']:raise ValueError('scored result requires evidence identifier')

def check(workspace=WORKSPACE):
    w=Path(workspace);m=read(w/'manifest.json');expected=set(m['inputs']);listed=[e for batch in m['batches'] for e in batch['tasks']]
    if m.get('rubric_sha256')!=EXPECTED_RUBRIC_SHA256:raise ValueError('unrecognized rubric')
    if len(listed)!=len(expected) or set(listed)!=expected:raise ValueError('manifest duplicate/missing ID')
    files={p.stem:p for p in (w/'tasks').glob('*.json')}
    if set(files)!=expected:raise ValueError('missing or unexpected tasks')
    tasks={}
    for eid,p in files.items():
        t=read(p);validate_task(t,m)
        if t['evaluation_id']!=eid:raise ValueError('task filename mismatch')
        tasks[eid]=t
    results={};invalid=[]; grouped=defaultdict(list)
    for p in (w/'results').glob('*.json'):
        match=re.fullmatch(r'(E[0-9a-f]{24})(?:\.v([0-9]+))?\.json',p.name)
        if not match or match[1] not in expected:invalid.append({'file':p.name,'error':'unexpected file or ID'});continue
        grouped[match[1]].append((int(match[2] or 1),p))
    for eid,versions in grouped.items():
        versions.sort();nums=[n for n,_ in versions]
        if nums!=list(range(1,len(nums)+1)):invalid.append({'id':eid,'error':'duplicate or nonconsecutive correction versions'});continue
        for n,p in versions:
            try:
                r=read(p);validate_result(tasks[eid],r)
                session=w/'sessions'/(r['judge_session_id']+'.json')
                if not session.is_file(): raise ValueError('session metadata missing')
                info=read(session)
                if info.get('session_id')!=r['judge_session_id'] or not info.get('model'): raise ValueError('invalid session metadata')
                if n>1:
                    note=w/'corrections'/(p.name+'.json');c=read(note)
                    if c['previous_sha256']!=sha(versions[n-2][1].read_bytes()) or not c['reason'].strip():raise ValueError('correction provenance invalid')
                results[eid]=r
            except (ValueError,KeyError,TypeError,OSError,json.JSONDecodeError) as e:invalid.append({'file':p.name,'error':str(e)});results.pop(eid,None);break
    return m,tasks,results,invalid

def status(workspace=WORKSPACE):
    m,tasks,results,invalid=check(workspace); counts=Counter(r['status'] for r in results.values());invalid_ids={x.get('id',x.get('file','').split('.')[0]) for x in invalid}
    return {'expected':len(tasks),'scored':counts['scored'],'unevaluable':counts['unevaluable'],'partial':counts['partial'],'invalid':invalid,'pending':len(set(tasks)-set(results)-invalid_ids),'batches':[{'batch':b['batch'],'completed':sum(e in results for e in b['tasks']),'total':len(b['tasks'])} for b in m['batches']]}

def next_task(workspace=WORKSPACE):
    m,tasks,results,invalid=check(workspace)
    if invalid:raise ValueError('invalid or duplicate results require author repair before continuing')
    for batch in m['batches']:
        for eid in batch['tasks']:
            if eid not in results:return {'batch':batch['batch'],'evaluation_id':eid,'task_path':str(Path(workspace)/'tasks'/f'{eid}.json'),'remaining_in_batch':sum(x not in results for x in batch['tasks'])}
    return {'complete':True}

def save_session(w,session,model):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',session) or not model.strip():raise ValueError('invalid session or model')
    write(Path(w)/'sessions'/f'{session}.json',{'session_id':session,'model':model,'identity_note':'Author-supplied; not independently verified. Record unknown if unavailable.'},True)

def submit(w,draft,correction_reason=None):
    w=Path(w);m,tasks,results,invalid=check(w)
    if invalid:raise ValueError('invalid existing records; author repair required')
    r=read(draft);eid=r.get('evaluation_id');
    if eid not in tasks:raise ValueError('unexpected evaluation id')
    validate_result(tasks[eid],r)
    if not (w/'sessions'/f"{r['judge_session_id']}.json").is_file():raise ValueError('session not registered')
    existing=sorted((w/'results').glob(eid+'*.json'),key=lambda p: int(re.search(r'\.v(\d+)\.json$',p.name).group(1)) if re.search(r'\.v(\d+)\.json$',p.name) else 1)
    if existing and not correction_reason:raise ValueError('completed result exists; explicit correction required')
    if not existing and correction_reason:raise ValueError('no result to correct')
    n=len(existing)+1;dest=w/'results'/(eid+('' if n==1 else f'.v{n}')+'.json')
    if n>1:write(w/'corrections'/(dest.name+'.json'),{'previous_sha256':sha(existing[-1].read_bytes()),'reason':correction_reason},True)
    write(dest,r,True);return str(dest)

def export(workspace=WORKSPACE,private=PRIVATE,output=None,luna_dir=None):
    """Unblind only here; no missing value is imputed."""
    import numpy as np
    import analyze_agreement as stats
    w=Path(workspace);m,tasks,results,invalid=check(w)
    if invalid:raise ValueError('invalid records block aggregation')
    pm=read(private)
    if pm['public_manifest_sha256']!=sha(canonical(m)) or pm['rubric_sha256']!=m['rubric_sha256']:raise ValueError('private/public manifest mismatch')
    rows=[]
    for eid,meta in pm['mapping'].items():
        r=results.get(eid);rows.append({'query_id':meta['query_id'],'baseline':meta['baseline'],'evaluation_id':eid,'status':r['status'] if r else 'missing','semantic_accuracy':r['semantic_accuracy'] if r else None,'faithfulness':r['faithfulness'] if r else None,'session_id':r['judge_session_id'] if r else None})
    grouped=defaultdict(dict)
    for row in rows: grouped[row['query_id']][row['baseline']]=row
    summary={'n_slots':len(rows),'statuses':dict(Counter(x['status'] for x in rows)),'model_identity_note':'Judge identity is self-reported; family independence is not automatically established.','metrics':{},'agreement':{}}
    for metric in ('semantic_accuracy','faithfulness'):
        pairs=[(v['B3'][metric],v['B5'][metric]) for v in grouped.values() if 'B3' in v and 'B5' in v and v['B3'][metric] is not None and v['B5'][metric] is not None]
        diffs=[a-b for a,b in pairs];ci=stats.bootstrap_ci(diffs,lambda x:sum(x)/len(x),20261106)
        summary['metrics'][metric]={'paired_n':len(pairs),'b3_mean_paired':float(np.mean([p[0] for p in pairs])) if pairs else None,'b5_mean_paired':float(np.mean([p[1] for p in pairs])) if pairs else None,'b3_minus_b5':float(np.mean(diffs)) if diffs else None,'ci95':[float(x) if math.isfinite(x) else None for x in ci]}
    if luna_dir:
        lunar={}
        source_rows={r['query_id']:r for r in source.packet()}
        for p in Path(luna_dir).glob('result-*.json'):
            rec=read(p)
            if rec.get('state')!='valid' or rec.get('model_requested')!='cx/gpt-6-luna-low' or rec.get('rubric_sha256')!=m['rubric_sha256']:continue
            q=rec['query_id'];base=rec['baseline'];row=source_rows.get(q)
            if row is None or base not in ('B3','B5') or rec['request_sha256']!=source.digest(source.encode(source.payload(row,base,rec['model_requested']))) or rec['evidence_sha256']!=row['judge_inputs'][base]['evidence_sha256']:raise ValueError('Luna input mismatch')
            key=(q,base)
            if key in lunar:raise ValueError('duplicate Luna results')
            lunar[key]=rec['parsed']
        for metric in ('semantic_accuracy','faithfulness'):
            obs=[(row,lunar[(row['query_id'],row['baseline'])][metric]) for row in rows if (row['query_id'],row['baseline']) in lunar and row[metric] is not None and lunar[(row['query_id'],row['baseline'])].get(metric) is not None]
            x=np.array([v for _,v in obs],dtype=float);y=np.array([row[metric] for row,_ in obs],dtype=float)
            sp=stats.safe_spearman(x,y) if len(x) else math.nan;icc=stats.icc_a1(x,y) if len(x) else math.nan
            summary['agreement'][metric]={'matched_observations':len(obs),'matched_queries':len({r['query_id'] for r,_ in obs}),'spearman':sp if math.isfinite(sp) else None,'icc_absolute_agreement':icc if math.isfinite(icc) else None,'mean_absolute_difference':float(np.mean(np.abs(x-y))) if len(x) else None,'largest_disagreements':[{'query_id':r['query_id'],'baseline':r['baseline'],'luna':v,'judge_b':r[metric],'absolute_difference':abs(v-r[metric]),'judge_b_rationale':results[r['evaluation_id']][metric+'_rationale'],'luna_rationale':lunar[(r['query_id'],r['baseline'])].get('rationale')} for r,v in sorted(obs,key=lambda z:abs(z[1]-z[0][metric]),reverse=True)[:10]]}
    if output:
        out=Path(output);out.mkdir(parents=True,exist_ok=True)
        with (out/'paired_scores.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        paired=[]
        for q,v in sorted(grouped.items()):
            paired.append({'query_id':q,**{base+'_'+metric:v.get(base,{}).get(metric) for base in ('B3','B5') for metric in ('semantic_accuracy','faithfulness')}})
        with (out/'query_pairs.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
        write(out/'summary.json',summary)
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,default=WORKSPACE);p.add_argument('--private',type=Path,default=PRIVATE)
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('prepare','status','next','validate'):sub.add_parser(name)
    s=sub.add_parser('session');s.add_argument('--id',required=True);s.add_argument('--model',required=True)
    s=sub.add_parser('submit');s.add_argument('draft');s.add_argument('--correction-reason')
    s=sub.add_parser('export');s.add_argument('--output',type=Path,required=True);s.add_argument('--luna-dir',type=Path)
    a=p.parse_args()
    if a.command=='prepare':result=prepare(a.workspace,a.private)
    elif a.command in ('status','validate'):result=status(a.workspace)
    elif a.command=='next':result=next_task(a.workspace)
    elif a.command=='session':save_session(a.workspace,a.id,a.model);result={'session':a.id}
    elif a.command=='submit':result={'saved':submit(a.workspace,a.draft,a.correction_reason)}
    else:result=export(a.workspace,a.private,a.output,a.luna_dir)
    print(json.dumps(result,indent=2,ensure_ascii=False))
    if a.command=='validate' and result['invalid']:raise SystemExit(1)
if __name__=='__main__':main()
