#!/usr/bin/env python3
"""Offline R2 provenance gates and supplementary negative-v1 processing.

This code validates records and computes statistics; it never produces judgments.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, math, re, shutil, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
import bridge
WORK=ROOT/'agy_judge_only'
PRIVATE=HERE/'private'
REVIEW_DIR=PRIVATE/'reviews'
NEGATIVE=HERE/'negative_safety_v1.md'
CORE_SCHEMA=HERE/'result.schema.json'
canonical=bridge.canonical
sha=bridge.sha

def negative_hash():return sha(NEGATIVE.read_bytes())
def suspicious(r):
    text=' '.join(str(r.get(k,'')) for k in ('semantic_accuracy_rationale','faithfulness_rationale','rationale'))
    patterns=('Semantic accuracy is determined by comparing','The task is marked answerable:false', 'token overlap', 'keyword matching', 'lexical overlap')
    return any(x.lower() in text.lower() for x in patterns)

def approved(record):
    h=sha(canonical(record));p=REVIEW_DIR/(h+'.json')
    if not p.is_file():return False
    try:r=bridge.read(p)
    except (ValueError,OSError):return False
    return r.get('record_sha256')==h and r.get('state')=='accepted' and all(isinstance(r.get(k),str) and r[k].strip() for k in ('reviewer','provenance_evidence','grounding_review'))

def validate_negative(task,r):
    keys={'evaluation_id','input_sha256','rubric_version','rubric_sha256','judge_session_id','status','negative_safety','potential_label_conflict','rationale','evidence_identifiers','error_reason'}
    if not isinstance(r,dict) or set(r)!=keys:raise ValueError('negative keys mismatch')
    if task['answerable'] or r['evaluation_id']!=task['evaluation_id'] or r['input_sha256']!=task['input_sha256'] or r['rubric_version']!='negative-v1' or r['rubric_sha256']!=negative_hash():raise ValueError('negative input/rubric mismatch')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',r['judge_session_id']):raise ValueError('invalid session')
    if type(r['potential_label_conflict']) is not bool or not isinstance(r['rationale'],str) or not r['rationale'].strip():raise ValueError('invalid conflict flag/rationale')
    ids=r['evidence_identifiers']
    if not isinstance(ids,list) or any(not isinstance(x,str) for x in ids) or len(ids)!=len(set(ids)) or not set(ids)<=set(task['evidence_identifiers']):raise ValueError('invalid evidence identifiers')
    status=r['status'];v=r['negative_safety']
    if status=='label_conflict':
        if not r['potential_label_conflict'] or v is not None or not ids or r['error_reason'] is not None:raise ValueError('conflict requires null score and evidence')
    elif status=='scored':
        if r['potential_label_conflict'] or type(v) not in (float,int) or v not in (0,0.5,1) or not ids or r['error_reason'] is not None:raise ValueError('invalid negative score')
    elif status=='unevaluable':
        reason=r['error_reason']
        if r['potential_label_conflict'] or v is not None or not isinstance(reason,dict) or set(reason)!={'code','detail'} or not all(isinstance(x,str) and x.strip() for x in reason.values()):raise ValueError('invalid unevaluable record')
    else:raise ValueError('invalid negative status')

def prepare():
    m,tasks,results,invalid=bridge.check(bridge.WORKSPACE)
    if invalid:raise ValueError('source packet or legacy results invalid')
    WORK.mkdir(parents=True,exist_ok=True)
    for eid,t in tasks.items():
        target=WORK/'tasks'/f'{eid}.json'
        if target.exists() and bridge.read(target)!=t:raise ValueError('judge task changed')
        if not target.exists():bridge.write(target,t,True)
    for dest,src in [('core_v2.md',bridge.source.RUBRIC),('negative_v1.md',NEGATIVE),('core.schema.json',CORE_SCHEMA),('negative.schema.json',HERE/'negative.schema.json'),('INSTRUCTIONS.md',HERE/'JUDGE_INSTRUCTIONS.md')]:
        p=WORK/dest;data=src.read_bytes()
        if p.exists() and p.read_bytes()!=data:raise ValueError('judge workspace file changed: '+dest)
        if not p.exists():p.write_bytes(data)
    p=WORK/'manifest.json'
    if p.exists() and bridge.read(p)!=m:raise ValueError('judge manifest changed')
    if not p.exists():bridge.write(p,m,True)
    (WORK/'results').mkdir(exist_ok=True);(WORK/'sessions').mkdir(exist_ok=True);(WORK/'negative_results').mkdir(exist_ok=True)
    return {'tasks':len(tasks),'batches':len(m['batches']),'negatives':sum(not t['answerable'] for t in tasks.values())}

def session_valid(w,sid):
    p=w/'sessions'/f'{sid}.json'
    try:s=bridge.read(p)
    except (OSError,ValueError):return False
    return s.get('session_id')==sid and isinstance(s.get('model'),str) and bool(s['model'].strip())

def inventory(w=WORK):
    w=Path(w);m,tasks,results,invalid=bridge.check(w)
    counts=collections.Counter();records=[]
    for eid,t in tasks.items():
        r=results.get(eid)
        if r is None:continue
        flags=[]
        if suspicious(r):flags.append('suspicious_heuristic_match')
        sess=bridge.read(w/'sessions'/f"{r['judge_session_id']}.json")
        if str(sess['model']).strip().lower() in ('unknown','unavailable'):flags.append('unknown_provenance')
        if r['faithfulness'] is None and r['status']=='partial' and (r.get('error_reason') or {}).get('code')!='claimless_abstention':flags.append('needs_semantic_review')
        state='author_accepted' if approved(r) and 'unknown_provenance' not in flags else ('suspicious_heuristic_match' if 'suspicious_heuristic_match' in flags else 'unknown_provenance' if 'unknown_provenance' in flags else 'provisionally_attributed')
        counts[state]+=1;records.append({'type':'core','evaluation_id':eid,'state':state,'flags':flags,'record_sha256':sha(canonical(r))})
    negative={};seen=set()
    for p in (w/'negative_results').glob('*.json'):
        try:
            r=bridge.read(p);eid=r['evaluation_id']
            if p.name!=eid+'.json' or eid in seen or eid not in tasks:raise ValueError('unexpected or duplicate negative record')
            seen.add(eid);validate_negative(tasks[eid],r)
            if not session_valid(w,r['judge_session_id']):raise ValueError('session missing')
            negative[eid]=r;flags=[]
            if suspicious(r):flags.append('suspicious_heuristic_match')
            sess=bridge.read(w/'sessions'/f"{r['judge_session_id']}.json")
            if str(sess['model']).strip().lower() in ('unknown','unavailable'):flags.append('unknown_provenance')
            if r['potential_label_conflict']:flags.append('needs_semantic_review')
            state='author_accepted' if approved(r) and 'unknown_provenance' not in flags else ('suspicious_heuristic_match' if 'suspicious_heuristic_match' in flags else 'unknown_provenance' if 'unknown_provenance' in flags else 'provisionally_attributed')
            counts['negative_'+state]+=1;records.append({'type':'negative','evaluation_id':eid,'state':state,'flags':flags,'record_sha256':sha(canonical(r))})
        except (ValueError,KeyError,TypeError,OSError,json.JSONDecodeError) as e:invalid.append({'file':str(p),'error':str(e)})
    summary={'core_expected':len(tasks),'core_structurally_valid':len(results),'core_pending':len(tasks)-len(results),'negative_expected':sum(not t['answerable'] for t in tasks.values()),'negative_structurally_valid':len(negative),'negative_pending':sum(not t['answerable'] for t in tasks.values())-len(negative),'provenance_states':dict(counts),'invalid':invalid,'records':records,'limitations':['Structural validation does not prove model provenance.','Previously deleted first-run results are not available in this workspace.']}
    return summary,m,tasks,results,negative

def accept(record_path,reviewer,provenance,grounding):
    p=Path(record_path);root=WORK.resolve()
    if not p.resolve().is_relative_to(root) or p.parent.name not in ('results','negative_results'):raise ValueError('only judge workspace result records may be accepted')
    summary,_,_,_,_=inventory()
    r=bridge.read(p);h=sha(canonical(r));entry=next((e for e in summary['records'] if e['record_sha256']==h),None)
    if not entry or summary['invalid']:raise ValueError('record must be structurally valid')
    if entry['state']=='unknown_provenance' or 'unknown_provenance' in entry['flags']:raise ValueError('unknown model identity cannot be accepted')
    if not all(x.strip() for x in (reviewer,provenance,grounding)):raise ValueError('author review fields required')
    bridge.write(REVIEW_DIR/(h+'.json'),{'record_sha256':h,'state':'accepted','reviewer':reviewer,'provenance_evidence':provenance,'grounding_review':grounding,'flags_at_review':entry['flags'],'disclaimer':'Author attestation, not cryptographic proof of generation.'},True)
    return {'accepted_sha256':h,'flags':entry['flags']}

def submit_negative(record_path, workspace=WORK):
    w=Path(workspace);m,tasks,core,invalid=bridge.check(w)
    if invalid:raise ValueError('core workspace invalid')
    r=bridge.read(record_path);eid=r.get('evaluation_id')
    if eid not in tasks:raise ValueError('unexpected evaluation id')
    validate_negative(tasks[eid],r)
    if not session_valid(w,r['judge_session_id']):raise ValueError('session missing')
    dest=w/'negative_results'/(eid+'.json')
    if dest.exists():raise ValueError('negative result exists; preserve it and use a correction workflow')
    bridge.write(dest,r,True);return str(dest)

def export_accepted(output=WORK/'exports'):
    summary,m,tasks,core,negative=inventory();
    accepted={x['evaluation_id'] for x in summary['records'] if x['state']=='author_accepted'}
    pm=bridge.read(PRIVATE/'mapping.json'); rows=[]
    for eid,meta in pm['mapping'].items():
        if meta['query_id'] not in {v['query_id'] for v in pm['mapping'].values()}:continue
        record=core.get(eid) if eid in core else negative.get(eid)
        if eid not in accepted or record is None:continue
        rows.append({'evaluation_id':eid,'query_id':meta['query_id'],'baseline':meta['baseline'],'record_type':'negative' if not tasks[eid]['answerable'] else 'core','state':'author_accepted','semantic_accuracy':record.get('semantic_accuracy'),'faithfulness':record.get('faithfulness'),'negative_safety':record.get('negative_safety'),'potential_label_conflict':record.get('potential_label_conflict')})
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if rows:
        with (out/'accepted_scores.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    bridge.write(out/'provenance_status.json',summary)
    return {'accepted':len(rows),'output':str(out),'excluded':len(summary['records'])-len(rows),'states':summary['provenance_states']}

def cli():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
    sub.add_parser('prepare');sub.add_parser('inventory');sub.add_parser('export');sn=sub.add_parser('submit-negative');sn.add_argument('record');ac=sub.add_parser('accept');ac.add_argument('record');ac.add_argument('--reviewer',required=True);ac.add_argument('--provenance',required=True);ac.add_argument('--grounding',required=True)
    a=p.parse_args()
    if a.cmd=='prepare':x=prepare()
    elif a.cmd=='inventory':x=inventory()[0]
    elif a.cmd=='submit-negative':x={'saved':submit_negative(a.record)}
    elif a.cmd=='accept':x=accept(a.record,a.reviewer,a.provenance,a.grounding)
    else:x=export_accepted()
    print(json.dumps(x,indent=2,ensure_ascii=False))
if __name__=='__main__':cli()
