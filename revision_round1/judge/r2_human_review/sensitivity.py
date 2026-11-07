#!/usr/bin/env python3
"""Post-hoc complete-pair sensitivity; never creates or replaces a score."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np

def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def ident(q,b):return hashlib.sha256(canonical([q,b])).hexdigest()[:24]
def summarize(results,exclude):
 out={}
 for provider in ('omniroute','openrouter'):
  qs={}
  for rec in results[provider]:
   t=rec['task'];q=t['query_id']
   if t['question_type']!='answerable' or q in exclude:continue
   baseline=next(b for b in ('B3','B5') if ident(q,b)==t['evaluation_id'])
   qs.setdefault(q,{})[baseline]=rec['result']['faithfulness']
  pairs=[(v['B3'],v['B5']) for v in qs.values() if set(v)=={'B3','B5'} and v['B3'] is not None and v['B5'] is not None]
  d=np.array([x-y for x,y in pairs]);rng=np.random.default_rng(20261106)
  boot=[float(np.mean(d[rng.integers(0,len(d),len(d))])) for _ in range(5000)] if len(d)>=2 else []
  out[provider]={'paired_n':len(pairs),'b3_mean':float(np.mean([x for x,y in pairs])) if pairs else None,'b5_mean':float(np.mean([y for x,y in pairs])) if pairs else None,'difference':float(np.mean(d)) if pairs else None,'ci95':np.quantile(boot,[.025,.975]).tolist() if boot else None}
 return out

def decisions(path,mapping):
 byid={m['case_id']:m for m in mapping}; rows=list(csv.DictReader(path.open()));seen=set();excluded=set()
 if len(rows)!=len(byid):raise ValueError('adjudicated form must contain all cases')
 for row in rows:
  case=row['case_id']; action=row['exclude_query'].strip().lower()
  if case not in byid or case in seen:raise ValueError('unknown or duplicate case')
  seen.add(case)
  if action not in ('yes','no') or not row['reviewer_id'].strip() or not row['adjudicator_id'].strip() or not row['category'].strip() or not row['reason'].strip():raise ValueError('human adjudication incomplete; no scenario C calculated')
  if action=='yes':excluded.add(byid[case]['query_id'])
 return excluded

def main():
 a=argparse.ArgumentParser();a.add_argument('--source',type=Path,required=True);a.add_argument('--packet',type=Path,required=True);a.add_argument('--decisions',type=Path);a.add_argument('--output',type=Path,required=True);x=a.parse_args();src=x.source.resolve();exp=src/'revision_round1/experiments/r2_api_v3';audit=src/'revision_round1/judge/r2_v3_audit'
 # Verify all frozen records and source inputs before calculating.
 for line in (audit/'frozen_sha256.txt').read_text().splitlines():
  h,name=line.split('  ',1);p=src/name
  if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise ValueError('frozen artifact hash mismatch: '+name)
 results={p:[json.loads(f.read_text()) for f in (exp/'results'/p).glob('*.json')] for p in ('omniroute','openrouter')}
 if any(len(v)!=200 for v in results.values()):raise ValueError('expected 200 records per judge')
 original=summarize(results,set());old=json.loads((audit/'audit_results.json').read_text())['contrasts']
 for p,v in original.items():
  prior=old[p+'/answerable/faithfulness']
  if v['paired_n']!=prior['n'] or abs(v['difference']-prior['diff'])>1e-12:raise ValueError('frozen primary mismatch')
 out={'label':'post-hoc sensitivity; original judge scores unchanged','scenario_A_original':original,'scenario_B_exclude_flagged_query':{'excluded_query_ids':['pubmedqa_16097998'],'results':summarize(results,{'pubmedqa_16097998'})},'scenario_C_adjudicated':{'status':'awaiting_actual_human_adjudication'}}
 if x.decisions:
  excluded=decisions(x.decisions,json.loads((x.packet/'private/faithfulness_map.json').read_text()));out['scenario_C_adjudicated']={'status':'human_decisions_supplied','excluded_query_ids':sorted(excluded),'results':summarize(results,excluded),'interpretation':'conditional on human exclusions; original LLM scores only, no imputed scores'}
 payload=json.dumps(out,indent=2)+'\n';x.output.parent.mkdir(parents=True,exist_ok=True)
 if x.output.exists() and x.output.read_text()!=payload:raise ValueError('refusing to overwrite existing sensitivity result; use a new path')
 x.output.write_text(payload);print(payload)
if __name__=='__main__':main()
