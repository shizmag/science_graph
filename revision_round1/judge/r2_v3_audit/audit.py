#!/usr/bin/env python3
"""Offline independent audit; reads original records, never modifies them."""
import collections, hashlib, json, math, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
HERE=Path(__file__).resolve().parent; JUDGE=HERE.parent; sys.path.insert(0,str(JUDGE))
import r2_api_runner_v3 as run
ROOT=JUDGE.parents[1]; EXP=ROOT/'revision_round1/experiments/r2_api_v3'

def icc(x,y):
 x=np.asarray(x,float);y=np.asarray(y,float); n=len(x)
 if n<2:return None
 m=np.stack([x,y],axis=1); grand=m.mean(); row=m.mean(axis=1);col=m.mean(axis=0)
 a=2*((row-grand)**2).sum()/(n-1); b=n*((col-grand)**2).sum();e=((m-row[:,None]-col[None,:]+grand)**2).sum()/(n-1)
 den=a+e+2*(b-e)/n
 return float((a-e)/den) if abs(den)>1e-15 else None

def spearman(x,y):
 if len(x)<2:return None
 a=rankdata(x);b=rankdata(y)
 if np.ptp(a)==0 or np.ptp(b)==0:return None
 return float(np.corrcoef(a,b)[0,1])
def kappa(x,y):
 if len(x)<2:return None
 x=np.array(x);y=np.array(y); observed=np.mean((x-y)**2); expected=np.mean((x[:,None]-y[None,:])**2)
 return float(1-observed/expected) if expected else None

def agreement(pairs, metric, bootstrap=2000):
 if not pairs:return {'n':0}
 x=[v[1] for v in pairs];y=[v[2] for v in pairs]; groups=collections.defaultdict(list)
 for q,a,b in pairs:groups[q].append((a,b))
 keys=sorted(groups);rng=np.random.default_rng(20261107); vals=collections.defaultdict(list)
 for _ in range(bootstrap):
  s=[v for i in rng.integers(0,len(keys),size=len(keys)) for v in groups[keys[i]]];a,b=zip(*s)
  for name,fn in [('mae',lambda a,b:float(np.mean(np.abs(np.array(a)-np.array(b))))),('spearman',spearman),('icc',icc)]+([('weighted_kappa',kappa)] if metric=='negative_safety' else []):
   v=fn(a,b)
   if v is not None and math.isfinite(v):vals[name].append(v)
 out={'n':len(x),'queries':len(keys),'mae':float(np.mean(np.abs(np.array(x)-np.array(y)))),'spearman':spearman(x,y),'icc':icc(x,y),'weighted_kappa':kappa(x,y) if metric=='negative_safety' else None,'bootstrap_valid':{k:len(v) for k,v in vals.items()},'cluster_ci95':{k:[float(z) for z in np.quantile(v,[.025,.975])] if len(v)>=bootstrap*.8 else None for k,v in vals.items()}}
 return out

def main():
 manifest=json.loads((EXP/'manifest.json').read_text());rh=run.sha(run.RUBRIC.read_bytes()); assert rh==manifest['rubric_sha256'];assert run.sha((JUDGE/'r2_packet_manifest.json').read_bytes())==manifest['packet_manifest_sha256']
 rows=run.cohort(); assert run.sha(run.canon(sorted(r['query_id'] for r in rows)))==manifest['cohort_ids_sha256']
 tasks=run.tasks_for(rows); byid={x['evaluation_id']:x for x in tasks};assert len(byid)==200
 results={}; provenance={}; query_order={}; limits=collections.Counter();not_app=[];app=collections.Counter(); contenthash=collections.defaultdict(set)
 for provider,cfg0 in run.MODELS.items():
  cfg=dict(cfg0,provider=provider);files=list((EXP/'results'/provider).glob('*.json'));query_order[provider]=list(dict.fromkeys(byid[f.stem]['query_id'] for f in files));assert len(files)==200 and {f.stem for f in files}==set(byid)
  for f in files:
   rec=json.loads(f.read_text());t=byid[f.stem];obj=rec['result'];assert rec['task']==t and rec['rubric_sha256']==rh and rec['requested_model']==cfg['model'] and rec['returned_model'] in cfg['aliases'];run.validate(obj,t)
   body=run.body_for(cfg,t); variants={run.sha(run.canon(dict(body,max_tokens=n))):n for n in (1400,3000,6000)}; assert rec['request_sha256'] in variants;limits[(provider,variants[rec['request_sha256']])]+=1
   matches=[]
   for path in (EXP/'attempts').glob(provider+'-'+f.stem+'-*.json'):
    a=json.loads(path.read_text()); resp=a.get('response') or {}; choices=resp.get('choices') or []
    if a.get('state')=='completed' and a.get('request_sha256')==rec['request_sha256'] and resp.get('model')==rec['returned_model'] and choices:
     content=choices[0]['message']['content']; assert json.loads(content)==obj; matches.append(path.name);contenthash[f.stem].add(hashlib.sha256(content.encode()).hexdigest())
   assert len(matches)==1,(provider,f.stem,matches)
   results[(provider,f.stem)]=obj; provenance[(provider,f.stem)]=matches[0]
   baseline=next(b for b in ('B3','B5') if run.sha(run.canon([t['query_id'],b]))[:24]==f.stem);app[(provider,t['question_type'],baseline,obj['faithfulness_status'])]+=1
   if obj['faithfulness'] is None and not obj['faithfulness_rationale']:
    not_app.append({'provider':provider,'evaluation_id':f.stem,'query_id':t['query_id'],'configuration':baseline,'question_type':t['question_type'],'answer':t['generated_answer'],'faithfulness_status':obj['faithfulness_status']})
 assert all(len(v)==2 for v in contenthash.values()),'identical raw provider content'
 # Independent implementation of the preexisting 5,000-resample estimator.
 contrasts={}; agreed={}; applicability={};negative=[]
 for provider in run.MODELS:
  for typ,metrics in [('answerable',['semantic_accuracy','faithfulness','answer_relevance']),('designated_negative',['negative_safety','faithfulness','answer_relevance'])]:
   for metric in metrics:
    pairs=[]
    for row in sorted(rows,key=lambda r:query_order[provider].index(r['query_id'])):
     if ('answerable' if row['answerable'] else 'designated_negative')!=typ:continue
     ids=[run.sha(run.canon([row['query_id'],b]))[:24] for b in ('B3','B5')]; a,b=[results[(provider,i)][metric] for i in ids]
     if a is not None and b is not None:pairs.append((a,b))
    diff=np.array([a-b for a,b in pairs]);rng=np.random.default_rng(20261106);est=[np.mean(diff[rng.integers(0,len(diff),len(diff))]) for _ in range(5000)] if len(diff)>=2 else []
    contrasts[f'{provider}/{typ}/{metric}']={'n':len(diff),'b3':float(np.mean([a for a,b in pairs])) if pairs else None,'b5':float(np.mean([b for a,b in pairs])) if pairs else None,'diff':float(np.mean(diff)) if pairs else None,'ci95':np.quantile(est,[.025,.975]).tolist() if est else [None,None]}
 for typ,metrics in [('answerable',['semantic_accuracy','faithfulness','answer_relevance']),('designated_negative',['negative_safety','faithfulness','answer_relevance'])]:
  for metric in metrics:
   for scope in ('B3','B5','pooled','paired_difference'):
    pairs=[]; counts=collections.Counter()
    for row in rows:
     if ('answerable' if row['answerable'] else 'designated_negative')!=typ:continue
     per=[]
     for b in ('B3','B5'):
      eid=run.sha(run.canon([row['query_id'],b]))[:24];a=results[('omniroute',eid)][metric];c=results[('openrouter',eid)][metric]
      if scope=='pooled':counts[('present' if a is not None else 'null')+'/'+('present' if c is not None else 'null')]+=1
      if metric=='negative_safety' and scope=='pooled':negative.append({'query_id':row['query_id'],'baseline':b,'evaluation_id':eid,'luna':a,'deepseek':c,'luna_rationale':results[('omniroute',eid)]['negative_safety_rationale'],'deepseek_rationale':results[('openrouter',eid)]['negative_safety_rationale'],'luna_attempt':provenance[('omniroute',eid)],'deepseek_attempt':provenance[('openrouter',eid)]})
      if scope in (b,'pooled') and a is not None and c is not None:pairs.append((row['query_id'],a,c))
      per.append((a,c))
     if scope=='paired_difference' and all(v is not None for pair in per for v in pair):pairs.append((row['query_id'],per[0][0]-per[1][0],per[0][1]-per[1][1]))
    # kappa only makes sense for original ordinal levels, not differences
    agreed[f'{typ}/{metric}/{scope}']=agreement(pairs,metric if scope!='paired_difference' else '')
    if scope=='pooled':applicability[f'{typ}/{metric}']={'/'.join(k) if isinstance(k,tuple) else k:v for k,v in counts.items()}
 # Compare all prior exported contrasts and pooled agreement, without overwriting exports.
 old=json.loads((EXP/'analysis/summary.json').read_text());checks=[]
 for k,v in contrasts.items():
  p,t,m=k.split('/');prior=old['judges'][p]['contrasts'][t+'_'+m]
  for oldkey,newkey in [('paired_n','n'),('b3_mean','b3'),('b5_mean','b5'),('difference','diff')]:
   a,b=prior[oldkey],v[newkey];assert a==b or a is not None and b is not None and abs(a-b)<1e-12,(k,oldkey,a,b)
  for a,b in zip(prior['ci95'],v['ci95']):assert a==b or a is not None and b is not None and abs(a-b)<1e-12,(k,'ci',a,b)
  checks.append(k)
 for k,v in old['agreement'].items():
  typ='designated_negative' if k.startswith('designated_negative_') else 'answerable';metric=k[len(typ)+1:];new=agreed[f'{typ}/{metric}/pooled']
  for a,b in [('matched_n','n'),('mean_absolute_difference','mae'),('spearman','spearman'),('icc_absolute','icc'),('weighted_kappa','weighted_kappa')]:
   x,y=v[a],new[b]; assert x==y or x is not None and y is not None and abs(x-y)<1e-10,(k,a,x,y)
 output={'accepted':len(results),'excluded_accepted_records':0,'rubric_sha256':rh,'request_limits':{f'{k[0]}/{k[1]}':v for k,v in limits.items()},'raw_provenance_distinct_content_pairs':sum(len(v)==2 for v in contenthash.values()),'contrasts':contrasts,'recomputed_prior_contrasts':checks,'agreement':agreed,'applicability_agreement':applicability,'faithfulness_applicability':{'/'.join(k):v for k,v in app.items()},'negative_pairs':negative,'missing_rationale_cases':not_app,'note':'Evidence support annotations are separate manual review, not score changes.'}
 (HERE/'audit_results.json').write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
 print('accepted',len(results),'missing rationale',len(not_app),'negative observations',len(negative))
if __name__=='__main__':main()
