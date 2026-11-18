"""Saved-data-only analysis; query-cluster bootstrap, never calls an LLM."""
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0,str(Path(__file__).resolve().parent/'runner'))
from common import HOME, rows, verify_freeze
from scoring import validate, scores
from run import validate_checkpoint,body_for,infrastructure
from common import encode,sha
SEED=20261116
NBOOT=10000


def icc(a,b):
    x=np.column_stack([a,b]);n=len(x);k=2
    if n<2:return None
    grand=x.mean(); msr=k*np.sum((x.mean(1)-grand)**2)/(n-1)
    msc=n*np.sum((x.mean(0)-grand)**2)/(k-1)
    residual=x-x.mean(1)[:,None]-x.mean(0)[None,:]+grand
    mse=np.sum(residual**2)/((n-1)*(k-1))
    den=msr+(k-1)*mse+k*(msc-mse)/n
    return float((msr-mse)/den) if den else None


def agreement(x):
    a,b=x[:,0],x[:,1]
    rho=float(spearmanr(a,b).statistic) if np.std(a)>0 and np.std(b)>0 else None
    return dict(ICC_A1=icc(a,b),Spearman=rho,judge_mean_difference=float(np.mean(a-b)),mean_absolute_difference=float(np.mean(np.abs(a-b))))


def ci(values):
    finite=[v for v in values if v is not None and np.isfinite(v)]
    return dict(interval=np.quantile(finite,[.025,.975]).tolist() if finite else None,valid_resamples=len(finite))


def analyze(home=HOME,nboot=NBOOT):
    protocol_hash=verify_freeze(home)
    tasks=rows(home/'judge_tasks.jsonl');data={};status={}
    for t in tasks:
        p=home/'accepted'/(t['task_id']+'.json')
        if not p.exists():
            status[t['task_id']]=t['status'] if t['status']!='ready' else 'judge_missing'
            continue
        r=json.loads(p.read_text()); validate_checkpoint(r,t,home,protocol_hash,sha(encode(body_for(t,infrastructure.MODELS[t['judge']]))))
        assert r['task']==t and r['scores']==scores(r['parsed'])
        data[t['query_id'],t['paper_configuration'],t['judge']]=r['scores']
        status[t['task_id']]='accepted'
    ids=sorted({t['query_id'] for t in tasks});judges=['omniroute','openrouter'];papers=['B3','B5']
    out=dict(status='PENDING RESULTS' if len(data)<len(tasks) else 'completed',planned=len(tasks),accepted=len(data),
        status_counts=dict(Counter(status.values())),generation_status_counts=dict(Counter(str(r['generation_status']) for r in rows(home/'evidence_manifest.jsonl'))),seed=SEED,bootstrap_resamples=nboot,metrics={},citation_diagnostics={})
    rng=np.random.default_rng(SEED)
    for metric in ('F','CF','coverage','resolvability','compliance','CF_no_citation_na','CF_resolved_only'):
        available={j:[q for q in ids if all(data.get((q,p,j),{}).get(metric) is not None for p in papers)] for j in judges}
        common=sorted(set(available[judges[0]])&set(available[judges[1]]))
        result=dict(common_ids=common,common_n=len(common),judge_available_ids=available,excluded={j:{q:[p+':'+(data.get((q,p,j),{}).get(metric+'_status','N/A') if (q,p,j) in data else status[next(t['task_id'] for t in tasks if t['query_id']==q and t['paper_configuration']==p and t['judge']==j)]) for p in papers if data.get((q,p,j),{}).get(metric) is None] for q in ids if q not in available[j]} for j in judges},contrasts={})
        for label,groups in [('primary_common',{j:common for j in judges}),('judge_available',available)]:
            result['contrasts'][label]={}
            for j,qs in groups.items():
                x=np.array([[data[q,p,j][metric] for p in papers] for q in qs])
                if not len(qs):result['contrasts'][label][j]={'n':0};continue
                d=x[:,0]-x[:,1];samples=rng.integers(0,len(qs),(nboot,len(qs)))
                result['contrasts'][label][j]=dict(n=len(qs),B3_mean=float(x[:,0].mean()),B5_mean=float(x[:,1].mean()),difference=float(d.mean()),CI95=ci(d[samples].mean(1)))
        if common:
            x=np.array([[[data[q,p,j][metric] for j in judges] for p in papers] for q in common])
            # Both configurations stay together inside each query bootstrap draw.
            draws=rng.integers(0,len(common),(nboot,len(common)))
            for label,arr in [('individual_responses',x),('paired_differences',x[:,0,:]-x[:,1,:])]:
                point=agreement(arr.reshape(-1,2));boots=[agreement(arr[idx].reshape(-1,2)) for idx in draws]
                result[label]=dict(point=point,CI95={k:ci([r[k] for r in boots]) for k in point})
        statuses=[]
        for q in ids:
            for p in papers:
                a,b=(data.get((q,p,j)) for j in judges)
                if a is not None and b is not None:statuses.append((a.get(metric) is not None,b.get(metric) is not None))
        result['applicability_agreement']=dict(n=len(statuses),fraction=float(np.mean([a==b for a,b in statuses])) if statuses else None)
        applicable_clusters=[]
        for q in ids:
            comparisons=[]
            for p in papers:
                a,b=(data.get((q,p,j)) for j in judges)
                if a is not None and b is not None:comparisons.append(float((a.get(metric) is not None)==(b.get(metric) is not None)))
            if comparisons:applicable_clusters.append(comparisons)
        if applicable_clusters:
            boot=[]
            for idx in rng.integers(0,len(applicable_clusters),(nboot,len(applicable_clusters))):
                boot.append(float(np.mean([v for i in idx for v in applicable_clusters[i]])))
            result['applicability_agreement']['CI95']=ci(boot)
        out['metrics'][metric]=result
    for j in judges:
        for p in papers:
            ss=[v for (q,pp,jj),v in data.items() if pp==p and jj==j];n=len(ss);nc=sum(s['n_citations'] for s in ss)
            out['citation_diagnostics'][j+'/'+p]=dict(n=n,no_citations=sum(s['n_citations']==0 for s in ss),
                no_citation_percentage=100*sum(s['n_citations']==0 for s in ss)/n if n else None,
                missing_required=sum(s['CF_status']=='missing_required_citations' for s in ss),
                F_NA=sum(s['F'] is None for s in ss),CF_NA=sum(s['CF'] is None for s in ss),
                citations=nc,resolved=sum(s['n_resolved'] for s in ss),incorrect=sum(s['n_incorrect'] for s in ss),
                unresolved=sum(s['n_unresolved'] for s in ss),ambiguous=sum(s['n_ambiguous'] for s in ss),
                incorrect_frequency=sum(s['n_incorrect'] for s in ss)/nc if nc else None,
                resolvable_fraction=sum(s['n_resolved'] for s in ss)/nc if nc else None)
    (home/'analysis_results').mkdir(exist_ok=True)
    (home/'analysis_results/summary.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    return out

if __name__=='__main__':
    result=analyze();print(json.dumps({k:result[k] for k in ('status','planned','accepted')}))
