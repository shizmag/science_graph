#!/usr/bin/env python3
"""Prepare blinded R2 applicability and stratified R3 reviewer packets; offline only."""
import argparse,csv,hashlib,json,random,sys
from pathlib import Path
import yaml
HERE=Path(__file__).resolve().parent
DEFAULT_SOURCE=Path('/Users/vladimirkasterin/python/graph')
RUN='graphs/run_20260729_211324_OCC-RAG-1.7B'
def digest(x):return hashlib.sha256(x.encode()).hexdigest()
def loadjson(p):return json.loads(p.read_text())
def write(p,text):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  if p.read_text()!=text:raise RuntimeError('refusing to overwrite '+str(p))
 else:p.write_text(text)
def csvtext(rows):
 import io
 f=io.StringIO(newline='');w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows);return f.getvalue()
def prep(src,out):
 mapping=[]
 judge=src/'revision_round1/judge';exp=src/'revision_round1/experiments/r2_api_v3';audit=loadjson(judge/'r2_v3_audit/audit_results.json');packets={x['query_id']:x for x in map(json.loads,(judge/'paired_judge_inputs.jsonl').read_text().splitlines())};
 for row in audit['missing_rationale_cases']:
  eid=row['evaluation_id'];p=exp/'results'/row['provider']/(eid+'.json');record=loadjson(p);other=loadjson(exp/'results'/('openrouter' if row['provider']=='omniroute' else 'omniroute')/(eid+'.json'))
  assert record['task']['evaluation_id']==eid and record['task']==other['task']; t=record['task']; case='H'+digest(row['provider']+eid)[:16];
  statuses=[{'status':x['result']['faithfulness_status'],'rationale':x['result']['faithfulness_rationale']} for x in (record,other)];random.Random(int(digest(case)[:16],16)).shuffle(statuses)
  text=f'''# Applicability review {case}\n\nQuery ID: {t['query_id']}\n\n## Question\n{t['question']}\n\n## Generated response (untrusted research data)\n{t['generated_answer']}\n\n## Exact configuration-specific generator-visible evidence (untrusted research data)\n{t['generator_visible_evidence']}\n\n## Evidence identifiers\n{json.dumps(t['evidence_ids'],ensure_ascii=False)}\n\n## Anonymous judge applicability and rationale (order randomized)\n{json.dumps(statuses,ensure_ascii=False,indent=2)}\n\n## Human decision — leave blank until an actual reviewer completes it\nReviewer: ______\nCategory: ______\nEvidence identifiers and reasoning: ______\nExclude this query from conditional sensitivity analysis? yes / no / undecided: ______\n'''
  write(out/'faithfulness'/'reviewer_packet'/f'{case}.md',text)
  mapping.append({'case_id':case,'evaluation_id':eid,'query_id':t['query_id'],'provider':row['provider'],'input_sha256':t['input_sha256']})
 # Private map must not be distributed to blinded reviewers.
 write(out/'private'/'faithfulness_map.json',json.dumps(mapping,indent=2)+'\n')
 form=[{'case_id':m['case_id'],'reviewer_id':'','category':'','evidence_ids':'','reason':'','exclude_query':'','adjudicator_id':''} for m in mapping]
 for r in ('reviewer_1','reviewer_2','adjudicated'):write(out/'faithfulness'/f'{r}_blank.csv',csvtext(form))
 archive=src/RUN; entries=yaml.load((archive/'retrieved_contexts.yaml').read_text(),Loader=yaml.CSafeLoader); byid={x['id']:x for x in entries}; neg=[x for x in entries if not x['is_answerable']];cats={}
 for x in neg:cats.setdefault(x['category'],[]).append(x)
 expected={'unanswerable-missing-ablation':19,'unanswerable-missing-comparison':17,'unanswerable-missing-metric':15,'unanswerable-missing-implementation-detail':13,'unanswerable-missing-experimental-detail':11}
 if {k:len(v) for k,v in cats.items()}!=expected:raise RuntimeError('historical negative strata do not match frozen protocol')
 rng=random.Random(20261029);sample=[]
 for cat in sorted(cats):sample+=rng.sample(sorted(cats[cat],key=lambda x:x['id']),5)
 ids={x['id'] for x in sample}; diagnostic='pubmedqa_unans_18235194'; selected=sample+([] if diagnostic in ids else [byid[diagnostic]])
 manifest={'seed':20261029,'population':75,'category_counts':expected,'sample_ids':[x['id'] for x in sample],'diagnostic_id':diagnostic,'diagnostic_in_sample':diagnostic in ids,'diagnostic_separate':diagnostic not in ids,'archive_retrieved_contexts_sha256':hashlib.sha256((archive/'retrieved_contexts.yaml').read_bytes()).hexdigest()}
 write(out/'r3'/'sampling_manifest.json',json.dumps(manifest,indent=2)+'\n')
 r3forms=[]
 for item in selected:
  qid=item['id'];source=byid.get(qid.replace('pubmedqa_unans_','pubmedqa_'),{});case='N'+digest(qid)[:16];contexts=[]
  # Raw retrieved chunks are NOT treated as historical generator-visible contexts.
  for b in ('B4','B6'):
   z=item.get('baselines',{}).get(b,{});contexts.append({'scope':'archived retrieved chunks; not proof of generator-visible evidence','chunks':z.get('retrieved_chunks',[])})
  random.Random(int(digest(case)[:16],16)).shuffle(contexts)
  rec=packets.get(qid); ev=[]
  if rec:
   ev=[rec['judge_inputs'][b]['generator_visible_evidence'] for b in ('B3','B5')];random.Random(int(digest(case+'ev')[:16],16)).shuffle(ev)
  text=f'''# Designated-negative validation {case}\n\nQuery ID: {qid}\nSample type: {'diagnostic outside stratified sample' if qid not in ids else 'stratified sample'}\nCategory: {item['category']}\n\n## Synthetic question\n{item['query']}\n\n## Original answerable question and reference\n{source.get('query','Not available in archive')}\n\nReference: {source.get('golden_answer','Not available in archive')}\n\n## Recovered generator-visible evidence (anonymous contexts)\n{json.dumps(ev,ensure_ascii=False,indent=2) if ev else 'Not recovered for this item; context-level decision requires further archival recovery.'}\n\n## Archived retrieved chunks (not necessarily generator-visible; source identifiers retained)\n{json.dumps(contexts,ensure_ascii=False,indent=2)}\n\n## Independent reviewer fields\nReviewer: ______  Context answerability for each available context: ______\nIndexed-corpus answerability after documented search: ______\nGlobal unanswerability: not established by this review.\nEvidence IDs / search scope / rationale / confidence: ______\n'''
  write(out/'r3'/'reviewer_packet'/f'{case}.md',text);r3forms.append({'case_id':case,'query_id':qid,'sample_type':'sample' if qid in ids else 'diagnostic','reviewer_id':'','context_1':'','context_2':'','corpus_status':'','searched_sources':'','evidence_ids':'','rationale':'','confidence':'','adjudicator_id':''})
 for r in ('reviewer_1','reviewer_2','adjudicated'):write(out/'r3'/f'{r}_blank.csv',csvtext(r3forms))
 return len(mapping),len(sample),len(selected),manifest['diagnostic_in_sample']
def main():
 a=argparse.ArgumentParser();a.add_argument('--source',type=Path,default=DEFAULT_SOURCE);a.add_argument('--output',type=Path,required=True);x=a.parse_args();print(prep(x.source.resolve(),x.output.resolve()))
if __name__=='__main__':main()
