"""Offline pre-run freeze and security screening (not an automatic public release)."""
import json
import os
import re
import sys
from common import HOME,ROOT,VERSION,encode,sha,write_json,verify_freeze


def main():
    names=['protocol.md','cohort_manifest.json','evidence_manifest.jsonl','reconstructed_prompt_manifest.jsonl','judge_tasks.jsonl','provenance_checks.json','statistical_analysis.py']
    names += [str(p.relative_to(HOME)) for folder in ('runner','judge_protocol','tests') for p in (HOME/folder).glob('*') if p.suffix in ('.py','.json','.md')]
    # Protect imported recovery and R2 infrastructure as dependencies as well.
    dependencies=['revision_round1/judge/historical_evidence_recovery.py','revision_round1/judge/r2_api_runner_v3.py','revision_round1/judge/r2_runner_v2.py']
    names += [os.path.relpath(ROOT/p,HOME) for p in dependencies]
    problems=[]
    secrets=[v for k,v in os.environ.items() if len(v)>=12 and any(w in k.upper() for w in ('API_KEY','SECRET','PASSWORD','TOKEN'))]
    for name in names:
        text=(HOME/name).read_text()
        if re.search(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bsk-[A-Za-z0-9_-]{24,}',text):
            problems.append(name+': possible credential')
        if any(s in text for s in secrets):problems.append(name+': environment credential match')
    if problems:raise ValueError(str(problems))
    write_json(HOME/'freeze.json',dict(protocol_version=VERSION,files={n:sha((HOME/n).read_bytes()) for n in sorted(names)}))
    print(json.dumps(dict(protocol_sha256=verify_freeze(),files=len(names),security='no detected credential patterns or environment-secret matches; manual public-release review still required')))

if __name__=='__main__': main()
