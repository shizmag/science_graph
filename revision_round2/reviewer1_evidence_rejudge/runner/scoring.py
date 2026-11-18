"""Validate structured evidence assessments; deterministic, judge-independent scores."""
import json
import re
import jsonschema
from common import HOME

WEIGHT = {'full':1., 'partial':.5, 'unsupported':0., 'unassessable':0.}
SCHEMA = json.loads((HOME/'judge_protocol/schema_v1.json').read_text())

def validate(result, task):
    jsonschema.Draft202012Validator(SCHEMA).validate(result)
    visible = task['visible']
    if result['evaluation_id'] != visible['evaluation_id']:
        raise ValueError('evaluation identity mismatch')
    answer, evidence = visible['candidate_answer'], visible['generator_visible_evidence']
    ids = [c['id'] for c in result['claims']]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate claim IDs')
    for c in result['claims']:
        if c['quote'] not in answer or c['evidence_quote'] not in evidence:
            raise ValueError('invented claim/evidence quote')
        if c['support'] != 'unsupported' and not c['evidence_quote'].strip():
            raise ValueError('support requires evidence quote')
    for c in result['citations']:
        if c['claim_id'] not in ids or c['quote'] not in answer:
            raise ValueError('citation claim/quote mismatch')
        if c['resolution'] != 'resolved' and c['support'] != 'unassessable':
            raise ValueError('unresolved citation must be unassessable')
        if c['resolution'] == 'resolved':
            if c['support'] == 'unassessable' or not c['source_quote'].strip() or c['source_quote'] not in evidence:
                raise ValueError('resolved citation lacks source evidence')
            block = re.fullmatch(r'\[?Block\s+(\d+)\]?', c['target'], flags=re.I)
            if block:
                n = block.group(1)
                if not re.search(r'(?:<\|source_id\|>\s*'+n+r'\b|\bBlock\s+'+n+r'\b)', evidence):
                    raise ValueError('inferred Block mapping forbidden')
            elif c['target'] not in evidence:
                raise ValueError('target not explicitly visible')
    return result

def scores(result, empty=False):
    claims, cites = result['claims'], result['citations']
    base = dict(n_claims=len(claims), n_citations=len(cites), n_resolved=sum(c['resolution']=='resolved' for c in cites),
        n_unresolved=sum(c['resolution']=='unresolved' for c in cites), n_ambiguous=sum(c['resolution']=='ambiguous' for c in cites),
        n_incorrect=sum(c['resolution']=='resolved' and c['support']=='unsupported' for c in cites))
    if empty or not claims:
        return dict(base,F=None,CF=None,CF_no_citation_na=None,CF_resolved_only=None,coverage=None,resolvability=None,compliance=None,
            F_status='empty_output' if empty else 'not_applicable_no_factual_claims',CF_status='empty_output' if empty else 'not_applicable_no_factual_claims')
    weights = sum(WEIGHT[c['support']] for c in cites)
    covered = {c['claim_id'] for c in cites if c['resolution']=='resolved' and c['support'] in ('full','partial')}
    return dict(base,F=sum(WEIGHT[c['support']] for c in claims)/len(claims),
        CF=weights/len(cites) if cites else 0., CF_no_citation_na=weights/len(cites) if cites else None,
        CF_resolved_only=weights/base['n_resolved'] if base['n_resolved'] else None,
        coverage=len(covered)/len(claims),resolvability=base['n_resolved']/len(cites) if cites else None,
        compliance=float(bool(cites)),F_status='scored',CF_status='scored' if cites else 'missing_required_citations')
