"""Phase A: reuse recovery contract, verify R2 identity, freeze inputs (offline)."""
import importlib.util
import json
import re
import sys
from collections import Counter
from common import HOME, ROOT, VERSION, encode, sha, rows, write_json, write_rows, frozen_write
sys.path.insert(0, str(ROOT/'revision_round1/judge'))
import historical_evidence_recovery as recovery
import r2_api_runner_v3 as previous


def render(case, paper, templates):
    code = recovery.PAPER_TO_CODE[paper]
    b = case['baselines'][code]
    enrichment = b.get('enrichment_block') or ''
    branch = 'ask_expander' if paper == 'B5' and enrichment and enrichment != 'No essential knowledge graph enrichment found.' else 'ask_no_expander'
    text = templates[branch]
    values = dict(enrichment_block=enrichment, context_text=b.get('trimmed_text') or '', context_graph=b.get('trimmed_graph') or '', history_str='', query=case['query'])
    # Match historical manager replacement order for each branch.
    keys = ('enrichment_block','history_str','query') if branch == 'ask_expander' else ('context_text','context_graph','history_str','query')
    for k in keys:
        text = re.sub(r'\{\{\s*'+k+r'\s*\}\}', lambda m: str(values[k]), text)
    return branch, text.strip()


def main():
    run = recovery.DEFAULT_RUN
    recovered, contract = recovery.load_historical_records(run, ROOT)
    packet = previous.cohort()  # verifies original frozen packet hashes before reading
    selected = [r for r in packet if r['answerable']]
    assert len(selected) == 80 and len({r['query_id'] for r in selected}) == 80
    regenerated = recovery.choose_stratified_cohort(recovered)
    assert set(regenerated) == {r['query_id'] for r in packet}, 'R2 selection reproduction failed'
    contexts = {r['id']:r for r in recovery._yaml(run/'retrieved_contexts.yaml')}
    historical = {(r['query_id'], r['paper_baseline']):r for r in recovered}
    templates = {b:recovery._git_blob(ROOT, contract['source_commit'], 'back/src/prompts/rag/'+b+'.txt')[0] for b in ('ask_expander','ask_no_expander')}
    # Read original raw generations rather than treating extracted answer as raw.
    raw = {r['id']:r for r in recovery._yaml(run/'evaluation_results.yaml')['results']}
    fixture_path = ROOT/'revision_closure/phase_02_entropy/corrected_prompt_replay/prompt_manifest.jsonl'
    fixture = rows(fixture_path)
    for f in fixture:
        branch, prompt = render(contexts[f['query_id']], f['paper_baseline'], templates)
        assert branch == f['prompt_type'] and sha(prompt.encode()) == f['prompt_sha256'] and prompt == f['prompt']
    records, prompts, tasks = [], [], []
    for r in selected:
        q = r['query_id']
        assert contexts[q]['query'] == r['query']
        for paper, code in recovery.PAPER_TO_CODE.items():
            h = historical[q,paper]; x = r['judge_inputs'][paper]; b = contexts[q]['baselines'][code]
            assert h['evidence_classification'] == 'EXACT', (q, paper, 'evidence not proven')
            assert h['generated_answer'] == x['candidate_answer']
            assert h['generator_visible_evidence'] == x['generator_visible_evidence']
            assert h['generator_visible_evidence_sha256'] == x['evidence_sha256']
            branch, prompt = render(contexts[q], paper, templates)
            for _ in range(3):
                assert render(contexts[q], paper, templates) == (branch, prompt)
            components = ['enrichment_block'] if branch == 'ask_expander' else ['trimmed_text','trimmed_graph']
            assert components == h['evidence_components']
            assert all((b[k] or '') in prompt for k in components)
            # Gold is never an argument to render or judge-visible whitelist; natural overlap is not leakage.
            source = raw[q]['baselines'][code]
            assert raw[q]['query'] == r['query'] and source.get('generated_answer') is not None
            eid = sha(encode([VERSION,q,paper]))[:24]
            fields = {k:b.get(k) for k in ('context_text','context_graph','trimmed_text','trimmed_graph','enrichment_block')}
            record = dict(query_id=q, paper_configuration=paper, code_configuration=code,
                question=r['query'], reference_answer=r['reference_answer'],
                raw_generated_answer=source['generated_answer'], cleaned_answer=x['candidate_answer'],
                generation_status=source.get('status'), archived_context_fields=fields, prompt_branch=branch,
                generator_visible_evidence=x['generator_visible_evidence'], reconstructed_full_prompt=prompt,
                source_commit=contract['source_commit'], archive_identity=run.name,
                provenance_status='archived_evidence_verified / historical_model_input_unverified',
                source_artifacts={k:{**v,'path':str(__import__('pathlib').Path(v['path']).relative_to(ROOT))} for k,v in h['source_artifacts'].items()})
            record['sha256'] = {k:sha(v.encode('utf-8') if isinstance(v,str) else encode(v)) for k,v in record.items() if k in ('question','reference_answer','raw_generated_answer','cleaned_answer','archived_context_fields','generator_visible_evidence','reconstructed_full_prompt')}
            record['record_sha256'] = sha(encode(record))
            records.append(record)
            prompts.append(dict(query_id=q,paper_configuration=paper,prompt_branch=branch,prompt_sha256=sha(prompt.encode()),prompt=prompt))
            visible = dict(evaluation_id=eid, question=r['query'], candidate_answer=x['candidate_answer'],
                generator_visible_evidence=x['generator_visible_evidence'],
                source_mapping_policy='Resolve only explicit identifiers present in this evidence and their local source spans. Repeated Page or Chunk labels are not unique. Never map Block N to retrieved-chunk order.',
                visible_identifiers=x['citation_metadata'])
            assert 'reference_answer' not in visible and 'paper_configuration' not in visible
            for judge in ('omniroute','openrouter'):
                tasks.append(dict(task_id=judge+'-'+eid, query_id=q, paper_configuration=paper, code_configuration=code,
                    anonymized_configuration='answer_'+eid, judge=judge, protocol_version=VERSION, visible=visible,
                    visible_sha256=sha(encode(visible)), input_record_sha256=record['record_sha256'], status='ready' if x['candidate_answer'].strip() else 'empty_output'))
    assert len(records)==160 and len(tasks)==320
    write_json(HOME/'cohort_manifest.json', dict(protocol_version=VERSION, query_ids=sorted(r['query_id'] for r in selected),
        selection='Same R2 v3 fixed80; seed 20261106, sorted eligible strata sampled conditional on paired answer and exact archived evidence availability; not independent random sample of all 300 answerable queries.',
        r2_packet_manifest_sha256=sha((ROOT/'revision_round1/judge/r2_packet_manifest.json').read_bytes()), selection_reproduced=True))
    write_rows(HOME/'evidence_manifest.jsonl',records)
    write_rows(HOME/'reconstructed_prompt_manifest.jsonl',prompts)
    write_rows(HOME/'judge_tasks.jsonl',tasks)
    write_json(HOME/'provenance_checks.json',dict(verified_records=160,exceptions=[],corrected_replay_fixtures=len(fixture),fixture_prompt_matches=len(fixture),serialization_repetitions=3,branches=dict(Counter(r['prompt_branch'] for r in records)),source_contract=contract,
        raw_archive_sha256=recovery.sha256_file(run/'evaluation_results.yaml'),fixture_sha256=sha(fixture_path.read_bytes()),gold_policy='Gold stored separately; absent from render arguments and judge-visible whitelist. Naturally shared factual text is not treated as leakage.'))
    print(json.dumps(dict(verified_records=len(records),tasks=len(tasks),fixture_matches=len(fixture))))

if __name__ == '__main__':
    main()
