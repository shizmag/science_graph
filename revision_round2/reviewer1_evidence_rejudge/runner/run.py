"""Versioned independent runner. Dry-run default; explicit authorization required."""
import argparse
import datetime
import json
import os
import sys
import time
import urllib.error
import urllib.request
import jsonschema
from pathlib import Path
from common import HOME, ROOT, VERSION, encode, sha, rows, write_json, verify_freeze
from scoring import validate, scores
sys.path.insert(0,str(ROOT/'revision_round1/judge'))
import r2_api_runner_v3 as infrastructure


def body_for(task, cfg):
    instructions = (HOME/'judge_protocol/instructions_v1.md').read_text()
    schema = json.loads((HOME/'judge_protocol/schema_v1.json').read_text())
    body = dict(model=cfg['model'], temperature=0, max_tokens=12000,
        response_format={'type':'json_object'},messages=[
            {'role':'system','content':instructions+'\nJSON SCHEMA:\n'+json.dumps(schema,sort_keys=True)},
            {'role':'user','content':encode(task['visible']).decode()}])
    if task['judge']=='openrouter':
        body['provider']={'allow_fallbacks':False}
    return body


def transport(url, key, request_bytes):
    req=urllib.request.Request(url.rstrip('/')+'/chat/completions',data=request_bytes,
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=240) as r:
        return r.read(), {k:v for k,v in r.headers.items() if k.lower() in ('x-request-id','request-id','x-provider','x-model','retry-after')}


def validate_checkpoint(r, task, out, protocol_hash, reqhash=None):
    if r['protocol_sha256']!=protocol_hash or r['task']!=task:
        raise ValueError('checkpoint identity/hash mismatch')
    if reqhash is not None and r['request_sha256']!=reqhash:
        raise ValueError('checkpoint request mismatch')
    validate(r['parsed'],task)
    if r['scores'] != scores(r['parsed']):
        raise ValueError('checkpoint score mismatch')
    attempt=json.loads((Path(out)/'attempts'/r['attempt_file']).read_text())
    if (attempt['response_sha256'] != sha(attempt['response_body'].encode())
            or attempt['request_sha256'] != r['request_sha256']
            or sha(attempt['request_utf8'].encode()) != r['request_sha256']
            or attempt['parsed'] != r['parsed'] or attempt['state'] != 'accepted'
            or json.loads(attempt['request_utf8']) != attempt['request_body']):
        raise ValueError('checkpoint attempt corrupted')
    response=json.loads(attempt['response_body'])
    if json.loads(response['choices'][0]['message']['content']) != r['parsed']:
        raise ValueError('checkpoint response differs from parsed score')
    return r


def execute(task, cfg, url, key, out, protocol_hash, send=transport, sleep=time.sleep):
    """Exclusive run lock is held by caller. First valid attempt wins, never rescore."""
    out=Path(out); tid=task['task_id']; body=body_for(task,cfg); request=encode(body)
    reqhash=sha(request); accepted=out/'accepted'/f'{tid}.json'
    if accepted.exists():
        validate_checkpoint(json.loads(accepted.read_text()),task,out,protocol_hash,reqhash)
        return 'accepted'
    if task['status']=='empty_output':
        return 'empty_output'
    previous=sorted((out/'attempts').glob(tid+'-*.json'))
    for p in previous:
        a=json.loads(p.read_text())
        if a['request_sha256']!=reqhash or a['protocol_sha256']!=protocol_hash:
            raise ValueError('attempt identity changed')
        if a['state']=='fatal':
            return 'fatal'
        if a['state']=='accepted':
            # Crash after saved response but before accepted checkpoint: recover without API call.
            parsed=validate(a['parsed'],task)
            write_json(accepted,dict(task=task,parsed=parsed,scores=scores(parsed),request_sha256=reqhash,
                protocol_sha256=protocol_hash,attempt_file=p.name,returned_model=a['returned_model']))
            return 'accepted'
    # A started marker without a finalized attempt is an uncertain request, never silently reissued.
    started=list((out/'started').glob(tid+'-*.json'))
    if len(started)!=len(previous):
        return 'uncertain_attempt_requires_reconciliation'
    for no in range(len(previous)+1,5):
        stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
        name=f'{tid}-{no}.json'
        record=dict(task_id=tid,query_id=task['query_id'],anonymized_configuration=task['anonymized_configuration'],
            judge=task['judge'],protocol_version=VERSION,protocol_sha256=protocol_hash,attempt_number=no,
            request_body=body,request_utf8=request.decode(),request_sha256=reqhash,endpoint=url,
            started_at=stamp,state='started',response_body=None,response_sha256=None,parsed=None,
            validation_status='not_validated',api_error=None)
        write_json(out/'started'/name,record)
        try:
            response,metadata=send(url,key,request)
            response_text=response.decode('utf-8')
            if key and key in response_text:
                raise RuntimeError('credential echoed; raw response withheld for security')
            record.update(response_body=response_text,response_sha256=sha(response),provider_metadata=metadata)
            payload=json.loads(response)
            record['usage']=payload.get('usage');record['returned_model']=payload.get('model')
            if payload.get('model') not in cfg['aliases']:
                raise RuntimeError('Unexpected returned model; stop rather than substitute')
            choice=payload['choices'][0]
            record['finish_reason']=choice.get('finish_reason')
            if choice.get('finish_reason')!='stop':
                raise ValueError('incomplete structured output')
            parsed=json.loads(choice['message']['content']); record['parsed']=parsed
            validate(parsed,task)
            record.update(state='accepted',validation_status='valid')
        except urllib.error.HTTPError as e:
            detail=e.read().decode('utf-8','replace').replace(key,'[REDACTED]')
            record.update(state='retryable' if e.code in (408,429,500,502,503,504) else 'fatal',
                api_error={'type':'HTTPError','status':e.code,'body':detail},response_body=detail,response_sha256=sha(detail.encode()))
        except (urllib.error.URLError,TimeoutError,ConnectionError) as e:
            record.update(state='retryable',api_error={'type':type(e).__name__,'message':str(e).replace(key,'[REDACTED]')})
        except (ValueError,KeyError,TypeError,jsonschema.ValidationError) as e:
            record.update(state='retryable',validation_status='invalid',api_error={'type':type(e).__name__,'message':str(e).replace(key,'[REDACTED]')})
        except Exception as e:
            record.update(state='fatal',api_error={'type':type(e).__name__,'message':str(e).replace(key,'[REDACTED]')})
        record['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(out/'attempts'/name,record)
        if record['state']=='accepted':
            write_json(accepted,dict(task=task,parsed=parsed,scores=scores(parsed),request_sha256=reqhash,
                protocol_sha256=protocol_hash,attempt_file=name,returned_model=record['returned_model']))
            return 'accepted'
        if record['state']=='fatal':
            return 'fatal'
        sleep(min(120,2**no))
    return 'retry_exhausted'


def run(authorize=False):
    protocol_hash=verify_freeze()
    tasks=rows(HOME/'judge_tasks.jsonl')
    if not authorize:
        return dict(mode='offline-dry-run',tasks=len(tasks),protocol_sha256=protocol_hash,
            expected_calls=320,max_attempts=1280,paid_calls_made=0)
    # Reuse R2 model IDs/aliases, credentials and endpoint conventions, not its gold-bearing messages.
    configs={k:dict(v) for k,v in infrastructure.MODELS.items()}
    keys={k:infrastructure.envkey('OMNIROUTE_API_KEY') or infrastructure.envkey('REVISION_JUDGE_API_KEY') if k=='omniroute' else infrastructure.envkey('OPENROUTER_API_KEY') for k in configs}
    if not all(keys.values()):
        raise RuntimeError('Missing judge credentials')
    base=infrastructure.envkey('REVISION_JUDGE_BASE_URL')
    if base not in ('http://localhost:20128/v1','http://127.0.0.1:20128/v1'):
        raise RuntimeError('Unexpected historical OmniRoute endpoint')
    blob=(HOME/'judge_tasks.jsonl').read_text()
    if any(len(v)>=8 and v in blob for v in keys.values()):
        raise RuntimeError('Credential in task payload')
    lock=HOME/'execution.lock'
    # flock releases on crash; stale lock path itself does not block resume.
    import fcntl
    with lock.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status={}
        for task in tasks:
            cfg=configs[task['judge']]
            url=base if task['judge']=='omniroute' else cfg['base_url']
            status[task['task_id']]=execute(task,cfg,url,keys[task['judge']],HOME,protocol_hash)
            # Conservative pacing, including resumed tasks; retries also get exponential backoff.
            time.sleep(60/cfg['rpm']+.2)
        infrastructure.atomic(HOME/'coverage_report.json',dict(planned=320,statuses=status,
            accepted=sum(v=='accepted' for v in status.values()),protocol_sha256=protocol_hash))
        accepted=[json.loads(p.read_text()) for p in sorted((HOME/'accepted').glob('*.json'))]
        # Derived export may be regenerated; individual checkpoints and attempts are immutable.
        (HOME/'accepted_scores.jsonl').write_bytes(b''.join(encode(r)+b'\n' for r in accepted))
        return dict(accepted=len(accepted),planned=320,complete=len(accepted)==320)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--authorize-judge-execution',action='store_true')
    print(json.dumps(run(p.parse_args().authorize_judge_execution),indent=2))
