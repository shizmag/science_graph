"""Canonical serialization and append-only experiment artifacts."""
import hashlib
import json
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
ROOT = HOME.parents[1]
VERSION = 'reviewer1-full-evidence-v1'

def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')

def sha(data):
    return hashlib.sha256(data).hexdigest()

def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def frozen_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Refusing to overwrite differing artifact: ' + str(path))
    else:
        with path.open('xb') as f:
            f.write(data)
            f.flush()
            import os
            os.fsync(f.fileno())

def write_json(path, obj):
    frozen_write(path, encode(obj) + b'\n')

def write_rows(path, records):
    frozen_write(path, b''.join(encode(r) + b'\n' for r in records))

def verify_freeze(home=HOME):
    manifest = json.loads((home/'freeze.json').read_text())
    for name, digest in manifest['files'].items():
        if sha((home/name).read_bytes()) != digest:
            raise ValueError('Frozen file changed: ' + name)
    return sha(encode(manifest))
