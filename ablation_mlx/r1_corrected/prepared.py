"""CPU-only immutable inputs shared by preparation and MLX generation."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib, json, os, tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping
from .protocol import ARM_OFF, ARM_ON, INPUT_TOKEN_LIMIT
from .identity import require_manifest, object_sha256

SCHEMA = "r1-prepared-input-v1"
class PreparedManifestMismatch(ValueError):
    """A prepared artifact belongs to a different frozen experiment."""
def _json(x: Any) -> bytes: return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
def digest(x: Any) -> str: return hashlib.sha256(_json(x)).hexdigest()

@dataclass(frozen=True)
class PreparedInput:
    query_id: str; arm_id: str; prompt: str; token_ids: tuple[int, ...]; prompt_token_count: int
    question: str = ""; evidence_ids: tuple[str, ...] = (); activation: bool = False
    static_relation_count: int = 0; status: str = "prepared"; failure_reason: str | None = None
    prompt_hash: str = ""; token_ids_hash: str = ""
    manifest_hash: str = ""
    evidence_identity: str = ""
    @property
    def input_hash(self) -> str: return digest({"schema": SCHEMA, **asdict(self)})
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self); d.update(schema=SCHEMA, token_ids=list(self.token_ids), evidence_ids=list(self.evidence_ids), input_hash=self.input_hash); return d
    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PreparedInput":
        if d.get("schema") != SCHEMA: raise ValueError("unsupported prepared input schema")
        x = dict(d); supplied=x.pop("input_hash", None); x.pop("schema", None); x["token_ids"]=tuple(x["token_ids"]); x["evidence_ids"]=tuple(x.get("evidence_ids", ()))
        try: item=cls(**x)
        except TypeError as exc: raise ValueError("invalid prepared input fields") from exc
        if supplied != item.input_hash: raise ValueError("prepared input hash mismatch")
        if item.prompt_hash != hashlib.sha256(item.prompt.encode("utf-8")).hexdigest(): raise ValueError("prepared prompt hash mismatch")
        if item.token_ids_hash != digest(item.token_ids) or item.prompt_token_count != len(item.token_ids): raise ValueError("prepared token IDs mismatch")
        return item

def write(path: Path, item: PreparedInput) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd,tmp=tempfile.mkstemp(dir=path.parent, prefix=".prepared-")
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f: json.dump(item.to_dict(),f,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        # A competing process must never replace an already published artifact.
        os.link(tmp,path)
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
def read(path: Path) -> PreparedInput: return PreparedInput.from_dict(json.loads(path.read_text(encoding="utf-8")))
def make(query_id: str, arm_id: str, prompt: str, token_ids: Iterable[int], **kwargs: Any) -> PreparedInput:
    if arm_id not in (ARM_ON, ARM_OFF): raise ValueError(f"unknown arm: {arm_id}")
    ids=tuple(int(x) for x in token_ids); eids=tuple(kwargs.pop("evidence_ids", ()))
    return PreparedInput(query_id,arm_id,prompt,ids,len(ids),evidence_ids=eids,prompt_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),token_ids_hash=digest(ids),**kwargs)
def prepare_all(items: Iterable[tuple[str,str,Any]], root: Path, *, manifest_path: Path, manifest_hash: str, cohort: Mapping[str, Any], expected: Mapping[tuple[str, str], str] | None = None) -> dict[str,int]:
    manifest = require_manifest(manifest_path, manifest_hash)
    cohort_hash = object_sha256({key: value for key, value in cohort.items() if key != "cohort_hash"})
    if cohort.get("cohort_hash") != cohort_hash or manifest.get("cohort_hash") != cohort_hash:
        raise ValueError("frozen cohort identity mismatch")
    arms, rows = manifest.get("arms"), cohort.get("rows")
    if not isinstance(rows, list) or not isinstance(arms, list) or not arms or any(a not in (ARM_ON, ARM_OFF) for a in arms):
        raise ValueError("invalid frozen cohort or arms")
    members = {(row["query_id"], arm) for row in rows for arm in arms}
    out={"prepared":0,"failed":0,"skipped":0}
    for q,a,factory in items:
        if (q, a) not in members: raise ValueError("slot is not in frozen cohort")
        path=root/f"{q}__{a}.json"
        if path.exists():
            try:
                existing = read(path)
                if existing.manifest_hash != manifest_hash:
                    raise PreparedManifestMismatch(f"prepared input {path} belongs to a different manifest")
                if existing.query_id != q or existing.arm_id != a:
                    raise ValueError(f"prepared input {path} belongs to a different slot")
                if expected and existing.input_hash != expected.get((q, a), existing.input_hash):
                    raise ValueError(f"prepared input {path} does not match the frozen expected identity")
                out["skipped"]+=1; continue
            except ValueError:
                raise
            except (OSError,TypeError,KeyError) as exc:
                raise ValueError(f"corrupt prepared input {path}: {exc}") from exc
        try:
            item=factory()
            if item.prompt_token_count>INPUT_TOKEN_LIMIT: raise ValueError(f"prompt exceeds {INPUT_TOKEN_LIMIT} tokens")
        except Exception as exc:
            item=make(q,a,"",(),status="failed",manifest_hash=manifest_hash,
                      failure_reason=f"{type(exc).__name__}: {exc}")
        if item.query_id != q or item.arm_id != a or item.manifest_hash != manifest_hash:
            raise ValueError("factory returned an input for the wrong frozen manifest or slot")
        write(path,item)
        out["failed" if item.status == "failed" else "prepared"]+=1
    return out
