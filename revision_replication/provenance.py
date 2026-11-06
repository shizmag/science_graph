"""Strict run identity and metadata-only local Gemma inspection."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

GEMMA_PATH = Path("/Volumes/happy-disk/models/llm/gemma-3-text-4b-it-4bit")
REQUIRED_PROVENANCE = (
    "query_registry_sha256", "dataset_id", "dataset_sha256", "corpus_id", "corpus_sha256",
    "retrieval_config_sha256", "context_origin", "generator_id", "generator_revision",
    "tokenizer_revision", "quantization", "prompt_template_sha256", "generation_settings",
    "entropy_definition", "evaluation_protocol_id",
)


@dataclass(frozen=True)
class ReplicationProvenance:
    query_registry_sha256: str
    dataset_id: str
    dataset_sha256: str
    corpus_id: str
    corpus_sha256: str
    retrieval_config_sha256: str
    context_origin: str
    generator_id: str
    generator_revision: str
    tokenizer_revision: str
    quantization: str
    prompt_template_sha256: str
    generation_settings: dict
    entropy_definition: str
    evaluation_protocol_id: str
    paper_to_code: dict

    def validate(self) -> None:
        missing = [key for key in REQUIRED_PROVENANCE if not getattr(self, key)]
        if missing:
            raise ValueError(f"missing provenance fields: {', '.join(missing)}")
        if self.paper_to_code != {"B3": "B4", "B5": "B6"}:
            raise ValueError("paper/code baseline mapping must be explicitly B3->B4 and B5->B6")
        if "occ-rag-1.7b" in self.generator_id.lower():
            raise ValueError("independent replication must not reuse the historical OCC-RAG-1.7B generator")
        if "cuda" in self.context_origin.lower():
            raise ValueError("CUDA archive contexts cannot be silently represented as historical MLX B3/B5 contexts")

    def as_dict(self) -> dict:
        self.validate()
        return asdict(self)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inspect_gemma_metadata(model_dir: Path = GEMMA_PATH) -> dict:
    """Inspect config/tokenizer/index metadata only. Never opens weight tensors."""
    model_dir = Path(model_dir)
    required = ["config.json", "model.safetensors.index.json", "tokenizer_config.json", "tokenizer.json", "tokenizer.model", "model.safetensors"]
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Gemma model directory missing required files: {missing}")
    config = json.loads((model_dir / "config.json").read_text())
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    tokenizer = json.loads((model_dir / "tokenizer_config.json").read_text())
    if config.get("model_type") != "gemma3" or config.get("quantization_config", {}).get("bits") != 4:
        raise ValueError("Expected a 4-bit Gemma 3 model directory")
    if not tokenizer.get("chat_template"):
        raise ValueError("tokenizer_config.json does not provide a chat template")
    weights = model_dir / "model.safetensors"
    return {
        "model_path": str(model_dir), "model_type": config["model_type"],
        "architectures": config.get("architectures"), "quantization": config.get("quantization_config"),
        "total_weight_bytes_metadata": index.get("metadata", {}).get("total_size"),
        "weight_file_bytes": weights.stat().st_size,
        "tensor_count": len(index.get("weight_map", {})),
        "text_config": config.get("text_config"),
        "eos_token_id": config.get("eos_token_id"),
        "chat_template_present": bool(tokenizer.get("chat_template")),
        "tokenizer_model_bytes": (model_dir / "tokenizer.model").stat().st_size,
        "metadata_fingerprints": {n: _sha256(model_dir / n) for n in ("config.json", "model.safetensors.index.json", "tokenizer_config.json", "tokenizer.json", "tokenizer.model")},
        "weights_loaded": False,
        "compatible_by_metadata": True,
        "runtime_compatibility_unverified": True,
    }
