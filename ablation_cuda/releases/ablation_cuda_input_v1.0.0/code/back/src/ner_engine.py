"""
NER Engine — extracts PERSON entities from text using a cached HuggingFace model.
Falls back to regex heuristics if the model is unavailable.
"""

import re
import os
from typing import List, Optional

# Regex to find human name patterns (2-4 capitalized words, allowing initials like "A.")
# Matches patterns like: "Ashish Vaswani", "Aidan N. Gomez", "Lukasz Kaiser"
_NAME_PATTERN = re.compile(
    r'\b([A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-z]+){1,3})\b'
)

# Comprehensive list of lowercase words that cannot appear as parts of human author names.
# Includes organization, institution, location, academic, and scientific terminology.
_FORBIDDEN_NAME_WORDS = {
    # Organizations / Institutions / Places indicators
    "university", "college", "school", "dept", "department", "institute", "laboratory", 
    "labs", "research", "academy", "corporation", "corp", "inc", "co", "ltd", "association", 
    "society", "foundation", "trust", "hospital", "clinic", "medical", "center", "centre", 
    "group", "team", "committee", "board", "commission", "agency", "ministry", "government", 
    "state", "national", "federal", "international", "global", "european", "american", "british",
    "chinese", "russian", "indian", "german", "french", "japanese", "korean", "spanish", "italian",
    "canadian", "australian", "city", "county", "district", "region", "province", "country",
    "san", "st", "new", "york", "california", "london", "paris", "berlin", "tokyo", "beijing", 
    "seoul", "moscow", "boston", "chicago", "seattle", "austin", "oxford", "cambridge", "stanford", 
    "mit", "harvard", "berkeley", "princeton", "yale", "columbia", "cornell", "caltech", "carnegie", 
    "mellon", "santa", "barbara", "urbana", "illinois", "beach", "lake", "river", "mountain", 
    "valley", "hill", "park", "road", "street", "avenue",
    
    # Academic / Publishing terms
    "abstract", "introduction", "related", "conclusion", "references", "bibliography", "figure", 
    "table", "appendix", "method", "methods", "result", "results", "discussion", "acknowledgement",
    "acknowledgements", "funding", "grant", "sponsorship", "journal", "proceedings", "conference", 
    "workshop", "symposium", "volume", "vol", "issue", "no", "pages", "pp", "doi", "arxiv", "biorxiv", 
    "preprint", "paper", "manuscript", "article", "thesis", "dissertation", "publisher", "published", 
    "editor", "editors", "review", "reviewer", "reviewers", "contributor", "contributors", "co-author",
    "author", "authors", "joint", "last", "first", "second", "third", "et", "al", "creative", "commons",
    "world", "health", "organization", "who", "ieee", "acm", "springer", "elsevier", "nature", "science",
    
    # Technologies / Concepts / Common Nouns
    "network", "networks", "learning", "model", "models", "system", "systems", "algorithm", "algorithms", 
    "process", "processes", "framework", "frameworks", "analysis", "synthesis", "evaluation", "experiment", 
    "experiments", "experimental", "performance", "accuracy", "dataset", "datasets", "database", "databases", 
    "data", "software", "code", "repository", "repositories", "github", "gitlab", "bitbucket", "web", "internet", 
    "website", "online", "digital", "analog", "computer", "machine", "intelligence", "artificial", "human", 
    "agent", "agents", "user", "users", "client", "server", "node", "nodes", "edge", "edges", "graph", "graphs", 
    "vector", "vectors", "embedding", "embeddings", "tensor", "tensors", "matrix", "matrices", "gradient", 
    "gradients", "loss", "losses", "optimizer", "optimizers", "attention", "transformer", "transformers", 
    "encoder", "encoders", "decoder", "decoders", "multiplier", "multipliers", "accumulator", "accumulators", 
    "block", "blocks", "spectrometry", "spectroscopy", "microscopy", "imaging", "chromatography", "tracing", 
    "tracking", "sampling", "tuning", "modeling", "matching", "programming", "mapping", "clustering", "routing",
    "compiling", "compiler", "compilers", "representation", "representations", "defaults", "synthesis", 
    "implementation", "design", "verification", "verifier", "verifiers", "evaluator", "evaluators", "predictor", 
    "predictors", "classifier", "classifiers", "regressor", "regressors", "outcome", "labels", "label", 
    "expert", "experts", "chain", "chains", "path", "paths", "cycle", "cycles", "loop", "loops", "tree", "trees", 
    "forest", "forests", "hivemind", "mixture", "mixtures", "log", "logs", "guideline", "guidelines", 
    "construction", "acquisition", "access", "open", "closed", "public", "private", "material", "materials", 
    "idea", "ideas", "summary", "summaries", "outline", "outlines", "generation", "defense", "survey", "surveys", 
    "policy", "policies", "optimization", "optimizations", "experience", "experiences", "vivado", "xilinx",
    "synopsys", "compiler", "compilers", "device", "devices", "hardware", "software", "silicon", "chip", "chips",
    "intel", "amd", "nvidia", "arm", "apple", "google", "meta", "microsoft", "amazon", "facebook", "twitter", 
    "ibm", "openai", "deepseek", "anthropic", "cohere", "huggingface", "run", "runs", "test", "tests", "testing",
    "development", "production"
}


def is_likely_name(text: str) -> bool:
    """Heuristic check: is this string a plausible human name?"""
    # Clean up whitespace and outer quotes
    text_clean = text.strip().strip(".,;:!?()[]{}'\"")
    words = text_clean.split()
    if len(words) < 2 or len(words) > 5:
        return False
    
    # Check for minor grammatical words that don't belong in names
    if any(w.lower() in {"and", "or", "the", "of", "in", "for", "with", "at", "to", "by", "from", "on"} for w in words):
        return False
        
    # Names should not contain digits
    if any(char.isdigit() for char in text_clean):
        return False
        
    # Check each word against forbidden name words
    for w in words:
        # Clean word from punctuation before checking
        w_clean = w.strip(".,;:!?()[]{}'\"").lower()
        if w_clean in _FORBIDDEN_NAME_WORDS:
            return False
            
    if len(text_clean) > 50:
        return False
        
    return True


def _is_model_cached(model_id: str) -> bool:
    """Check if a HuggingFace model is already in the local cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        result = try_to_load_from_cache(model_id, "config.json")
        return result is not None
    except Exception:
        return False


class NEREngine:
    """Extracts PERSON entities from text using bert-base-NER."""
    
    def __init__(self):
        from src.config import config
        self.model_id = config.ner_model_name
        self._pipeline = None
        self._load_model()

    def _load_model(self):
        from src import console as con
        from src.config import config

        # Set HF token if configured
        hf_token = config.hf_token
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

        is_cached = _is_model_cached(self.model_id)
        if not is_cached:
            con.info(f"Downloading NER model [bold]{self.model_id}[/bold] (one-time, ~400 MB)…")
        else:
            con.model_msg(f"Loading NER model [bold]{self.model_id}[/bold] from cache…")

        try:
            import transformers
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Use 'first' strategy — more reliable for multi-token names than 'simple'
                self._pipeline = transformers.pipeline(
                    "ner",
                    model=self.model_id,
                    aggregation_strategy="first",
                    token=hf_token or None,
                )
            con.success(f"NER model ready: [bold]{self.model_id}[/bold]")
        except Exception as e:
            con.warning(f"NER model could not load ({e}). Falling back to regex extraction.")
            self._pipeline = None

    def extract_persons(self, text: str, max_text: int = 4000) -> List[str]:
        """
        Extract person names from `text`.
        Uses HuggingFace NER pipeline when available, else falls back to regex.
        """
        snippet = text[:max_text]
        
        if self._pipeline is not None:
            return self._ner_extract(snippet)
        return self._regex_extract(snippet)

    def _ner_extract(self, text: str) -> List[str]:
        try:
            entities = self._pipeline(text)
            persons = []
            for ent in entities:
                if ent.get("entity_group") == "PER":
                    word = ent["word"].strip()
                    # Clean WordPiece subword artifacts: "##ish Vaswani" → "Vaswani" (partial)
                    # We rebuild the name from the original text span when possible
                    start, end = ent.get("start"), ent.get("end")
                    if start is not None and end is not None:
                        # Extract directly from source text to avoid subword issues
                        name = text[start:end].strip()
                    else:
                        name = re.sub(r'^#+', '', word).strip()
                    if is_likely_name(name) and name not in persons:
                        persons.append(name)
            return persons
        except Exception:
            return self._regex_extract(text)

    def _regex_extract(self, text: str) -> List[str]:
        """Pure-regex fallback — looks for capitalized name patterns."""
        found = _NAME_PATTERN.findall(text)
        seen = set()
        result = []
        for name in found:
            if is_likely_name(name) and name not in seen:
                seen.add(name)
                result.append(name)
        return result


# Module-level singleton, lazy-initialized
_ner_instance: Optional[NEREngine] = None


def get_ner_engine() -> NEREngine:
    global _ner_instance
    if _ner_instance is None:
        _ner_instance = NEREngine()
    return _ner_instance


def extract_persons_from_text(text: str) -> List[str]:
    """Convenience wrapper — returns deduplicated list of person names."""
    return get_ner_engine().extract_persons(text)
