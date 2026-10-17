"""
Normalization Pipeline Module.

Handles cleaning, translating (stub), alias resolution, lemmatization (via spaCy),
and slug generation for concepts, tags, and authors to ensure graph determinism.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional
try:
    import spacy
    import spacy.cli
except ImportError:
    spacy = None
from src.models import slugify
from src.llm_schemas import LLMExtractionResponse, LLMConcept, LLMCitationIntent, LLMConceptRelation, LLMDataset

logger = logging.getLogger(__name__)

# Default global acronyms and variation aliases mapped to canonical names.
DEFAULT_ALIASES: Dict[str, str] = {
    "ml": "Machine Learning",
    "machine learning": "Machine Learning",
    "cnn": "Convolutional Neural Network",
    "cnns": "Convolutional Neural Network",
    "rnn": "Recurrent Neural Network",
    "rnns": "Recurrent Neural Network",
    "nlp": "Natural Language Processing",
    "llm": "Large Language Model",
    "llms": "Large Language Model",
    "gan": "Generative Adversarial Network",
    "gans": "Generative Adversarial Network",
    "rl": "Reinforcement Learning",
    "sgd": "Stochastic Gradient Descent",
    "adam": "Adam Optimizer",
    "vit": "Vision Transformer",
    "vits": "Vision Transformer",
    "gemma": "Gemma Model",
    "bert": "BERT Model",
    "gpt": "GPT Model",
    "t5": "T5 Model",
    "lstm": "LSTM Network",
    "lstms": "LSTM Network",
}

_nlp: Optional[spacy.language.Language] = None
_spacy_attempted: bool = False


def get_spacy_nlp() -> Optional[spacy.language.Language]:
    """Lazy loaded spaCy model tokenizer & parser."""
    global _nlp, _spacy_attempted
    if _nlp is not None:
        return _nlp
    if spacy is None:
        return None
    if _spacy_attempted:
        return None

    _spacy_attempted = True
    import os
    import sys
    from src.config import config
    model_name = config.spacy_model_name
    try:
        # Disable parser and NER components to make loading and lemmatization much faster
        _nlp = spacy.load(model_name, disable=["parser", "ner"])
    except OSError as e:
        is_path = os.path.sep in model_name or os.path.exists(model_name)
        if is_path:
            from src import console as con
            con.warning(
                f"spaCy model path '{model_name}' could not be loaded: {e}. "
                "Lemmatization will fall back to original text."
            )
            logger.warning(
                f"spaCy model path '{model_name}' could not be loaded: {e}. "
                "Lemmatization will fall back to original text."
            )
        else:
            try:
                from src import console as con
                con.model_msg(f"spaCy model '{model_name}' is not installed. Attempting to download...")
                
                # Check if running in a virtual environment
                # sys.prefix != sys.base_prefix is the true indicator for the current Python interpreter
                is_venv = sys.prefix != sys.base_prefix
                env_backup = {}
                
                # Backup current values of all relevant environment variables
                for var in ["VIRTUAL_ENV", "UV_SYSTEM_PYTHON", "PIP_BREAK_SYSTEM_PACKAGES"]:
                    if var in os.environ:
                        env_backup[var] = os.environ[var]
                
                if is_venv:
                    # Force VIRTUAL_ENV to match the current python interpreter's virtualenv prefix
                    os.environ["VIRTUAL_ENV"] = sys.prefix
                    # Ensure system python flags are NOT set (we want to install to the venv)
                    os.environ.pop("UV_SYSTEM_PYTHON", None)
                    os.environ.pop("PIP_BREAK_SYSTEM_PACKAGES", None)
                else:
                    # Not in virtualenv, we want to allow installing to the system python if using uv/pip
                    os.environ["UV_SYSTEM_PYTHON"] = "true"
                    os.environ["PIP_BREAK_SYSTEM_PACKAGES"] = "true"
                    # Remove any leftover VIRTUAL_ENV env var that could confuse uv
                    os.environ.pop("VIRTUAL_ENV", None)
                
                try:
                    # Always use spacy.cli.download – it uses pip or uv internally,
                    # which respects the environment variables configured above.
                    spacy.cli.download(model_name)
                finally:
                    # Restore original environment variables
                    for var in ["VIRTUAL_ENV", "UV_SYSTEM_PYTHON", "PIP_BREAK_SYSTEM_PACKAGES"]:
                        if var in env_backup:
                            os.environ[var] = env_backup[var]
                        else:
                            os.environ.pop(var, None)
                                
                _nlp = spacy.load(model_name, disable=["parser", "ner"])
                con.success(f"spaCy model '{model_name}' downloaded and loaded successfully.")
            except Exception as ex:
                from src import console as con
                con.warning(
                    f"spaCy model '{model_name}' is not installed and auto-download failed ({ex}). "
                    "Lemmatization will fall back to original text."
                )
                logger.warning(
                    f"spaCy model '{model_name}' is not installed and auto-download failed ({ex}). "
                    "Lemmatization will fall back to original text."
                )
    return _nlp


class NormalizationPipeline:
    """
    Decoupled pipeline for sanitizing, translating, lemmatizing, and resolving aliases
    for extracted graph nodes (Authors, Concepts, and Tags).
    """

    def __init__(self, aliases: Optional[Dict[str, str]] = None) -> None:
        self.aliases = aliases if aliases is not None else DEFAULT_ALIASES

    def translate_to_english(self, text: str) -> str:
        """
        Translates non-English concept names and tags to English.
        Currently implemented as a stub, returning text as-is.
        """
        # In future implementations, a translation library or API can be integrated here.
        return text

    def _lemmatize(self, text: str) -> str:
        """Converts words in the text to their base singular forms using spaCy, preserving scientific terms and proper nouns."""
        nlp = get_spacy_nlp()
        if not nlp:
            return text

        words = text.strip().split()
        if not words:
            return ""

        # Blacklist of common surnames ending in 's' that should not be singularized
        surnames_ending_in_s = {
            "williams", "stevens", "stephens", "jones", "harris", "davis", "evans", 
            "roberts", "rogers", "hughes", "morris", "james", "adams", "phillips", 
            "thomas", "baboshina", "skovina", "nagornov"
        }

        lemmatized_words = []
        for i, word in enumerate(words):
            is_last = (i == len(words) - 1)
            
            # Clean punctuation from word for lookup
            clean_word = word.strip(".,;:!?()[]{}'\"")
            lower_word = clean_word.lower()
            
            # 1. Do not lemmatize modifiers (non-last words) to preserve "supervised", "distributed", etc.
            if not is_last:
                lemmatized_words.append(lower_word)
                continue
                
            # 2. Preserve common scientific -ing suffixes (but not -ings)
            if lower_word.endswith("ing") and not lower_word.endswith("ings"):
                lemmatized_words.append(lower_word)
                continue
                
            # 3. Singularize plural "ings" to "ing"
            if lower_word.endswith("ings"):
                lemmatized_words.append(lower_word[:-1])
                continue

            # 4. Preserve known surnames ending in 's'
            if lower_word in surnames_ending_in_s:
                lemmatized_words.append(lower_word)
                continue
                
            # 5. Fallback: lemmatize using spaCy
            doc = nlp(lower_word)
            if doc and len(doc) > 0:
                lemmatized_words.extend([t.lemma_ for t in doc])
            else:
                lemmatized_words.append(lower_word)

        joined = " ".join(lemmatized_words)
        import re
        return re.sub(r'\s*-\s*', '-', joined)


    def normalize_concept_name(self, name: str) -> str:
        """
        Performs full normalization on a concept name:
        1. English translation (stub)
        2. Alias/acronym lookup (case-insensitive)
        3. spaCy lemmatization (base singular uninflected form)
        4. Canonical Title Casing (preserving interior casing like acronyms if possible)
        """
        # Translate
        translated = self.translate_to_english(name.strip())
        
        # Check alias lookup (case-insensitive key comparison)
        lookup_key = translated.lower()
        if lookup_key in self.aliases:
            resolved = self.aliases[lookup_key]
        else:
            resolved = translated

        # Lemmatize
        lemmatized = self._lemmatize(resolved)

        return self._title_case(lemmatized)

    def _title_case(self, text: str) -> str:
        """Capitalizes each word, supporting sub-words separated by hyphens."""
        words = text.split()
        capitalized_words = []
        for w in words:
            if not w:
                continue
            parts = w.split('-')
            cap_parts = [p[0].upper() + p[1:] if p else "" for p in parts]
            capitalized_words.append("-".join(cap_parts))
        return " ".join(capitalized_words)

    def normalize_tag(self, tag: str) -> str:
        """
        Normalizes topic tags:
        1. English translation (stub)
        2. Alias resolution
        3. Title Case formatting
        """
        translated = self.translate_to_english(tag.strip())
        lookup_key = translated.lower()
        
        if lookup_key in self.aliases:
            resolved = self.aliases[lookup_key]
        else:
            resolved = translated
            
        return self._title_case(resolved)

    def normalize_author_name(self, name: str) -> str:
        """
        Cleans author name by removing leading/trailing spaces and formatting
        to Capital Case. Does NOT apply lemmatization or translation.
        """
        return self._title_case(name.strip())

    def normalize_description(self, description: str) -> str:
        """
        Sanitizes the description by stripping any thinking/reasoning tags and text before/within them.
        """
        if not description:
            return ""
        import re
        desc = description.strip()
        # Find first closing think/thought/reasoning tag and discard everything before it
        match = re.search(r'</(think|thought|reasoning)>', desc, re.IGNORECASE)
        if match:
            desc = desc[match.end():].strip()
        # Strip any remaining tags
        desc = re.sub(r'</?(think|thought|reasoning)>', '', desc, flags=re.IGNORECASE).strip()
        return desc

    def normalize_extraction_response(self, response: LLMExtractionResponse) -> LLMExtractionResponse:
        """
        Intercepts an LLMExtractionResponse Pydantic model and returns a new
        fully-normalized, deduplicated model prior to graph persistence.
        """
        # 1. Normalize authors (deduplicating case variations)
        normalized_authors: List[str] = []
        seen_authors = set()
        authors_raw = getattr(response, "authors", []) or []
        for author in authors_raw:
            norm_author = self.normalize_author_name(author)
            if norm_author and norm_author.lower() not in seen_authors:
                seen_authors.add(norm_author.lower())
                normalized_authors.append(norm_author)

        # 2. Normalize and deduplicate concepts
        normalized_concepts: List[LLMConcept] = []
        seen_concepts = set()
        concepts_raw = getattr(response, "concepts", []) or []
        for concept in concepts_raw:
            if not concept.name:
                continue
            norm_name = self.normalize_concept_name(concept.name)
            concept_slug = slugify(norm_name)
            
            if concept_slug and concept_slug not in seen_concepts:
                seen_concepts.add(concept_slug)
                aliases = getattr(concept, "aliases", []) or []
                normalized_concepts.append(
                    LLMConcept(
                        name=norm_name,
                        description=self.normalize_description(concept.description),
                        aliases=[self.normalize_concept_name(al) for al in aliases if al.strip()]
                    )
                )

        # 3. Normalize and deduplicate tags
        normalized_tags: List[str] = []
        seen_tags = set()
        tags_raw = getattr(response, "tags", []) or []
        for tag in tags_raw:
            norm_tag = self.normalize_tag(tag)
            tag_slug = slugify(norm_tag)
            
            if tag_slug and tag_slug not in seen_tags:
                seen_tags.add(tag_slug)
                normalized_tags.append(norm_tag)

        # 4. Normalize institutions
        normalized_institutions = []
        seen_insts = set()
        insts_raw = getattr(response, "institutions", []) or []
        for inst in insts_raw:
            norm_inst = self._title_case(inst.strip())
            if norm_inst and norm_inst.lower() not in seen_insts:
                seen_insts.add(norm_inst.lower())
                normalized_institutions.append(norm_inst)

        # 5. Normalize author_institutions
        normalized_author_insts = []
        ai_raw = getattr(response, "author_institutions", []) or []
        for ai in ai_raw:
            norm_auth = self.normalize_author_name(ai.get("author", ""))
            norm_inst = self._title_case(ai.get("institution", "").strip())
            if norm_auth and norm_inst:
                normalized_author_insts.append({"author": norm_auth, "institution": norm_inst})

        # 6. sponsored_by
        normalized_sponsored = []
        seen_sp = set()
        sp_raw = getattr(response, "sponsored_by", []) or []
        for sp in sp_raw:
            norm_sp = self._title_case(sp.strip())
            if norm_sp and norm_sp.lower() not in seen_sp:
                seen_sp.add(norm_sp.lower())
                normalized_sponsored.append(norm_sp)

        # 7. datasets
        normalized_datasets = []
        seen_ds = set()
        ds_raw = getattr(response, "datasets", []) or []
        for ds in ds_raw:
            norm_ds_name = self._title_case(ds.name.strip())
            if norm_ds_name and norm_ds_name.lower() not in seen_ds:
                seen_ds.add(norm_ds_name.lower())
                normalized_datasets.append(LLMDataset(name=norm_ds_name, relation=ds.relation))

        # 8. code_repositories
        normalized_code = []
        seen_code = set()
        code_raw = getattr(response, "code_repositories", []) or []
        for cr in code_raw:
            cleaned = cr.strip()
            if cleaned and cleaned.lower() not in seen_code:
                seen_code.add(cleaned.lower())
                normalized_code.append(cleaned)

        # 9. journal_or_conference
        jc_val = getattr(response, "journal_or_conference", None)
        normalized_jc = jc_val.strip() if jc_val else None

        # 10. citation_intents
        normalized_citations = []
        cit_raw = getattr(response, "citation_intents", []) or []
        for ci in cit_raw:
            if ci.target_title:
                normalized_citations.append(LLMCitationIntent(target_title=ci.target_title.strip(), intent=ci.intent))

        # 11. concept_relations
        normalized_concept_rels = []
        cr_raw = getattr(response, "concept_relations", []) or []
        for cr in cr_raw:
            norm_src = self.normalize_concept_name(cr.source)
            norm_tgt = self.normalize_concept_name(cr.target)
            if norm_src and norm_tgt:
                normalized_concept_rels.append(LLMConceptRelation(source=norm_src, target=norm_tgt, relation_type=cr.relation_type))

        return LLMExtractionResponse(
            authors=normalized_authors,
            concepts=normalized_concepts,
            tags=normalized_tags,
            institutions=normalized_institutions,
            author_institutions=normalized_author_insts,
            sponsored_by=normalized_sponsored,
            datasets=normalized_datasets,
            code_repositories=normalized_code,
            journal_or_conference=normalized_jc,
            citation_intents=normalized_citations,
            concept_relations=normalized_concept_rels
        )
