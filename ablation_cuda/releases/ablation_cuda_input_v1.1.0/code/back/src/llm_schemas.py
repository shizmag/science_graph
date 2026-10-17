import re
from typing import List, Dict, Tuple, Optional, Any
from pydantic import BaseModel, Field, RootModel, field_validator

class LLMConcept(BaseModel):
    name: str
    description: str
    aliases: List[str] = Field(default_factory=list)

class LLMCitationIntent(BaseModel):
    target_title: str
    intent: str = Field(default="BACKGROUND")

class LLMConceptRelation(BaseModel):
    source: str
    target: str
    relation_type: str

class LLMDataset(BaseModel):
    name: str
    relation: str = Field(default="USED_DATASET")

class LLMExtractionResponse(BaseModel):
    authors: List[str] = Field(default_factory=list)
    concepts: List[LLMConcept] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    institutions: List[str] = Field(default_factory=list)
    author_institutions: List[Dict[str, str]] = Field(default_factory=list)
    sponsored_by: List[str] = Field(default_factory=list)
    datasets: List[LLMDataset] = Field(default_factory=list)
    code_repositories: List[str] = Field(default_factory=list)
    journal_or_conference: Optional[str] = None
    citation_intents: List[LLMCitationIntent] = Field(default_factory=list)
    concept_relations: List[LLMConceptRelation] = Field(default_factory=list)

    @field_validator("authors", "tags", "institutions", "sponsored_by", mode="before")
    @classmethod
    def clean_and_deduplicate_strings(cls, v: Any) -> List[str]:
        if not isinstance(v, list):
            return []
        seen = set()
        cleaned = []
        for x in v:
            if isinstance(x, str):
                s = x.strip()
                if s and s.lower() not in seen:
                    seen.add(s.lower())
                    cleaned.append(s)
        return cleaned

    @field_validator("code_repositories", mode="before")
    @classmethod
    def validate_and_clean_urls(cls, v: Any) -> List[str]:
        if not isinstance(v, list):
            return []
        seen = set()
        cleaned = []
        for x in v:
            if isinstance(x, str):
                s = x.strip()
                if re.match(r"^https?://[^\s/$.?#].[^\s]*$", s, re.IGNORECASE):
                    if s.lower() not in seen:
                        seen.add(s.lower())
                        cleaned.append(s)
        return cleaned

    @field_validator("concepts", mode="before")
    @classmethod
    def validate_unique_concepts(cls, v: Any) -> List[Any]:
        if not isinstance(v, list):
            return []
        seen = set()
        cleaned = []
        for item in v:
            if isinstance(item, dict):
                name = item.get("name", "")
            elif hasattr(item, "name"):
                name = getattr(item, "name", "")
            else:
                name = ""
            if isinstance(name, str):
                n_cleaned = name.strip().lower()
                if n_cleaned not in seen:
                    seen.add(n_cleaned)
                    cleaned.append(item)
        return cleaned

    @field_validator("citation_intents", mode="before")
    @classmethod
    def validate_unique_citation_intents(cls, v: Any) -> List[Any]:
        if not isinstance(v, list):
            return []
        seen = set()
        cleaned = []
        for item in v:
            if isinstance(item, dict):
                title = item.get("target_title", "")
                intent = item.get("intent", "BACKGROUND")
            elif hasattr(item, "target_title"):
                title = getattr(item, "target_title", "")
                intent = getattr(item, "intent", "BACKGROUND")
            else:
                title, intent = "", ""
            if isinstance(title, str) and isinstance(intent, str):
                key = (title.strip().lower(), intent.strip().upper())
                if key not in seen:
                    seen.add(key)
                    cleaned.append(item)
        return cleaned

    @field_validator("concept_relations", mode="before")
    @classmethod
    def validate_unique_concept_relations(cls, v: Any) -> List[Any]:
        if not isinstance(v, list):
            return []
        seen = set()
        cleaned = []
        for item in v:
            if isinstance(item, dict):
                src = item.get("source", "")
                tgt = item.get("target", "")
                rel = item.get("relation_type", "")
            elif hasattr(item, "source"):
                src = getattr(item, "source", "")
                tgt = getattr(item, "target", "")
                rel = getattr(item, "relation_type", "")
            else:
                src, tgt, rel = "", "", ""
            if isinstance(src, str) and isinstance(tgt, str) and isinstance(rel, str):
                key = (src.strip().lower(), tgt.strip().lower(), rel.strip().upper())
                if key not in seen:
                    seen.add(key)
                    cleaned.append(item)
        return cleaned

class LLMClusteringResponse(RootModel[Dict[str, List[str]]]):
    pass

def validate_extraction_response(raw_data: dict) -> Tuple[LLMExtractionResponse, List[str]]:
    """
    Validates LLM extraction response at structured level,
    filtering out noisy data and providing a detailed list of warnings.
    """
    warnings = []
    
    authors_raw = raw_data.get("authors", [])
    concepts_raw = raw_data.get("concepts", [])
    tags_raw = raw_data.get("tags", [])
    
    if not isinstance(authors_raw, list):
        warnings.append("Expected 'authors' to be a list, resetting to empty.")
        authors_raw = []
    if not isinstance(concepts_raw, list):
        warnings.append("Expected 'concepts' to be a list, resetting to empty.")
        concepts_raw = []
    if not isinstance(tags_raw, list):
        warnings.append("Expected 'tags' to be a list, resetting to empty.")
        tags_raw = []
    # 1. Validate authors
    valid_authors = []
    for author in authors_raw:
        if not isinstance(author, str):
            warnings.append(f"Ignored non-string author: {author}")
            continue
        cleaned = author.strip()
        if not cleaned:
            continue
            
        cleaned = re.sub(r"\d+", "", cleaned)
        cleaned = re.sub(r"[.,;]+$", "", cleaned)
        cleaned = re.sub(r'^["\']|["\']$', "", cleaned)
        cleaned = cleaned.strip()
        
        if len(cleaned) < 2 or not re.search(r'[A-Za-z\u0400-\u04FF]', cleaned):
            warnings.append(f"Filtered out invalid author name format: '{author}'")
            continue
            
        from src.ner_engine import is_likely_name
        if not is_likely_name(cleaned):
            warnings.append(f"Filtered out institutional/noisy/unlikely author entry: '{author}'")
            continue
            
        valid_authors.append(cleaned)


    # 2. Validate concepts
    valid_concepts = []
    for concept in concepts_raw:
        if not isinstance(concept, dict):
            warnings.append(f"Ignored non-dict concept item: {concept}")
            continue
        c_name = concept.get("name", "")
        c_desc = concept.get("description", "")
        c_aliases_raw = concept.get("aliases", [])
        
        if not isinstance(c_name, str) or not isinstance(c_desc, str):
            warnings.append(f"Ignored concept with non-string name/description: {concept}")
            continue
            
        c_name = c_name.strip()
        c_desc = c_desc.strip()
        
        if not c_name:
            warnings.append("Filtered out concept with empty name.")
            continue
            
        # Clean concept name from citation brackets
        c_name = re.sub(r"\[\d+\]", "", c_name)
        c_name = re.sub(r"\(\s*[\w\s,]+,\s*\d{4}\s*\)", "", c_name)
        c_name = re.sub(r'^["\']|["\']$', "", c_name).strip()
        
        word_count = len(c_name.split())
        if word_count > 5:
            warnings.append(f"Filtered out concept '{c_name}' because the name is too long ({word_count} words). Expected 1-3 words.")
            continue
            
        desc_word_count = len(c_desc.split())
        if c_desc and (desc_word_count < 3 or desc_word_count > 60):
            warnings.append(f"Warning: Concept '{c_name}' description has unusual length ({desc_word_count} words).")
            
        valid_aliases = []
        if isinstance(c_aliases_raw, list):
            for al in c_aliases_raw:
                if isinstance(al, str) and al.strip():
                    valid_aliases.append(al.strip())
            
        valid_concepts.append(LLMConcept(name=c_name, description=c_desc, aliases=valid_aliases))

    # 3. Validate tags
    valid_tags = []
    for tag in tags_raw:
        if not isinstance(tag, str):
            warnings.append(f"Ignored non-string tag: {tag}")
            continue
        cleaned = tag.strip().lower()
        if not cleaned:
            continue
            
        word_count = len(cleaned.split())
        if word_count > 4:
            warnings.append(f"Filtered out tag '{tag}' because it is too long ({word_count} words). Expected 1-3 words.")
            continue
            
        if re.search(r"\[\d+\]", cleaned) or any(bad in cleaned for bad in ["@", "*", "http"]):
            warnings.append(f"Filtered out tag with invalid characters: '{tag}'")
            continue
            
        valid_tags.append(cleaned)

    # 4. Validate institutions
    institutions_raw = raw_data.get("institutions", [])
    valid_institutions = []
    if isinstance(institutions_raw, list):
        for inst in institutions_raw:
            if isinstance(inst, str):
                cleaned = inst.strip()
                if cleaned:
                    valid_institutions.append(cleaned)
            else:
                warnings.append(f"Ignored non-string institution: {inst}")
    else:
        warnings.append("Expected 'institutions' to be a list.")

    # 5. Validate author_institutions
    author_inst_raw = raw_data.get("author_institutions", [])
    valid_author_inst = []
    if isinstance(author_inst_raw, list):
        for ai in author_inst_raw:
            if isinstance(ai, dict):
                author = ai.get("author", "")
                institution = ai.get("institution", "")
                if isinstance(author, str) and isinstance(institution, str):
                    author = author.strip()
                    institution = institution.strip()
                    if author and institution:
                        valid_author_inst.append({"author": author, "institution": institution})
            else:
                warnings.append(f"Ignored non-dict author_institution entry: {ai}")
    else:
        warnings.append("Expected 'author_institutions' to be a list.")

    # 6. Validate sponsored_by
    sponsored_raw = raw_data.get("sponsored_by", [])
    valid_sponsored = []
    if isinstance(sponsored_raw, list):
        for sp in sponsored_raw:
            if isinstance(sp, str):
                cleaned = sp.strip()
                if cleaned:
                    valid_sponsored.append(cleaned)
            else:
                warnings.append(f"Ignored non-string sponsored_by entry: {sp}")
    else:
        warnings.append("Expected 'sponsored_by' to be a list.")

    # 7. Validate datasets
    datasets_raw = raw_data.get("datasets", [])
    valid_datasets = []
    if isinstance(datasets_raw, list):
        for ds in datasets_raw:
            if isinstance(ds, dict):
                name = ds.get("name", "")
                relation = ds.get("relation", "USED_DATASET")
                if isinstance(name, str) and isinstance(relation, str):
                    name = name.strip()
                    relation = relation.strip().upper()
                    if relation not in ("USED_DATASET", "INTRODUCED_DATASET"):
                        relation = "USED_DATASET"
                    if name:
                        valid_datasets.append(LLMDataset(name=name, relation=relation))
            elif isinstance(ds, str):
                cleaned = ds.strip()
                if cleaned:
                    valid_datasets.append(LLMDataset(name=cleaned, relation="USED_DATASET"))
    else:
        warnings.append("Expected 'datasets' to be a list.")

    # 8. Validate code_repositories
    code_raw = raw_data.get("code_repositories", [])
    valid_code = []
    if isinstance(code_raw, list):
        for cr in code_raw:
            if isinstance(cr, str):
                cleaned = cr.strip()
                if cleaned:
                    valid_code.append(cleaned)
    else:
        warnings.append("Expected 'code_repositories' to be a list.")

    # 9. Validate journal_or_conference
    jc_raw = raw_data.get("journal_or_conference")
    valid_jc = None
    if isinstance(jc_raw, str):
        cleaned = jc_raw.strip()
        if cleaned:
            valid_jc = cleaned

    # 10. Validate citation_intents
    citation_intents_raw = raw_data.get("citation_intents", [])
    valid_citations = []
    if isinstance(citation_intents_raw, list):
        for ci in citation_intents_raw:
            if isinstance(ci, dict):
                title = ci.get("target_title", "")
                intent = ci.get("intent", "BACKGROUND")
                if isinstance(title, str) and isinstance(intent, str):
                    title = title.strip()
                    intent = intent.strip().upper()
                    if not intent:
                        intent = "BACKGROUND"
                    if title:
                        valid_citations.append(LLMCitationIntent(target_title=title, intent=intent))
            else:
                warnings.append(f"Ignored non-dict citation_intents entry: {ci}")
    else:
        warnings.append("Expected 'citation_intents' to be a list.")

    # 11. Validate concept_relations
    concept_rel_raw = raw_data.get("concept_relations", [])
    valid_concept_rels = []
    if isinstance(concept_rel_raw, list):
        for cr in concept_rel_raw:
            if isinstance(cr, dict):
                src = cr.get("source", "")
                tgt = cr.get("target", "")
                rel = cr.get("relation_type", "")
                if isinstance(src, str) and isinstance(tgt, str) and isinstance(rel, str):
                    src = src.strip()
                    tgt = tgt.strip()
                    rel = rel.strip().upper()
                    if src and tgt and rel in ("SUBCLASS_OF", "IS_A", "PREREQUISITE_FOR"):
                        valid_concept_rels.append(LLMConceptRelation(source=src, target=tgt, relation_type=rel))
            else:
                warnings.append(f"Ignored non-dict concept_relations entry: {cr}")
    else:
        warnings.append("Expected 'concept_relations' to be a list.")

    model = LLMExtractionResponse(
        authors=valid_authors,
        concepts=valid_concepts,
        tags=valid_tags,
        institutions=valid_institutions,
        author_institutions=valid_author_inst,
        sponsored_by=valid_sponsored,
        datasets=valid_datasets,
        code_repositories=valid_code,
        journal_or_conference=valid_jc,
        citation_intents=valid_citations,
        concept_relations=valid_concept_rels
    )
    return model, warnings

def validate_clustering_response(raw_data: dict) -> Tuple[LLMClusteringResponse, List[str]]:
    """
    Validates LLM clustering response at structured level,
    filtering out invalid structures and chunk IDs, returning list of warnings.
    """
    warnings = []
    if not isinstance(raw_data, dict):
        warnings.append("Expected clustering response to be a JSON dictionary.")
        return LLMClusteringResponse.model_validate({}), warnings
        
    cleaned_dict = {}
    for section_title, chunk_ids in raw_data.items():
        if not isinstance(section_title, str):
            warnings.append(f"Ignored non-string section title: {section_title}")
            continue
        title = section_title.strip()
        if not title:
            warnings.append("Filtered out section with empty title.")
            continue
            
        if len(title.split()) > 10:
            warnings.append(f"Warning: Section title '{title}' is unusually long.")
            
        if not isinstance(chunk_ids, list):
            warnings.append(f"Ignored section '{title}' because chunk IDs is not a list.")
            continue
            
        valid_chunk_ids = []
        for cid in chunk_ids:
            if not isinstance(cid, str):
                warnings.append(f"Ignored non-string chunk ID '{cid}' in section '{title}'.")
                continue
            cid = cid.strip()
            if not cid:
                continue
            valid_chunk_ids.append(cid)
            
        if not valid_chunk_ids:
            warnings.append(f"Filtered out section '{title}' because it has no valid chunk IDs.")
            continue
            
        cleaned_dict[title] = valid_chunk_ids
        
    model = LLMClusteringResponse.model_validate(cleaned_dict)
    return model, warnings

class LLMVerificationResponse(BaseModel):
    relevant: bool = Field(description="True if the chunk text contains relevant educational, informational, or scientific concepts/knowledge for the database. False if it is advertisement, self-promotion, sponsor content/reads, intro/outro chat, or irrelevant filler.")
    reason: str = Field(description="Brief explanation of why the chunk is relevant or irrelevant.")

class LLMVideoSummaryResponse(BaseModel):
    overview: str = Field(description="A concise summary/overview of the video (2-3 paragraphs).")
    themes: List[str] = Field(description="List of key themes or topics discussed in the video, with brief explanations.")
    outline: List[str] = Field(description="Detailed lecture outline or chronological/structured breakdown of the video's content.")

    @field_validator("themes", "outline", mode="before")
    @classmethod
    def clean_list_strings(cls, v: Any) -> List[str]:
        if not isinstance(v, list):
            return []
        seen = set()
        cleaned = []
        for x in v:
            if isinstance(x, str):
                s = x.strip()
                if s and s.lower() not in seen:
                    seen.add(s.lower())
                    cleaned.append(s)
        return cleaned


class EvidenceItem(BaseModel):
    id: str
    score: float
    is_essential: bool


class EvidenceListResponse(BaseModel):
    evidence_list: List[EvidenceItem]


class LLMQueryExpansionResponse(RootModel[List[str]]):
    pass


