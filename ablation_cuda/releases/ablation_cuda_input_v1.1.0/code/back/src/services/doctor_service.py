import re
import json
import logging
from typing import List, Dict, Any, Tuple
from src.models import Author, Concept, Paper, slugify
from src.repository.base import GraphRepository, VectorRepository
from src.services.normalization_pipeline import NormalizationPipeline

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Cleans raw text values to remove technical tokens and other artifacts of LLM outputs.
    Specifically, it:
    1. Removes reasoning/thinking blocks (<think>...</think> or <thought>...</thought>).
    2. Removes unclosed thinking blocks at the end of text.
    3. Removes markdown code block wraps (```json ... ```).
    4. Strips outer wrapping quotes.
    5. Normalizes consecutive whitespaces and double line breaks.
    """
    if not text:
        return ""

    # 1. Remove closed think/thought/reasoning blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # 2. Remove unclosed think/thought/reasoning blocks at the end of text
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<thought>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<reasoning>.*", "", text, flags=re.DOTALL | re.IGNORECASE)

    # 3. Clean up leftover tags
    text = re.sub(r"</?(think|thought|reasoning)>", "", text, flags=re.IGNORECASE)

    # 4. Remove markdown code blocks (e.g. ```json ... ```)
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|markdown|text|xml|html)?\n?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    # 5. Remove wrapping quotes if they wrap the entire string
    if len(text) >= 2:
        for q in ['"', "'", '`']:
            if text.startswith(q) and text.endswith(q):
                text = text[1:-1].strip()
                break

    # 6. Replace multiple consecutive spaces with a single space (preserving newlines)
    lines = []
    for line in text.splitlines():
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(cleaned_line)

    # 7. Join lines and replace multiple consecutive blank lines with a single blank line
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


class DoctorService:
    """
    Service to scan and sanitize all entities in the database.
    Checks and fixes unapplied formatters and LLM artifacts.
    """

    def __init__(self, graph_repo: GraphRepository, vector_repo: VectorRepository, llm_engine: Any = None, emb_engine: Any = None):
        self.graph_repo = graph_repo
        self.vector_repo = vector_repo
        self.llm_engine = llm_engine
        self.emb_engine = emb_engine
        self.normalizer = NormalizationPipeline()

    def run_diagnostics(self, fix: bool = False) -> Dict[str, Any]:
        """
        Runs diagnostics and returns a dictionary detailing changes and stats.
        If fix=True, actually writes changes to the repository.
        """
        report = {
            "stats": {
                "papers_checked": 0,
                "papers_fixed": 0,
                "concepts_checked": 0,
                "concepts_fixed": 0,
                "concepts_migrated": 0,
                "concepts_merged": 0,
                "authors_checked": 0,
                "authors_fixed": 0,
                "authors_migrated": 0,
                "authors_merged": 0,
                "chunks_checked": 0,
                "chunks_fixed": 0,
            },
            "anomalies": {
                "papers": [],
                "concepts": [],
                "authors": [],
                "chunks": [],
            }
        }

        # 1. Fetch all nodes and classify
        all_nodes = self.graph_repo.get_all_nodes()
        papers_data = []
        concepts_data = []
        authors_data = []

        for node_id, label, properties_json in all_nodes:
            try:
                props = json.loads(properties_json) if properties_json else {}
            except Exception as e:
                logger.error(f"Error parsing properties JSON for node {node_id}: {e}")
                props = {}

            if label == "Paper":
                papers_data.append((node_id, props))
            elif label == "Concept":
                concepts_data.append((node_id, props))
            elif label == "Author":
                authors_data.append((node_id, props))

        # We will retrieve all edges to handle migrations
        all_edges = self.graph_repo.get_all_edges()

        # Helper to find edges connected to an ID
        def get_edges_for_node(node_id: str) -> List[Tuple[str, str, str, Dict[str, Any]]]:
            node_edges = []
            for src, tgt, etype, eprops_json in all_edges:
                if src == node_id or tgt == node_id:
                    try:
                        eprops = json.loads(eprops_json) if eprops_json else {}
                    except Exception:
                        eprops = {}
                    node_edges.append((src, tgt, etype, eprops))
            return node_edges

        # ── Step 1: Sanitize Authors ──────────────────────────────────────────
        report["stats"]["authors_checked"] = len(authors_data)
        for author_id, props in authors_data:
            name = props.get("name", author_id)
            normalized_name = self.normalizer.normalize_author_name(name)
            new_name = clean_text(normalized_name)

            if new_name != name:
                new_id = slugify(new_name)
                
                # Check if it requires ID migration
                if new_id != author_id:
                    # Check if destination already exists
                    dest_exists = any(node[0] == new_id for node in all_nodes)
                    if dest_exists:
                        report["stats"]["authors_merged"] += 1
                        report["anomalies"]["authors"].append({
                            "id": author_id,
                            "old_name": name,
                            "new_name": new_name,
                            "action": f"merged into existing ID '{new_id}'"
                        })
                        if fix:
                            # Migrate edges
                            edges = get_edges_for_node(author_id)
                            for src, tgt, etype, eprops in edges:
                                new_src = new_id if src == author_id else src
                                new_tgt = new_id if tgt == author_id else tgt
                                # Avoid duplicating self-loops or identical edges if they already exist
                                self.graph_repo.add_edge(new_src, new_tgt, etype, eprops)
                            self.graph_repo.delete_node(author_id)
                    else:
                        report["stats"]["authors_migrated"] += 1
                        report["anomalies"]["authors"].append({
                            "id": author_id,
                            "old_name": name,
                            "new_name": new_name,
                            "action": f"migrated to new ID '{new_id}'"
                        })
                        if fix:
                            # Create new node
                            new_props = {**props, "name": new_name}
                            new_author = Author(id=new_id, name=new_name, properties=new_props)
                            self.graph_repo.save_author(new_author)
                            
                            # Migrate edges
                            edges = get_edges_for_node(author_id)
                            for src, tgt, etype, eprops in edges:
                                new_src = new_id if src == author_id else src
                                new_tgt = new_id if tgt == author_id else tgt
                                self.graph_repo.add_edge(new_src, new_tgt, etype, eprops)
                            
                            # Delete old node
                            self.graph_repo.delete_node(author_id)
                else:
                    # Same ID, just update properties
                    report["stats"]["authors_fixed"] += 1
                    report["anomalies"]["authors"].append({
                        "id": author_id,
                        "old_name": name,
                        "new_name": new_name,
                        "action": "properties updated"
                    })
                    if fix:
                        new_props = {**props, "name": new_name}
                        self.graph_repo.update_node_properties(author_id, new_props)

        # ── Step 2: Sanitize Concepts ─────────────────────────────────────────
        report["stats"]["concepts_checked"] = len(concepts_data)
        for concept_id, props in concepts_data:
            name = props.get("name", concept_id)
            description = props.get("description", "")
            is_tag = props.get("is_tag", False)

            # Apply appropriate formatter
            if is_tag:
                normalized_name = self.normalizer.normalize_tag(name)
            else:
                normalized_name = self.normalizer.normalize_concept_name(name)

            new_name = clean_text(normalized_name)
            new_desc = clean_text(self.normalizer.normalize_description(description))

            if new_name != name or new_desc != description:
                new_id = slugify(new_name)
                
                # Check if it requires ID migration
                if new_id != concept_id:
                    # Check if destination already exists
                    dest_exists = any(node[0] == new_id for node in all_nodes)
                    if dest_exists:
                        report["stats"]["concepts_merged"] += 1
                        report["anomalies"]["concepts"].append({
                            "id": concept_id,
                            "old_name": name,
                            "new_name": new_name,
                            "old_description": description[:60] + "..." if description else "",
                            "new_description": new_desc[:60] + "..." if new_desc else "",
                            "action": f"merged into existing ID '{new_id}'"
                        })
                        if fix:
                            # Fetch existing node properties to merge description if needed
                            existing_label_props = self.graph_repo.get_node_by_id(new_id)
                            merged_desc = new_desc
                            if existing_label_props:
                                try:
                                    existing_props = json.loads(existing_label_props[1])
                                    merged_desc = existing_props.get("description") or new_desc
                                except Exception:
                                    pass
                            
                            # Update the existing node properties
                            updated_props = {
                                 **props, 
                                 "name": new_name, 
                                 "description": merged_desc
                            }
                            if self.emb_engine and new_name != name:
                                try:
                                    updated_props["embedding"] = self.emb_engine.get_embedding(new_name, is_query=False)
                                except Exception as ex:
                                    logger.error(f"Failed to regenerate embedding during concept merge: {ex}")
                            self.graph_repo.update_node_properties(new_id, updated_props)
                            
                            # Migrate edges
                            edges = get_edges_for_node(concept_id)
                            for src, tgt, etype, eprops in edges:
                                new_src = new_id if src == concept_id else src
                                new_tgt = new_id if tgt == concept_id else tgt
                                self.graph_repo.add_edge(new_src, new_tgt, etype, eprops)
                            
                            self.graph_repo.delete_node(concept_id)
                    else:
                        report["stats"]["concepts_migrated"] += 1
                        report["anomalies"]["concepts"].append({
                            "id": concept_id,
                            "old_name": name,
                            "new_name": new_name,
                            "old_description": description[:60] + "..." if description else "",
                            "new_description": new_desc[:60] + "..." if new_desc else "",
                            "action": f"migrated to new ID '{new_id}'"
                        })
                        if fix:
                            # Create new node
                            new_props = {**props, "name": new_name, "description": new_desc}
                            if self.emb_engine and new_name != name:
                                try:
                                    new_props["embedding"] = self.emb_engine.get_embedding(new_name, is_query=False)
                                except Exception as ex:
                                    logger.error(f"Failed to regenerate embedding during concept migration: {ex}")
                            new_concept = Concept(id=new_id, name=new_name, properties=new_props)
                            self.graph_repo.save_concept(new_concept)
                            
                            # Migrate edges
                            edges = get_edges_for_node(concept_id)
                            for src, tgt, etype, eprops in edges:
                                new_src = new_id if src == concept_id else src
                                new_tgt = new_id if tgt == concept_id else tgt
                                self.graph_repo.add_edge(new_src, new_tgt, etype, eprops)
                            
                            # Delete old node
                            self.graph_repo.delete_node(concept_id)
                else:
                    # Same ID, just update properties
                    report["stats"]["concepts_fixed"] += 1
                    report["anomalies"]["concepts"].append({
                        "id": concept_id,
                        "old_name": name,
                        "new_name": new_name,
                        "old_description": description[:60] + "..." if description else "",
                        "new_description": new_desc[:60] + "..." if new_desc else "",
                        "action": "properties updated"
                    })
                    if fix:
                        new_props = {**props, "name": new_name, "description": new_desc}
                        if self.emb_engine and new_name != name:
                            try:
                                new_props["embedding"] = self.emb_engine.get_embedding(new_name, is_query=False)
                            except Exception as ex:
                                logger.error(f"Failed to regenerate embedding during concept update: {ex}")
                        self.graph_repo.update_node_properties(concept_id, new_props)

        # ── Step 3: Sanitize Papers ───────────────────────────────────────────
        report["stats"]["papers_checked"] = len(papers_data)
        for paper_id, props in papers_data:
            title = props.get("title", "")
            abstract = props.get("abstract", "")
            authors = props.get("authors", [])
            summary = props.get("summary", "")
            is_placeholder = props.get("is_placeholder", False) or props.get("placeholder", False)

            new_title = clean_text(title)
            new_abstract = clean_text(abstract)
            
            # Clean and normalize author list inside paper properties
            new_authors = []
            for author in authors:
                norm_author = self.normalizer.normalize_author_name(author)
                new_authors.append(clean_text(norm_author))

            # Run NER to find missing authors
            final_authors = list(new_authors)
            try:
                # Find source text for NER
                text_for_ner = new_abstract
                if not text_for_ner:
                    chunks = self.vector_repo.get_chunks_for_paper(paper_id)
                    if chunks:
                        chunks.sort(key=lambda c: (c.page_number, c.id))
                        text_for_ner = "\n\n".join(c.text_content for c in chunks[:5])
                
                if text_for_ner:
                    from src.ner_engine import extract_persons_from_text
                    extracted_names = extract_persons_from_text(text_for_ner)
                    for name in extracted_names:
                        norm_name = clean_text(self.normalizer.normalize_author_name(name))
                        if norm_name and len(norm_name) > 2 and norm_name not in final_authors:
                            final_authors.append(norm_name)
            except Exception as e:
                logger.error(f"Error running NER on paper {paper_id}: {e}")

            has_author_enrichment = (final_authors != new_authors)
            has_formatting_anomaly = (new_title != title or new_abstract != abstract or new_authors != authors or has_author_enrichment)
            missing_abstract = not abstract and not is_placeholder
            missing_summary = not summary and not is_placeholder

            if has_formatting_anomaly or missing_abstract or missing_summary:
                report["stats"]["papers_fixed"] += 1
                anomaly_entry = {
                    "id": paper_id,
                    "old_title": title,
                    "new_title": new_title,
                    "old_abstract": abstract[:60] + "..." if abstract else "",
                    "new_abstract": new_abstract[:60] + "..." if new_abstract else "",
                    "old_authors": authors,
                    "new_authors": final_authors,
                    "missing_abstract": missing_abstract,
                    "missing_summary": missing_summary,
                }
                report["anomalies"]["papers"].append(anomaly_entry)

                if fix:
                    fixed_props = {
                        **props,
                        "title": new_title,
                        "abstract": new_abstract,
                        "authors": final_authors
                    }
                    updated = has_formatting_anomaly
                    paper_obj = None

                    if missing_abstract:
                        paper_obj = Paper(
                            id=paper_id,
                            title=fixed_props.get("title", ""),
                            authors=fixed_props.get("authors", []),
                            year=fixed_props.get("year"),
                            doi=fixed_props.get("doi"),
                            abstract=fixed_props.get("abstract"),
                            file_path=fixed_props.get("file_path"),
                            created_at=fixed_props.get("created_at"),
                            properties=fixed_props
                        )
                        from src.services.metadata_enricher import MetadataEnricher
                        enricher = MetadataEnricher()
                        try:
                            api_meta = enricher.enrich(paper_obj)
                            if api_meta:
                                paper_obj, _, _ = enricher.apply(paper_obj, api_meta)
                                if paper_obj.abstract:
                                    fixed_props["abstract"] = paper_obj.abstract
                                    updated = True
                                    missing_abstract = False
                                    anomaly_entry["generated_abstract"] = True
                        except Exception as e:
                            logger.error(f"Error enriching paper {paper_id}: {e}")

                        if missing_abstract and self.llm_engine:
                            chunks = self.vector_repo.get_chunks_for_paper(paper_id)
                            if chunks:
                                chunks.sort(key=lambda c: (c.page_number, c.id))
                                full_text = "\n\n".join(c.text_content for c in chunks)
                                if full_text:
                                    prompt = (
                                        "You are an expert scientific writing assistant. Based on the following text from a scientific paper, "
                                        "generate a concise, academic abstract (approx 150-250 words) that outlines the main goals, methods, and key findings of the work.\n\n"
                                        f"Title: {fixed_props.get('title')}\n\n"
                                        f"Paper Text:\n{full_text[:4000]}\n\n"
                                        "Abstract:"
                                    )
                                    temps = [0.7, 0.3, 0.0]
                                    generated_abstract = None
                                    for attempt, temp in enumerate(temps):
                                        try:
                                            resp = self.llm_engine.generate_response(prompt, task="synthesis", temp=temp)
                                            if resp:
                                                cleaned_resp = clean_text(resp)
                                                from src.llm_engine.base import validate_no_hallucinations
                                                validate_no_hallucinations(cleaned_resp)
                                                generated_abstract = cleaned_resp
                                                break
                                        except ValueError as ve:
                                            logger.warning(f"Hallucination detected in generated abstract on attempt {attempt + 1}: {ve}. Retrying...")
                                        except Exception as e:
                                            logger.error(f"Failed to generate abstract via LLM on attempt {attempt + 1}: {e}")
                                            
                                    if generated_abstract:
                                        fixed_props["abstract"] = generated_abstract
                                        paper_obj.abstract = generated_abstract
                                        updated = True
                                        missing_abstract = False
                                        anomaly_entry["generated_abstract"] = True

                    if missing_summary and self.llm_engine:
                        if not paper_obj:
                            paper_obj = Paper(
                                id=paper_id,
                                title=fixed_props.get("title", ""),
                                authors=fixed_props.get("authors", []),
                                year=fixed_props.get("year"),
                                doi=fixed_props.get("doi"),
                                abstract=fixed_props.get("abstract"),
                                file_path=fixed_props.get("file_path"),
                                created_at=fixed_props.get("created_at"),
                                properties=fixed_props
                            )
                        from src.services.extraction_service import ExtractionService
                        extractor = ExtractionService(llm_engine=self.llm_engine)
                        chunks = self.vector_repo.get_chunks_for_paper(paper_id)
                        if chunks:
                            chunks.sort(key=lambda c: (c.page_number, c.id))
                            full_text = "\n\n".join(c.text_content for c in chunks)
                        else:
                            full_text = ""
                        
                        temps = [0.7, 0.3, 0.0]
                        summary_text = None
                        for attempt, temp in enumerate(temps):
                            try:
                                resp = extractor.generate_summary(paper_obj, full_text, temp=temp)
                                if resp:
                                    cleaned_resp = clean_text(resp)
                                    from src.llm_engine.base import validate_no_hallucinations
                                    validate_no_hallucinations(cleaned_resp)
                                    summary_text = cleaned_resp
                                    break
                            except ValueError as ve:
                                logger.warning(f"Hallucination detected in generated summary on attempt {attempt + 1}: {ve}. Retrying...")
                            except Exception as e:
                                logger.error(f"Failed to generate summary on attempt {attempt + 1}: {e}")
                                
                        if summary_text:
                            fixed_props["summary"] = summary_text
                            if paper_obj.properties.get("source_type") == "video":
                                fixed_props["video_overview"] = paper_obj.properties.get("video_overview")
                                fixed_props["video_themes"] = paper_obj.properties.get("video_themes")
                                fixed_props["video_outline"] = paper_obj.properties.get("video_outline")
                            updated = True
                            anomaly_entry["generated_summary"] = True

                    if updated:
                        if "abstract" in fixed_props:
                            fixed_props["abstract"] = clean_text(fixed_props["abstract"])
                        if "summary" in fixed_props:
                            fixed_props["summary"] = clean_text(fixed_props["summary"])
                        self.graph_repo.update_node_properties(paper_id, fixed_props)
                        
                        # Save/create Author nodes and edges for any enriched authors
                        if has_author_enrichment:
                            for author_name in final_authors:
                                if author_name not in new_authors:
                                    author_id = slugify(author_name)
                                    new_author = Author(id=author_id, name=author_name, properties={"name": author_name})
                                    self.graph_repo.save_author(new_author)
                                    self.graph_repo.add_edge(author_id, paper_id, "AUTHORED")
                                    
                        anomaly_entry["new_abstract"] = fixed_props.get("abstract", "")[:60] + "..." if fixed_props.get("abstract") else ""

        # ── Step 4: Sanitize Chunks ───────────────────────────────────────────
        try:
            all_chunks = self.vector_repo.get_all_chunks()
        except Exception as e:
            logger.error(f"Error fetching chunks from vector repository: {e}")
            all_chunks = []

        report["stats"]["chunks_checked"] = len(all_chunks)
        chunks_to_save = []
        for chunk in all_chunks:
            text_content = chunk.text_content
            new_text = clean_text(text_content)
            if new_text != text_content:
                report["stats"]["chunks_fixed"] += 1
                report["anomalies"]["chunks"].append({
                    "id": chunk.id,
                    "paper_id": chunk.paper_id,
                    "old_text": text_content[:100] + "...",
                    "new_text": new_text[:100] + "..."
                })
                if fix:
                    chunk.text_content = new_text
                    chunks_to_save.append(chunk)

        if fix and chunks_to_save:
            # Save chunks in batches of 100
            for i in range(0, len(chunks_to_save), 100):
                self.vector_repo.save_chunks(chunks_to_save[i:i+100])

        return report
