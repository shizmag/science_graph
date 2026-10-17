import math
from typing import List, Dict, Any, Set
from src.repository.base import GraphRepository

class GraphConceptRetriever:
    def __init__(self, graph_repo: GraphRepository):
        self.graph_repo = graph_repo

    def retrieve(
        self,
        query: str,
        query_concepts: List[str],
        exclude_paper_ids: List[str],
        max_candidate_papers: Any,
        dropped_concepts_not_used: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves paper candidates that mention query concepts, ranked by concept IDF.
        """
        if not query_concepts:
            return []

        # Resolve max_candidate_papers
        limit = max_candidate_papers
        if limit == "auto":
            limit = 0  # no seed papers => no expansion
        elif not isinstance(limit, int):
            try:
                limit = int(limit)
            except Exception:
                limit = 0

        if limit <= 0:
            return []

        exclude_set = set(exclude_paper_ids)

        # 1. Fetch papers mentioning query concepts
        paper_tuples = self.graph_repo.get_papers_mentioning_concepts(query_concepts)
        
        # Filter matched paper IDs
        matched_paper_ids = [p[0] for p in paper_tuples if p[0] not in exclude_set]
        if not matched_paper_ids:
            return []
            
        # Group matched concepts by paper_id
        paper_concepts: Dict[str, Set[str]] = {}
        paper_titles: Dict[str, str] = {}
        
        concepts_for_papers = self.graph_repo.get_concepts_for_papers(matched_paper_ids)
        for paper_id, concept_id, concept_name in concepts_for_papers:
            if concept_id in query_concepts:
                if paper_id not in paper_concepts:
                    paper_concepts[paper_id] = set()
                paper_concepts[paper_id].add(concept_id)

        for paper_id, title in paper_tuples:
            if paper_id in matched_paper_ids:
                paper_titles[paper_id] = title

        # 2. Get document frequencies and total paper count for IDF
        doc_freqs = self.graph_repo.get_concept_document_frequencies(query_concepts)
        total_papers = self.graph_repo.get_total_paper_count()

        # Compute IDF for each query concept
        # idf(concept) = log((1 + total_papers) / (1 + document_frequency(concept)))
        idfs: Dict[str, float] = {}
        for c in query_concepts:
            df = doc_freqs.get(c, 0)
            idfs[c] = math.log((1 + total_papers) / (1 + df))

        # 3. Score candidates
        candidates = []
        for paper_id in matched_paper_ids:
            matched_concepts = list(paper_concepts.get(paper_id, set()))
            if not matched_concepts:
                continue
            concept_idf_sum = sum(idfs[c] for c in matched_concepts)
            
            candidates.append({
                "paper_id": paper_id,
                "title": paper_titles.get(paper_id, ""),
                "matched_concepts": matched_concepts,
                "concept_idf_sum": concept_idf_sum,
                "matched_concepts_count": len(matched_concepts),
            })

        # 4. Sort stable: tuple-based
        candidates.sort(key=lambda x: (
            -x["matched_concepts_count"],
            -x["concept_idf_sum"],
            x["paper_id"]
        ))

        # 5. Apply limit and build output reason metadata
        output = []
        for c in candidates[:limit]:
            output.append({
                "source": "graph_concept_retrieval",
                "paper_id": c["paper_id"],
                "title": c["title"],
                "matched_concepts": sorted(c["matched_concepts"]),
                "matched_strong_concepts": sorted(c["matched_concepts"]),
                "concept_idf_sum": c["concept_idf_sum"],
                "dropped_concepts_not_used": sorted(dropped_concepts_not_used or []),
                "reason": "paper_mentions_query_concept"
            })
            
        return output


class GraphBridgeRetriever:
    def __init__(self, graph_repo: GraphRepository):
        self.graph_repo = graph_repo

    def retrieve(
        self,
        query: str,
        seed_paper_ids: List[str],
        query_concepts: List[str],
        exclude_paper_ids: List[str],
        max_candidate_papers: Any
    ) -> List[Dict[str, Any]]:
        """
        Retrieves bridge paper candidates linked to seed papers and query concepts.
        """
        if not seed_paper_ids or not query_concepts:
            return []

        limit = max_candidate_papers
        if limit == "auto":
            limit = len(seed_paper_ids)
        elif not isinstance(limit, int):
            try:
                limit = int(limit)
            except Exception:
                limit = 0

        if limit <= 0:
            return []

        exclude_set = set(exclude_paper_ids)
        seed_set = set(seed_paper_ids)

        # 1. Fetch concepts for all seed papers
        seed_concepts_list = self.graph_repo.get_concepts_for_papers(seed_paper_ids)
        seed_concepts: Dict[str, Set[str]] = {}
        for paper_id, concept_id, _ in seed_concepts_list:
            if paper_id not in seed_concepts:
                seed_concepts[paper_id] = set()
            seed_concepts[paper_id].add(concept_id)

        # All concepts mentioned by at least one seed paper
        all_seed_concepts = set()
        for s_id in seed_paper_ids:
            all_seed_concepts.update(seed_concepts.get(s_id, set()))

        # 2. Fetch citation neighbors of all seed papers
        citation_neighbors = self.graph_repo.get_citation_neighbors(seed_paper_ids)
        
        # 3. Find candidates mentioning query concepts
        query_concept_papers = self.graph_repo.get_papers_mentioning_concepts(query_concepts)
        query_concept_paper_ids = {p[0] for p in query_concept_papers if p[0] not in exclude_set and p[0] not in seed_set}

        # Concept neighbors of seed papers for Condition 3 (bridge between two seed papers)
        concept_neighbor_papers = self.graph_repo.get_papers_mentioning_concepts(list(all_seed_concepts))
        concept_neighbor_paper_ids = {p[0] for p in concept_neighbor_papers if p[0] not in exclude_set and p[0] not in seed_set}
        
        # Combine candidate IDs to fetch their concept associations
        all_candidates_to_fetch = list(query_concept_paper_ids.union(concept_neighbor_paper_ids))
        
        # Include citation neighbors
        citation_neighbor_ids = {r[1] for r in citation_neighbors if r[1] not in exclude_set and r[1] not in seed_set}
        all_candidates_to_fetch = list(set(all_candidates_to_fetch).union(citation_neighbor_ids))

        # Title map
        candidate_titles: Dict[str, str] = {}
        for paper_id, title in query_concept_papers:
            candidate_titles[paper_id] = title
        for paper_id, title in concept_neighbor_papers:
            candidate_titles[paper_id] = title
        for _, neighbor_id, _, title in citation_neighbors:
            candidate_titles[neighbor_id] = title

        candidate_concepts: Dict[str, Set[str]] = {}
        if all_candidates_to_fetch:
            concepts_for_candidates = self.graph_repo.get_concepts_for_papers(all_candidates_to_fetch)
            for paper_id, concept_id, _ in concepts_for_candidates:
                if paper_id not in candidate_concepts:
                    candidate_concepts[paper_id] = set()
                candidate_concepts[paper_id].add(concept_id)

        # 4. Calculate IDF for covered query concepts
        doc_freqs = self.graph_repo.get_concept_document_frequencies(query_concepts)
        total_papers = self.graph_repo.get_total_paper_count()
        idfs: Dict[str, float] = {}
        for c in query_concepts:
            df = doc_freqs.get(c, 0)
            idfs[c] = math.log((1 + total_papers) / (1 + df))

        # Evaluate each candidate paper to see if it meets at least one bridge condition
        candidate_paths: Dict[str, List[Dict[str, Any]]] = {}

        for cand_id in all_candidates_to_fetch:
            cand_concepts = candidate_concepts.get(cand_id, set())
            covered_strong_query_concepts = cand_concepts.intersection(query_concepts)
            
            # Bridge candidate must contain at least one strong query concept
            if len(covered_strong_query_concepts) == 0:
                continue

            paths = []

            # Condition 1: seed_shared_strong_query_concept & seed_shared_query_concept
            for seed_id in seed_paper_ids:
                shared_query_concepts = cand_concepts.intersection(seed_concepts.get(seed_id, set())).intersection(query_concepts)
                for q_concept in shared_query_concepts:
                    paths.append({
                        "type": "seed_shared_query_concept",
                        "seed_paper_id": seed_id,
                        "concept_id": q_concept,
                        "candidate_paper_id": cand_id
                    })
                    paths.append({
                        "type": "seed_shared_strong_query_concept",
                        "seed_paper_id": seed_id,
                        "concept_id": q_concept,
                        "candidate_paper_id": cand_id
                    })

            # Condition 2: seed_citation_neighbor_with_strong_query_concept & seed_citation_neighbor_with_query_concept
            for seed_id, neighbor_id, direction, _ in citation_neighbors:
                if neighbor_id == cand_id:
                    shared_query_concepts = cand_concepts.intersection(query_concepts)
                    for q_concept in shared_query_concepts:
                        paths.append({
                            "type": "seed_citation_neighbor_with_query_concept",
                            "seed_paper_id": seed_id,
                            "candidate_paper_id": cand_id,
                            "concept_id": q_concept,
                            "direction": direction
                        })
                        paths.append({
                            "type": "seed_citation_neighbor_with_strong_query_concept",
                            "seed_paper_id": seed_id,
                            "candidate_paper_id": cand_id,
                            "concept_id": q_concept,
                            "direction": direction
                        })

            # Condition 3: candidate_bridges_two_seed_papers_via_concept & seed_shared_concept
            for concept_id in cand_concepts:
                matching_seeds = [s_id for s_id in seed_paper_ids if concept_id in seed_concepts.get(s_id, set())]
                if len(matching_seeds) >= 2:
                    for seed_id in matching_seeds:
                        paths.append({
                            "type": "seed_shared_concept",
                            "seed_paper_id": seed_id,
                            "concept_id": concept_id,
                            "candidate_paper_id": cand_id
                        })
                        paths.append({
                            "type": "candidate_bridges_two_seed_papers_via_concept",
                            "seed_paper_id": seed_id,
                            "concept_id": concept_id,
                            "candidate_paper_id": cand_id
                        })

            if paths:
                candidate_paths[cand_id] = paths

        # 5. Build candidate scoring structures
        scored_candidates = []
        for cand_id, paths in candidate_paths.items():
            cand_concepts = candidate_concepts.get(cand_id, set())
            covered_strong_query_concepts = cand_concepts.intersection(query_concepts)
            connected_seed_papers = {p["seed_paper_id"] for p in paths}
            
            min_dist = 2
            for p in paths:
                if p["type"] in ("seed_citation_neighbor_with_query_concept", "seed_citation_neighbor_with_strong_query_concept"):
                    min_dist = 1
                    break
                    
            concept_idf_sum = sum(idfs.get(c, 0.0) for c in covered_strong_query_concepts)

            # Sort paths to be deterministic
            paths.sort(key=lambda p: (
                p["type"],
                p["seed_paper_id"],
                p.get("concept_id", ""),
                p.get("direction", "")
            ))

            scored_candidates.append({
                "paper_id": cand_id,
                "title": candidate_titles.get(cand_id, ""),
                "covered_strong_query_concepts": list(covered_strong_query_concepts),
                "matched_concepts": list(covered_strong_query_concepts),
                "connected_seed_papers": list(connected_seed_papers),
                "min_graph_distance": min_dist,
                "concept_idf_sum": concept_idf_sum,
                "paths": paths,
            })

        # 6. Ranking without manual weights
        scored_candidates.sort(key=lambda x: (
            -len(x["covered_strong_query_concepts"]),
            -len(x["connected_seed_papers"]),
            x["min_graph_distance"],
            -x["concept_idf_sum"],
            x["paper_id"]
        ))

        # 7. Format output reason metadata
        output = []
        for c in scored_candidates[:limit]:
            limited_paths = c["paths"][:10]  # Allow up to 10 paths for combined aliases
            output.append({
                "source": "graph_bridge_retrieval",
                "paper_id": c["paper_id"],
                "title": c["title"],
                "connected_seed_papers": sorted(c["connected_seed_papers"]),
                "covered_strong_query_concepts": sorted(c["covered_strong_query_concepts"]),
                "matched_concepts": sorted(c["covered_strong_query_concepts"]),
                "concept_idf_sum": c["concept_idf_sum"],
                "min_graph_distance": c["min_graph_distance"],
                "paths": limited_paths
            })

        return output
