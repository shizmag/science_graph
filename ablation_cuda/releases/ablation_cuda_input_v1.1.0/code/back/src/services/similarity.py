import re
from typing import List, Set, Tuple
from src.models import slugify

class JaccardSimilarity:
    """
    Utility class for calculating Jaccard similarity between texts, authors,
    and performing paragraph-level deduplication across chunks.
    """

    @staticmethod
    def get_3_shingles(text: str) -> Set[Tuple[str, str, str]]:
        """Splits text into 3-word shingles."""
        if not text:
            return set()
        words = re.findall(r'\b\w+\b', text.lower())
        shingles = set()
        for i in range(len(words) - 2):
            shingles.add((words[i], words[i+1], words[i+2]))
        return shingles

    @classmethod
    def shingle_jaccard_similarity(cls, text1: str, text2: str) -> float:
        """Calculates Jaccard similarity based on 3-word shingles."""
        shingles1 = cls.get_3_shingles(text1)
        shingles2 = cls.get_3_shingles(text2)
        if not shingles1 and not shingles2:
            return 1.0
        if not shingles1 or not shingles2:
            return 0.0
        intersection = len(shingles1.intersection(shingles2))
        union = len(shingles1.union(shingles2))
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def word_jaccard_similarity(text1: str, text2: str) -> float:
        """Calculates Jaccard similarity based on individual words."""
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        return len(words1.intersection(words2)) / len(words1.union(words2))

    @staticmethod
    def author_jaccard_similarity(authors1: List[str], authors2: List[str]) -> float:
        """Calculates Jaccard similarity between two lists of author names."""
        set1 = {slugify(a) for a in authors1 if slugify(a)}
        set2 = {slugify(a) for a in authors2 if slugify(a)}
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        return len(set1.intersection(set2)) / len(set1.union(set2))

    @classmethod
    def deduplicate_chunks_paragraph_level(
        cls,
        chunks: List[str],
        chunk_similarity_threshold: float = 0.3,
        paragraph_similarity_threshold: float = 0.8
    ) -> List[str]:
        """
        Deduplicates a list of text chunks at the paragraph level.
        
        1. Compares chunks pairwise to find highly similar chunks (Jaccard >= chunk_similarity_threshold).
        2. If chunks are similar, removes duplicate paragraphs (either exact match or Jaccard similarity >= paragraph_similarity_threshold)
           from subsequent chunks.
        """
        if not chunks:
            return []

        # Keep track of all unique paragraphs accepted so far
        accepted_paragraphs: List[str] = []
        # Precomputed shingles for accepted paragraphs to avoid re-calculation
        accepted_shingles: List[Set[Tuple[str, str, str]]] = []
        
        deduplicated_chunks: List[str] = []

        for chunk in chunks:
            paragraphs = [p.strip() for p in chunk.split("\n\n") if p.strip()]
            remaining_paragraphs = []
            
            for para in paragraphs:
                # 1. Exact match check (O(1) fast path)
                if para in accepted_paragraphs:
                    continue
                
                # 2. Check similarity against already accepted paragraphs
                is_duplicate = False
                para_shingles = cls.get_3_shingles(para)
                
                for idx, accepted_p in enumerate(accepted_paragraphs):
                    # Quick check: if the lengths are very different, they aren't duplicates
                    # A basic length heuristic: if length ratio < 0.5 or > 2.0, skip Jaccard check
                    len_ratio = len(para) / max(len(accepted_p), 1)
                    if len_ratio < 0.5 or len_ratio > 2.0:
                        continue
                        
                    # Calculate Jaccard similarity using shingles
                    sh1 = para_shingles
                    sh2 = accepted_shingles[idx]
                    
                    if not sh1 and not sh2:
                        # Both are extremely short/empty of shingles but weren't exact matches
                        continue
                        
                    intersection = len(sh1.intersection(sh2))
                    union = len(sh1.union(sh2))
                    jaccard = intersection / union if union > 0 else 0.0
                    
                    if jaccard >= paragraph_similarity_threshold:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    remaining_paragraphs.append(para)
                    accepted_paragraphs.append(para)
                    accepted_shingles.append(para_shingles)

            if remaining_paragraphs:
                deduplicated_chunks.append("\n\n".join(remaining_paragraphs))

        return deduplicated_chunks
