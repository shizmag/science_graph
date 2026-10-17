import os
import re
import time
from typing import List, Tuple, Dict, Optional
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    class BM25Okapi:
        def __init__(self, *args, **kwargs):
            raise ImportError("rank_bm25 package is required for BM25 search.")
from src.config import config
from src.models import Chunk

class EmbeddingEngine:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.embedding_model_name
        self.model = None
        self.device = None
        self._query_cache = {}

    def _get_device(self) -> str:
        try:
            import torch
            if os.environ.get("ABLA_CUDA_REQUIRE_DEVICE") == "1":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is required for this run, but PyTorch cannot access a CUDA device")
                return "cuda"
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except (ImportError, AttributeError):
            pass
        return "cpu"

    def _ensure_model_loaded(self):
        if self.model is None:
            from src import console as con
            short = self.model_name.split("/")[-1]
            con.model_msg(f"Loading embeddings [bold]{short}[/bold] …")
            with con.suppress_stderr(), con.suppress_stdout():
                from sentence_transformers import SentenceTransformer
                self.device = self._get_device()
                self.model = SentenceTransformer(self.model_name, device=self.device)
                if os.environ.get("ABLA_CUDA_REQUIRE_DEVICE") == "1":
                    model_devices = {str(parameter.device) for parameter in self.model.parameters()}
                    if not model_devices or not all(device.startswith("cuda") for device in model_devices):
                        raise RuntimeError(f"E5 model did not load entirely on CUDA: {sorted(model_devices)}")
            con.success(f"Embeddings ready: [bold]{short}[/bold] on {self.device.upper()}")

    def _empty_device_cache(self, device: str):
        try:
            import torch
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
        except (ImportError, AttributeError):
            pass

    def unload_model(self):
        if self.model is not None:
            import gc
            self.model = None
            gc.collect()
            device_to_clear = self.device or self._get_device()
            self._empty_device_cache(device_to_clear)
            self.device = None
            from src import console as con
            con.success("EmbeddingEngine model unloaded and GPU cache cleared")

    def get_embedding(self, text: str, is_query: bool = True) -> List[float]:
        """Generates embedding for a single text string."""
        if is_query and text in self._query_cache:
            return self._query_cache[text]

        self._ensure_model_loaded()
        
        # Prepend query: or passage: prefix for E5 models if not already present
        if "e5" in self.model_name.lower():
            prefix = "query: " if is_query else "passage: "
            if not text.startswith(prefix):
                text = prefix + text

        # E5 inputs are cached under their prefixed form as well as the raw
        # query. Check this key before inference so later B6 arms share the
        # exact already computed embedding.
        if is_query and text in self._query_cache:
            return self._query_cache[text]
                
        emb = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        emb_list = emb.tolist()
        if is_query:
            self._query_cache[text] = emb_list
            self._query_cache[text.removeprefix("query: ")] = emb_list
        return emb_list

    def get_embeddings(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Generates embeddings for a list of text strings."""
        if not texts:
            return []

        # If is_query, check which texts are already cached
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []
        
        for idx, text in enumerate(texts):
            if is_query and text in self._query_cache:
                results[idx] = self._query_cache[text]
            else:
                uncached_indices.append(idx)
                uncached_texts.append(text)

        if uncached_texts:
            self._ensure_model_loaded()
            
            # Prepend query: or passage: prefix for E5 models if not already present
            processed_texts = []
            if "e5" in self.model_name.lower():
                prefix = "query: " if is_query else "passage: "
                for t in uncached_texts:
                    if not t.startswith(prefix):
                        processed_texts.append(prefix + t)
                    else:
                        processed_texts.append(t)
            else:
                processed_texts = uncached_texts

            # Resolve prefixed cache keys before encode; otherwise raw text
            # from another arm would trigger the same E5 CUDA inference again.
            if is_query:
                still_uncached = []
                still_indices = []
                for idx, processed in zip(uncached_indices, processed_texts):
                    if processed in self._query_cache:
                        results[idx] = self._query_cache[processed]
                    else:
                        still_indices.append(idx)
                        still_uncached.append(processed)
                uncached_indices = still_indices
                processed_texts = still_uncached

            embs_list = []
            if processed_texts:
                embs = self.model.encode(processed_texts, convert_to_numpy=True, show_progress_bar=False)
                embs_list = embs.tolist()
            
            for idx, emb_val in zip(uncached_indices, embs_list):
                results[idx] = emb_val
                if is_query:
                    self._query_cache[texts[idx]] = emb_val
                    self._query_cache[processed_texts[uncached_indices.index(idx)]] = emb_val
                    
        return results

def split_segment_to_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if not text or not text.strip():
        return []
        
    # Prevent infinite loops if overlap is greater than or equal to chunk size
    chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))
        
    # Pre-extract non-splittable structures (Markdown tables & LaTeX equations)
    table_pattern = re.compile(r'((?:\n\|[^\n]+\|\s*)+)', re.MULTILINE)
    tables = table_pattern.findall(text)
    
    equation_pattern = re.compile(
        r'(\$\$.*?\$\$|\\begin\{equation\}.*?\\end\{equation\}|\\\[.*?\\\]|\\\(.*?\\\))', 
        re.DOTALL
    )
    equations = equation_pattern.findall(text)
    
    non_splittable = []
    text_clean = text
    
    for idx, table in enumerate(tables):
        placeholder = f" __STRUCT_BLOCK_TABLE_{idx}__ "
        non_splittable.append((placeholder.strip(), table))
        text_clean = text_clean.replace(table, placeholder)
        
    for idx, eq in enumerate(equations):
        placeholder = f" __STRUCT_BLOCK_EQ_{idx}__ "
        non_splittable.append((placeholder.strip(), eq))
        text_clean = text_clean.replace(eq, placeholder)

    inline_dollar_pattern = re.compile(r'\$([^\$\n]+)\$')
    inline_dollar_idx = 0
    
    def is_valid_latex_inline(content: str) -> bool:
        if not content.strip():
            return False
        if re.match(r'^\s*\d+[\d\s\.,]*(?:[kKmMbB]|\s*billion|\s*million)?\s*$', content):
            return False
        if ' ' in content:
            math_indicators = r'[\+\-\*/=\\_^<>\(\)\[\]\{\}\\\|~]'
            if not re.search(math_indicators, content):
                return False
        return True

    def dollar_replacer(match):
        nonlocal inline_dollar_idx
        content = match.group(1)
        if is_valid_latex_inline(content):
            placeholder = f" __STRUCT_BLOCK_INLINE_DOLLAR_{inline_dollar_idx}__ "
            non_splittable.append((placeholder.strip(), match.group(0)))
            inline_dollar_idx += 1
            return placeholder
        return match.group(0)

    text_clean = inline_dollar_pattern.sub(dollar_replacer, text_clean)
    text_clean = re.sub(r'\s+', ' ', text_clean).strip()
    
    if not text_clean:
        return []
        
    results = []
    if len(text_clean) <= chunk_size:
        results.append(text_clean)
    else:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text_clean) if s.strip()]
        
        current_chunk = []
        current_len = 0
        for sentence in sentences:
            sentence_len = len(sentence)
            if current_len + sentence_len > chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                results.append(chunk_text)
                
                # Setup overlap sentences
                overlap_sentences = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    s_len = len(s)
                    if overlap_len + s_len + (1 if overlap_sentences else 0) <= chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += s_len + (1 if overlap_sentences else 0)
                    else:
                        break
                current_chunk = overlap_sentences + [sentence]
                current_len = sum(len(s) for s in current_chunk) + len(current_chunk) - 1
            else:
                if sentence_len > chunk_size:
                    start = 0
                    while start < sentence_len:
                        end = start + chunk_size
                        sub_text = sentence[start:end].strip()
                        if sub_text:
                            results.append(sub_text)
                        start += (chunk_size - chunk_overlap)
                    current_chunk = []
                    current_len = 0
                else:
                    current_chunk.append(sentence)
                    current_len += sentence_len + (1 if len(current_chunk) > 1 else 0)
                
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            results.append(chunk_text)

    # Substitute non-splittable structures back into chunks
    final_results = []
    for chunk_text in results:
        for placeholder, block in non_splittable:
            if placeholder in chunk_text:
                chunk_text = chunk_text.replace(placeholder, block)
        final_results.append(chunk_text)
        
    return final_results


def split_text_to_chunks(paper_id: str, file_path: str, chunk_size: int = None, chunk_overlap: int = None) -> List[Chunk]:
    """
    Reads PDF page by page and splits it into parent-child overlapping text chunks.
    Parent chunks are sized 2000-3000 chars. Child chunks are sized 200-400 chars.
    Preserves page number references and table/LaTeX structures.
    """
    import logging
    import fitz
    t0 = time.perf_counter()
    
    parent_size = chunk_size or getattr(config, "parent_chunk_size", 2500)
    parent_overlap = chunk_overlap if chunk_overlap is not None else getattr(config, "parent_chunk_overlap", 200)
    
    child_size = getattr(config, "child_chunk_size", 300)
    child_overlap = getattr(config, "child_chunk_overlap", 50)
    
    doc = fitz.open(file_path)
    child_chunks = []
    global_child_idx = 0
    parent_idx = 0
    
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if not text or not text.strip():
            continue
            
        parent_texts = split_segment_to_chunks(text, parent_size, parent_overlap)
        
        for p_text in parent_texts:
            parent_id = f"{paper_id}#parent_{parent_idx}"
            parent_idx += 1
            
            child_texts = split_segment_to_chunks(p_text, child_size, child_overlap)
            
            for c_text in child_texts:
                child_chunks.append(Chunk(
                    id=f"{paper_id}#{global_child_idx}",
                    paper_id=paper_id,
                    text_content=c_text,
                    page_number=page_num,
                    parent_id=parent_id,
                    parent_text=p_text
                ))
                global_child_idx += 1
                
    dt = time.perf_counter() - t0
    logging.debug(f"split_text_to_chunks for paper {paper_id} completed in {dt:.6f}s (generated {len(child_chunks)} child chunks from {parent_idx} parent chunks)")
    return child_chunks


class BM25(BM25Okapi):
    """
    A long-lived, rank_bm25-based BM25 implementation supporting incremental updates.
    """
    def __init__(self, corpus: List[Tuple[str, str]], k1: Optional[float] = None, b: Optional[float] = None):
        # Store original mapping
        self.doc_ids: List[str] = []
        self.doc_texts: Dict[str, str] = {}
        self.nd: Dict[str, int] = {}
        
        k1_val = k1 if k1 is not None else config.bm25_k1
        b_val = b if b is not None else config.bm25_b
        
        tokenized_corpus = []
        for doc_id, text in corpus:
            self.doc_ids.append(doc_id)
            self.doc_texts[doc_id] = text
            words = self._tokenize(text)
            tokenized_corpus.append(words)
            
        super().__init__(tokenized_corpus, k1=k1_val, b=b_val)
        
        # Populating self.nd from base class's doc_freqs
        for tf in self.doc_freqs:
            for word in tf.keys():
                self.nd[word] = self.nd.get(word, 0) + 1

    def _tokenize(self, text: str) -> List[str]:
        return [w for w in re.findall(r'\w+', text.lower()) if w]

    def add_documents(self, documents: List[Tuple[str, str]]) -> None:
        """
        Incrementally updates the corpus without full re-tokenization.
        """
        if not documents:
            return
            
        for doc_id, text in documents:
            if doc_id in self.doc_texts:
                continue  # Avoid duplicate document IDs
            self.doc_ids.append(doc_id)
            self.doc_texts[doc_id] = text
            
            words = self._tokenize(text)
            self.doc_len.append(len(words))
            self.corpus_size += 1
            
            frequencies = {}
            for word in words:
                frequencies[word] = frequencies.get(word, 0) + 1
            self.doc_freqs.append(frequencies)
            
            for word in frequencies.keys():
                self.nd[word] = self.nd.get(word, 0) + 1
                
        total_len = sum(self.doc_len)
        self.avgdl = total_len / self.corpus_size if self.corpus_size > 0 else 0
        self._calc_idf(self.nd)

    def score(self, query: str) -> List[Tuple[str, float]]:
        """
        Scores all documents in the corpus for the given query.
        Returns a sorted list of (doc_id, score) tuples in descending order.
        """
        import time
        import logging
        t0 = time.perf_counter()
        
        tokenized_query = self._tokenize(query)
        scores = self.get_scores(tokenized_query)
        results = [(self.doc_ids[i], float(scores[i])) for i in range(len(self.doc_ids))]
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        
        dt = time.perf_counter() - t0
        logging.debug(f"BM25 score completed in {dt:.6f}s for query: {query!r}")
        return sorted_results
