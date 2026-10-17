import re
from typing import Tuple, List
from urllib.parse import urljoin
from bs4 import BeautifulSoup
try:
    import markdownify
except ImportError:
    markdownify = None

try:
    from curl_cffi import requests
except ImportError:
    requests = None
from src.models import Paper, slugify
from src import console as con
from src.parsers.base import BaseParser

class UrlParser(BaseParser):
    def parse(self, source: str) -> Tuple[Paper, List[str], str]:
        """
        Fetches a URL using curl_cffi to bypass bot protection, 
        parses HTML to markdown, and returns a Paper (note), list of references, and the markdown text.
        Enriches metadata (authors, title, year, abstract, DOI) using HTML meta tags
        and Semantic Scholar API.
        """
        url = source
        con.dim(f"Fetching URL with anti-detect browser: {url}")
        try:
            # impersonate="chrome120" makes the request look like a real Chrome browser
            response = requests.get(url, impersonate="chrome120", timeout=30)
            response.raise_for_status()
            html_content = response.text
        except Exception as e:
            raise RuntimeError(f"Failed to fetch URL {url}: {e}")

        # Parse HTML
        soup = BeautifulSoup(html_content, "html.parser")

        # 1. Try to extract identifiers from URL
        arxiv_id = None
        doi = None

        # Check for arXiv ID in URL
        arxiv_match = re.search(r'arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+(?:v[0-9]+)?)', url.lower())
        if arxiv_match:
            arxiv_id = arxiv_match.group(1)
            con.dim(f"Detected arXiv ID from URL: {arxiv_id}")

        # Check for DOI in URL
        doi_match = re.search(r'doi\.org/(10\.[0-9]{4,}(?:\.[0-9]+)*/(?:[^\s"<>]+))', url.lower())
        if doi_match:
            doi = doi_match.group(1)
            con.dim(f"Detected DOI from URL: {doi}")

        # 2. Extract metadata from HTML meta tags (standard academic tags)
        meta_title = soup.find("meta", attrs={"name": re.compile(r"^citation_title$", re.I)})
        meta_title_val = meta_title.get("content").strip() if meta_title and meta_title.get("content") else None

        # Authors
        meta_authors = []
        
        # 1. Standard academic tags
        for meta in soup.find_all("meta", attrs={"name": re.compile(r"^(citation_author|dc\.creator)$", re.I)}):
            val = meta.get("content")
            if val:
                name = val.strip()
                if name and name not in meta_authors:
                    meta_authors.append(name)

        # 2. General web metadata tags for authors
        author_meta_names = [
            "author", "article:author", "og:article:author", "twitter:creator", 
            "sailthru.author", "parsely-author", "author-name"
        ]
        for name_or_prop in author_meta_names:
            for meta in soup.find_all("meta", attrs={"name": name_or_prop}):
                val = meta.get("content")
                if val:
                    names = [n.strip() for n in re.split(r'[,;]|\band\b', val)]
                    for name in names:
                        if name and name not in meta_authors:
                            meta_authors.append(name)
            for meta in soup.find_all("meta", attrs={"property": name_or_prop}):
                val = meta.get("content")
                if val:
                    names = [n.strip() for n in re.split(r'[,;]|\band\b', val)]
                    for name in names:
                        if name and name not in meta_authors:
                            meta_authors.append(name)

        # 3. HTML microdata itemprop="author" or itemprop="creator"
        for el in soup.find_all(attrs={"itemprop": re.compile(r"^(author|creator)$", re.I)}):
            name_el = el.find(attrs={"itemprop": "name"})
            val = name_el.get_text() if name_el else el.get_text()
            if val:
                name = val.strip()
                if name and name not in meta_authors:
                    meta_authors.append(name)

        # 4. Common CSS classes for author name
        author_classes = [
            "author__name", "tm-user-info__username", "user-info__name", "author-name", 
            "author", "username", "creator", "tm-user-info__user", "post__author"
        ]
        for cls in author_classes:
            for el in soup.find_all(class_=re.compile(r'\b' + re.escape(cls) + r'\b', re.I)):
                val = el.get_text()
                if val:
                    name = val.strip()
                    name = re.sub(r'^(by\s+|автор:\s*)', '', name, flags=re.I).strip()
                    if name and len(name) < 50 and name not in meta_authors:
                        meta_authors.append(name)

        # 5. Search for profile links
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if re.search(r'/(users|author|user)/([^/]+)/?$', href.lower()):
                val = a.get_text()
                if val:
                    name = val.strip()
                    name = re.sub(r'^(by\s+|автор:\s*)', '', name, flags=re.I).strip()
                    if name and len(name) < 50 and name not in meta_authors:
                        meta_authors.append(name)

        # Clean up and filter extracted names
        cleaned_authors = []
        for a in meta_authors:
            a_clean = re.sub(r'\s+', ' ', a).strip()
            # If it's a URL, don't use it directly; extract the username/name
            if a_clean.startswith("http://") or a_clean.startswith("https://"):
                parts = [p for p in a_clean.split('/') if p]
                if parts:
                    a_clean = parts[-1]
                else:
                    continue
            # Check length constraints
            if not a_clean or len(a_clean) < 2 or len(a_clean) > 50:
                continue
            # Avoid common UI/garbage words
            if a_clean.lower() in ("login", "signin", "sign in", "sign up", "register", "comments", "reply", "subscribe", "anonymous", "admin", "administrator", "moderator"):
                continue
            # Filter out email-like or domain-like strings
            if "@" in a_clean or "." in a_clean and len(a_clean.split()) == 1 and not a_clean.startswith("http"):
                continue
            if a_clean not in cleaned_authors:
                cleaned_authors.append(a_clean)
                
        meta_authors = cleaned_authors

        # DOI meta tag
        meta_doi = soup.find("meta", attrs={"name": re.compile(r"^(citation_doi|dc\.identifier)$", re.I)})
        if meta_doi and meta_doi.get("content"):
            doi_val = meta_doi.get("content").strip()
            # Ensure it looks like a DOI
            if doi_val.startswith("10."):
                doi = doi_val
                con.dim(f"Detected DOI from meta tag: {doi}")

        # arXiv ID meta tag
        meta_arxiv = soup.find("meta", attrs={"name": re.compile(r"^citation_arxiv_id$", re.I)})
        if meta_arxiv and meta_arxiv.get("content"):
            arxiv_id = meta_arxiv.get("content").strip()
            con.dim(f"Detected arXiv ID from meta tag: {arxiv_id}")

        # Year
        meta_date = soup.find("meta", attrs={"name": re.compile(r"^(citation_date|citation_publication_date|dc\.date)$", re.I)})
        year_val = None
        if meta_date and meta_date.get("content"):
            date_str = meta_date.get("content").strip()
            year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
            if year_match:
                year_val = int(year_match.group(0))

        # Abstract
        meta_abstract = soup.find("meta", attrs={"name": re.compile(r"^(citation_abstract|description)$", re.I)})
        abstract_val = meta_abstract.get("content").strip() if meta_abstract and meta_abstract.get("content") else ""

        # Title
        title = meta_title_val or (soup.title.string.strip() if soup.title else url)

        # 3. Enrich using Semantic Scholar if identifier is available
        api_enriched = False
        if arxiv_id or doi:
            try:
                from src.external_api import fetch_paper_metadata
                con.dim("Fetching enriched metadata from Semantic Scholar …")
                api_meta = fetch_paper_metadata(doi=doi, arxiv_id=arxiv_id)
                if api_meta:
                    if api_meta.get("title"):
                        title = api_meta["title"]
                    if api_meta.get("authors"):
                        meta_authors = api_meta["authors"]
                    if api_meta.get("year"):
                        year_val = api_meta["year"]
                    if api_meta.get("abstract"):
                        abstract_val = api_meta["abstract"]
                    if api_meta.get("doi"):
                        doi = api_meta["doi"]
                    api_enriched = True
                    con.success(f"Enriched metadata for URL via Semantic Scholar: {title[:60]}")
            except Exception as e:
                con.warning(f"Could not enrich URL metadata via Semantic Scholar: {e}")

        # Clean up unnecessary tags before markdown conversion
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"]):
            element.decompose()

        # Convert to markdown
        main_content = soup.find("main") or soup.find("article") or soup.body
        if not main_content:
            main_content = soup
        
        # Extract links from main content
        links = []
        for a in main_content.find_all("a", href=True):
            href = a["href"].strip()
            if not href:
                continue
            absolute_url = urljoin(url, href)
            absolute_url_clean = absolute_url.split('#')[0]
            if not absolute_url_clean:
                continue
            if absolute_url_clean.startswith(("mailto:", "javascript:", "tel:")):
                continue
            url_compare = url.split('#')[0].rstrip('/')
            link_compare = absolute_url_clean.rstrip('/')
            if link_compare == url_compare:
                continue
            if absolute_url_clean not in links:
                links.append(absolute_url_clean)
        
        html_to_convert = str(main_content)
        
        md_content = markdownify.markdownify(
            html_to_convert,
            heading_style="ATX",
            bullets="-",
        ).strip()

        # Clean up excessive newlines
        md_content = re.sub(r'\n{3,}', '\n\n', md_content)

        # Generate a stable paper ID using slugify
        if arxiv_id:
            paper_id = f"arxiv_{slugify(arxiv_id)}"
        elif doi:
            paper_id = f"doi_{slugify(doi)}"
        else:
            paper_id = slugify(url.replace("https://", "").replace("http://", "").strip("/"))[:120]

        paper = Paper(
            id=paper_id,
            title=title,
            authors=meta_authors,
            year=year_val,
            abstract=abstract_val,
            doi=doi or "",
            file_path=url,
            properties={
                "source_type": "webpage" if not (arxiv_id or doi) else "paper",
                "url": url,
                "arxiv_id": arxiv_id or "",
                "api_enriched": api_enriched
            }
        )

        return paper, links, md_content

