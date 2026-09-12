#!/usr/bin/env python3
"""Verify every BibTeX record against Crossref, OpenAlex, and official pages."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import html
import json
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

import requests


ROOT = Path(__file__).resolve().parents[1]
BIB_PATH = ROOT / "paper" / "references.bib"
REPORT_DATE = "2026-09-11"
JSON_OUTPUT = ROOT / "paper_support" / f"reference_verification_{REPORT_DATE}.json"
MARKDOWN_OUTPUT = ROOT / "artifacts" / "paper_reports" / f"reference_verification_{REPORT_DATE}.md"
CACHE_DIR = ROOT / "artifacts" / "reference_verification_cache"
USER_AGENT = (
    "CHI2027-reference-verifier/1.0 "
    "(anonymous scholarly metadata audit; contact=anonymous@example.invalid)"
)
EXPECTED_ENTRIES = 86


def find_balanced(text: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f"Unbalanced {opening}{closing} beginning at offset {start}")


def parse_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    index = 0
    while index < len(body):
        while index < len(body) and (body[index].isspace() or body[index] == ","):
            index += 1
        match = re.match(r"[A-Za-z][A-Za-z0-9_-]*", body[index:])
        if not match:
            break
        name = match.group(0).lower()
        index += len(match.group(0))
        while index < len(body) and body[index].isspace():
            index += 1
        if index >= len(body) or body[index] != "=":
            raise ValueError(f"Missing equals sign after field {name}")
        index += 1
        while index < len(body) and body[index].isspace():
            index += 1
        if index >= len(body):
            raise ValueError(f"Missing value for field {name}")
        if body[index] == "{":
            end = find_balanced(body, index, "{", "}")
            value = body[index + 1 : end]
            index = end + 1
        elif body[index] == '"':
            end = index + 1
            escaped = False
            while end < len(body):
                if body[end] == '"' and not escaped:
                    break
                escaped = body[end] == "\\" and not escaped
                if body[end] != "\\":
                    escaped = False
                end += 1
            if end >= len(body):
                raise ValueError(f"Unbalanced quote for field {name}")
            value = body[index + 1 : end]
            index = end + 1
        else:
            end = body.find(",", index)
            if end < 0:
                end = len(body)
            value = body[index:end]
            index = end
        fields[name] = value.strip()
    return fields


def parse_bibtex(text: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for match in re.finditer(r"(?m)^@([A-Za-z]+)\s*\{", text):
        start = match.end() - 1
        end = find_balanced(text, start, "{", "}")
        inner = text[start + 1 : end]
        comma = inner.find(",")
        if comma < 0:
            raise ValueError(f"Malformed BibTeX entry near offset {start}")
        key = inner[:comma].strip()
        entries.append(
            {
                "type": match.group(1).lower(),
                "key": key,
                "fields": parse_fields(inner[comma + 1 :]),
            }
        )
    keys = [entry["key"] for entry in entries]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate BibTeX keys detected")
    return entries


LATEX_REPLACEMENTS = {
    r"\&": "&",
    r"\L": "L",
    r"\l": "l",
    r"\o": "o",
    r"\O": "O",
    r"\ss": "ss",
    "---": "-",
    "--": "-",
}


def strip_latex(value: str) -> str:
    text = value
    text = re.sub(r"</?[A-Za-z][^>]*>", "", text)
    for source, target in LATEX_REPLACEMENTS.items():
        text = text.replace(source, target)
    text = re.sub(r"\\(?:\"|'|\^|~|=)\s*\{?([A-Za-z])\}?", r"\1", text)
    text = re.sub(r"\$\^\{?\\circ\}?\$", " degree ", text)
    text = re.sub(r"\\[A-Za-z]+\s*", " ", text)
    text = text.replace("{", "").replace("}", "").replace("$", "")
    return html.unescape(text).strip()


def normalized(value: Any) -> str:
    if value is None:
        return ""
    text = strip_latex(str(value))
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def title_similarity(left: Any, right: Any) -> float:
    first = normalized(left)
    second = normalized(right)
    if not first or not second:
        return 0.0
    score = difflib.SequenceMatcher(None, first, second).ratio()
    shorter, longer = sorted((first, second), key=len)
    if len(shorter) >= 18 and longer.startswith(shorter):
        score = max(score, 0.97)
    return score


def split_bib_authors(value: str) -> list[str]:
    parts = re.split(r"\s+and\s+", value)
    authors: list[str] = []
    for part in parts:
        clean = strip_latex(part).strip()
        if clean.startswith("{") and clean.endswith("}"):
            clean = clean[1:-1]
        authors.append(clean)
    return [author for author in authors if author]


def family_name(value: str) -> str:
    clean = strip_latex(value).strip()
    if "," in clean:
        return clean.split(",", 1)[0].strip()
    return clean.split()[-1] if clean.split() else ""


def same_family(left: str, right: str) -> bool:
    first = normalized(family_name(left))
    second = normalized(family_name(right))
    if not first or not second:
        return False
    return first == second or first.endswith(second) or second.endswith(first)


def cache_path(kind: str, url: str, params: dict[str, Any] | None) -> Path:
    serialized = url
    if params:
        serialized += "?" + urlencode(sorted((key, str(value)) for key, value in params.items()))
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    return CACHE_DIR / f"{kind}_{digest}.json"


def request_json(
    url: str,
    params: dict[str, Any] | None,
    refresh: bool,
    retry_cached_failure: bool = False,
) -> dict[str, Any]:
    path = cache_path("json", url, params)
    if path.exists() and not refresh:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("ok") or not retry_cached_failure:
            return cached
    result: dict[str, Any] = {
        "ok": False,
        "request_url": url,
        "status_code": None,
        "data": None,
        "error": None,
    }
    for attempt in range(3):
        try:
            response = requests.get(
                url,
                params=params,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                timeout=25,
            )
            result["request_url"] = response.url
            result["status_code"] = response.status_code
            if response.status_code == 200:
                result["ok"] = True
                result["data"] = response.json()
                result["error"] = None
                break
            result["error"] = f"HTTP {response.status_code}"
            if response.status_code not in {429, 500, 502, 503, 504}:
                break
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        time.sleep(0.6 * (attempt + 1))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result["fetched_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def request_json_post(
    url: str,
    params: dict[str, Any],
    body: dict[str, Any],
    refresh: bool,
) -> dict[str, Any]:
    cache_params = dict(params)
    cache_params["body"] = json.dumps(body, sort_keys=True)
    path = cache_path("post", url, cache_params)
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "ok": False,
        "request_url": url,
        "status_code": None,
        "data": None,
        "error": None,
    }
    for attempt in range(3):
        try:
            response = requests.post(
                url,
                params=params,
                json=body,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                timeout=35,
            )
            result["request_url"] = response.url
            result["status_code"] = response.status_code
            if response.status_code == 200:
                result["ok"] = True
                result["data"] = response.json()
                result["error"] = None
                break
            result["error"] = f"HTTP {response.status_code}"
            if response.status_code not in {429, 500, 502, 503, 504}:
                break
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        time.sleep(0.8 * (attempt + 1))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def request_html(url: str, refresh: bool) -> dict[str, Any]:
    path = cache_path("html", url, None)
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "ok": False,
        "request_url": url,
        "status_code": None,
        "html": None,
        "content_type": None,
        "error": None,
    }
    for attempt in range(3):
        try:
            response = requests.get(
                url,
                headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
                timeout=25,
                allow_redirects=True,
            )
            result["request_url"] = response.url
            result["status_code"] = response.status_code
            result["content_type"] = response.headers.get("Content-Type", "")
            if response.status_code == 200:
                result["ok"] = True
                result["html"] = response.text[:2_000_000]
                result["error"] = None
                break
            result["error"] = f"HTTP {response.status_code}"
            if response.status_code not in {429, 500, 502, 503, 504}:
                break
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        time.sleep(0.6 * (attempt + 1))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result["fetched_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def date_year(message: dict[str, Any]) -> str:
    for field in ("published-print", "published-online", "issued", "created"):
        parts = message.get(field, {}).get("date-parts", [])
        if parts and parts[0]:
            return str(parts[0][0])
    return ""


def doi_article_number(doi: str) -> str:
    """Recover Frontiers article numbers encoded in canonical DOI suffixes."""
    clean = doi.strip().casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if clean.startswith(prefix):
            clean = clean.removeprefix(prefix).strip()
            break
    match = re.fullmatch(
        r"10\.3389/[a-z0-9-]+\.(?:19|20)\d{2}\.(\d+)",
        clean,
    )
    return str(int(match.group(1))) if match else ""


def crossref_metadata(message: dict[str, Any]) -> dict[str, Any]:
    title = (message.get("title") or [""])[0]
    subtitle = (message.get("subtitle") or [""])[0]
    if subtitle and normalized(subtitle) not in normalized(title):
        title = title.rstrip(" :.") + ": " + subtitle
    authors = []
    for author in message.get("author", []):
        name = " ".join(
            part for part in (author.get("given", ""), author.get("family", "")) if part
        )
        if name:
            authors.append(name)
    doi = str(message.get("DOI") or "")
    article_number = str(message.get("article-number") or "") or doi_article_number(
        doi
    )
    return {
        "title": title,
        "authors": authors,
        "year": date_year(message),
        "container": (message.get("container-title") or [""])[0],
        "volume": str(message.get("volume") or ""),
        "issue": str(message.get("issue") or ""),
        "pages": str(message.get("page") or message.get("article-number") or ""),
        "article_number": article_number,
        "publisher": str(message.get("publisher") or ""),
        "doi": doi,
        "url": str(message.get("URL") or ""),
    }


def openalex_metadata(work: dict[str, Any]) -> dict[str, Any]:
    biblio = work.get("biblio") or {}
    first_page = str(biblio.get("first_page") or "")
    last_page = str(biblio.get("last_page") or "")
    pages = first_page
    if first_page and last_page and first_page != last_page:
        pages = f"{first_page}-{last_page}"
    location = work.get("primary_location") or {}
    source = location.get("source") or {}
    return {
        "title": str(work.get("title") or ""),
        "authors": [
            str((authorship.get("author") or {}).get("display_name") or "")
            for authorship in work.get("authorships", [])
            if (authorship.get("author") or {}).get("display_name")
        ],
        "year": str(work.get("publication_year") or ""),
        "container": str(source.get("display_name") or ""),
        "volume": str(biblio.get("volume") or ""),
        "issue": str(biblio.get("issue") or ""),
        "pages": pages,
        "article_number": "",
        "publisher": "",
        "doi": str(work.get("doi") or "").removeprefix("https://doi.org/"),
        "url": str(work.get("id") or ""),
    }


def semantic_scholar_metadata(paper: dict[str, Any]) -> dict[str, Any]:
    journal = paper.get("journal") or {}
    external = paper.get("externalIds") or {}
    return {
        "title": str(paper.get("title") or ""),
        "authors": [
            str(author.get("name") or "")
            for author in paper.get("authors", [])
            if author.get("name")
        ],
        "year": str(paper.get("year") or ""),
        "container": str(journal.get("name") or paper.get("venue") or ""),
        "volume": str(journal.get("volume") or ""),
        "issue": str(journal.get("issue") or ""),
        "pages": str(journal.get("pages") or ""),
        "article_number": "",
        "publisher": "",
        "doi": str(external.get("DOI") or ""),
        "url": str(paper.get("url") or ""),
    }


class CitationMetaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.metadata: dict[str, list[str]] = {}
        self.page_title = ""
        self.in_title = False
        self.visible_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.casefold(): value for key, value in attrs if value is not None}
        if tag.casefold() == "meta":
            key = (values.get("name") or values.get("property") or "").casefold()
            content = values.get("content", "").strip()
            if key and content:
                self.metadata.setdefault(key, []).append(content)
        elif tag.casefold() == "title":
            self.in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        clean = data.strip()
        if clean:
            self.visible_text.append(clean)
        if self.in_title:
            self.page_title += data


def first_meta(metadata: dict[str, list[str]], *keys: str) -> str:
    for key in keys:
        values = metadata.get(key.casefold(), [])
        if values:
            return html.unescape(values[0]).strip()
    return ""


def official_metadata(page: str) -> dict[str, Any]:
    parser = CitationMetaParser()
    parser.feed(page)
    metadata = parser.metadata
    first_page = first_meta(metadata, "citation_firstpage", "prism.startingpage")
    last_page = first_meta(metadata, "citation_lastpage", "prism.endingpage")
    pages = first_page
    if first_page and last_page and first_page != last_page:
        pages = f"{first_page}-{last_page}"
    published = first_meta(
        metadata,
        "citation_publication_date",
        "citation_date",
        "dc.date",
        "article:published_time",
    )
    year_match = re.search(r"(?:19|20)\d{2}", published)
    authors = metadata.get("citation_author", []) or metadata.get("dc.creator", [])
    return {
        "title": first_meta(
            metadata,
            "citation_title",
            "dc.title",
            "og:title",
        )
        or html.unescape(parser.page_title).strip(),
        "authors": [html.unescape(author).strip() for author in authors],
        "year": year_match.group(0) if year_match else "",
        "container": first_meta(
            metadata,
            "citation_journal_title",
            "citation_conference_title",
            "citation_inbook_title",
            "prism.publicationname",
        ),
        "volume": first_meta(metadata, "citation_volume", "prism.volume"),
        "issue": first_meta(metadata, "citation_issue", "prism.number"),
        "pages": pages,
        "article_number": first_meta(
            metadata,
            "citation_article_number",
            "citation_articlenumber",
        ),
        "publisher": first_meta(metadata, "citation_publisher", "dc.publisher"),
        "doi": first_meta(metadata, "citation_doi", "dc.identifier"),
        "url": first_meta(metadata, "citation_public_url", "og:url"),
    }


def source_record(
    name: str,
    lookup: str,
    request: dict[str, Any],
    metadata: dict[str, Any] | None,
    bib_title: str,
) -> dict[str, Any]:
    score = title_similarity(bib_title, (metadata or {}).get("title", ""))
    return {
        "source": name,
        "lookup": lookup,
        "status": "ok" if request.get("ok") and metadata else "unavailable",
        "status_code": request.get("status_code"),
        "evidence_url": request.get("request_url"),
        "fetched_at_utc": request.get("fetched_at_utc"),
        "error": request.get("error"),
        "title_similarity": round(score, 4),
        "metadata": metadata,
    }


def candidate_identity_score(
    entry: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[float, float, bool, float]:
    fields = entry["fields"]
    title_score = title_similarity(fields.get("title", ""), metadata.get("title", ""))
    bib_year = strip_latex(fields.get("year", ""))
    remote_year = str(metadata.get("year", ""))
    year_match = bool(bib_year and remote_year and bib_year == remote_year)
    bib_container = fields.get("journal") or fields.get("booktitle") or ""
    remote_container = metadata.get("container", "")
    container_score = (
        title_similarity(bib_container, remote_container)
        if bib_container and remote_container
        else 0.0
    )
    score = title_score + (0.18 if year_match else 0.0) + 0.08 * container_score
    return score, title_score, year_match, container_score


def accept_title_candidate(
    entry: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    metadata = record.get("metadata") or {}
    _, title_score, year_match, container_score = candidate_identity_score(entry, metadata)
    fields = entry["fields"]
    bib_year = strip_latex(fields.get("year", ""))
    remote_year = str(metadata.get("year", ""))
    bib_container = fields.get("journal") or fields.get("booktitle") or ""
    remote_container = metadata.get("container", "")
    if title_score < 0.72:
        record["status"] = "no_match"
    elif bib_year and remote_year and not year_match:
        record["status"] = "ambiguous_match"
        record["error"] = (
            f"Same/similar title but publication year {remote_year} "
            f"does not match intended year {bib_year}"
        )
    elif bib_container and remote_container and container_score < 0.55:
        record["status"] = "ambiguous_match"
        record["error"] = (
            f"Venue {remote_container!r} does not match intended venue "
            f"{strip_latex(bib_container)!r}"
        )
    return record


def query_crossref(entry: dict[str, Any], refresh: bool) -> dict[str, Any]:
    fields = entry["fields"]
    title = strip_latex(fields.get("title", ""))
    doi = strip_latex(fields.get("doi", "")).strip()
    if doi:
        url = "https://api.crossref.org/works/" + quote(doi, safe="")
        request = request_json(url, None, refresh)
        message = ((request.get("data") or {}).get("message") if request.get("ok") else None)
        if isinstance(message, dict):
            metadata = crossref_metadata(message)
            return source_record("Crossref", "doi", request, metadata, title)
        fallback = request_json(
            "https://api.crossref.org/works",
            {"query.bibliographic": title, "rows": 5},
            refresh,
        )
        items = ((fallback.get("data") or {}).get("message") or {}).get("items", [])
        best = max(
            items,
            key=lambda item: candidate_identity_score(entry, crossref_metadata(item))[0],
            default=None,
        )
        record = source_record(
            "Crossref",
            "doi-fallback-title",
            fallback,
            crossref_metadata(best) if best else None,
            title,
        )
        return accept_title_candidate(entry, record)

    request = request_json(
        "https://api.crossref.org/works",
        {"query.bibliographic": title, "rows": 5},
        refresh,
    )
    items = ((request.get("data") or {}).get("message") or {}).get("items", [])
    best = max(
        items,
        key=lambda item: candidate_identity_score(entry, crossref_metadata(item))[0],
        default=None,
    )
    metadata = crossref_metadata(best) if best else None
    record = source_record("Crossref", "title", request, metadata, title)
    return accept_title_candidate(entry, record)


def query_openalex(entry: dict[str, Any], refresh: bool) -> dict[str, Any]:
    fields = entry["fields"]
    title = strip_latex(fields.get("title", ""))
    doi = strip_latex(fields.get("doi", "")).strip()
    params: dict[str, Any]
    lookup: str
    if doi:
        params = {"filter": f"doi:https://doi.org/{doi}", "per-page": 5}
        lookup = "doi"
    else:
        params = {"search": title, "per-page": 5}
        lookup = "title"
    request = request_json("https://api.openalex.org/works", params, refresh)
    works = (request.get("data") or {}).get("results", [])
    best = max(
        works,
        key=lambda work: candidate_identity_score(entry, openalex_metadata(work))[0],
        default=None,
    )
    metadata = openalex_metadata(best) if best else None
    record = source_record("OpenAlex", lookup, request, metadata, title)
    if not doi:
        return accept_title_candidate(entry, record)
    if record["title_similarity"] < 0.72:
        record["status"] = "no_match"
    return record


def query_semantic_scholar(
    entry: dict[str, Any],
    refresh: bool,
    batch_papers: dict[str, dict[str, Any] | None] | None = None,
    batch_evidence_url: str = "",
) -> dict[str, Any]:
    fields = entry["fields"]
    title = strip_latex(fields.get("title", ""))
    doi = strip_latex(fields.get("doi", "")).strip()
    selected_fields = "title,authors,year,venue,journal,externalIds,url"
    if doi:
        if batch_papers is not None and entry["key"] in batch_papers:
            paper = batch_papers[entry["key"]]
            request = {
                "ok": paper is not None,
                "request_url": batch_evidence_url,
                "status_code": 200,
                "error": None if paper is not None else "DOI absent from batch response",
            }
            metadata = (
                semantic_scholar_metadata(paper)
                if isinstance(paper, dict)
                else None
            )
            record = source_record(
                "Semantic Scholar",
                "doi-batch",
                request,
                metadata,
                title,
            )
            if record["title_similarity"] < 0.72:
                record["status"] = "no_match"
            return record
        identifier = quote(f"DOI:{doi}", safe="")
        request = request_json(
            f"https://api.semanticscholar.org/graph/v1/paper/{identifier}",
            {"fields": selected_fields},
            refresh,
            retry_cached_failure=True,
        )
        paper = request.get("data") if request.get("ok") else None
        metadata = semantic_scholar_metadata(paper) if isinstance(paper, dict) else None
        record = source_record(
            "Semantic Scholar",
            "doi",
            request,
            metadata,
            title,
        )
        if record["title_similarity"] < 0.72:
            record["status"] = "no_match"
        return record

    request = request_json(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        {"query": title, "limit": 5, "fields": selected_fields},
        refresh,
    )
    papers = (request.get("data") or {}).get("data", [])
    best = max(
        papers,
        key=lambda paper: candidate_identity_score(
            entry,
            semantic_scholar_metadata(paper),
        )[0],
        default=None,
    )
    record = source_record(
        "Semantic Scholar",
        "title",
        request,
        semantic_scholar_metadata(best) if best else None,
        title,
    )
    return accept_title_candidate(entry, record)


def query_semantic_scholar_batch(
    entries: list[dict[str, Any]],
    refresh: bool,
) -> tuple[dict[str, dict[str, Any] | None], str]:
    doi_entries = [
        entry
        for entry in entries
        if strip_latex(entry["fields"].get("doi", "")).strip()
    ]
    identifiers = [
        "DOI:" + strip_latex(entry["fields"]["doi"]).strip()
        for entry in doi_entries
    ]
    fields = "title,authors,year,venue,journal,externalIds,url"
    request = request_json_post(
        "https://api.semanticscholar.org/graph/v1/paper/batch",
        {"fields": fields},
        {"ids": identifiers},
        refresh,
    )
    if not request.get("ok"):
        return {}, request.get("request_url", "")
    papers = request.get("data") or []
    if len(papers) != len(doi_entries):
        raise ValueError(
            "Semantic Scholar batch response length does not match DOI request"
        )
    return (
        {
            entry["key"]: paper if isinstance(paper, dict) else None
            for entry, paper in zip(doi_entries, papers)
        },
        request.get("request_url", ""),
    )


def query_official(entry: dict[str, Any], refresh: bool) -> dict[str, Any] | None:
    fields = entry["fields"]
    url = strip_latex(fields.get("url", "")).strip()
    lookup = "url"
    if not url:
        doi = strip_latex(fields.get("doi", "")).strip()
        if doi:
            url = f"https://doi.org/{doi}"
            lookup = "doi-landing"
    if not url:
        return None
    request = request_html(url, refresh)
    metadata = official_metadata(request["html"]) if request.get("ok") else None
    expected_title = strip_latex(fields.get("title", ""))
    if metadata and request.get("html"):
        parser = CitationMetaParser()
        parser.feed(request["html"])
        visible = " ".join(parser.visible_text)
        if normalized(expected_title) in normalized(visible):
            metadata["title"] = expected_title
    record = source_record(
        "Official page",
        lookup,
        request,
        metadata,
        strip_latex(fields.get("title", "")),
    )
    if record["title_similarity"] < 0.55:
        record["status"] = "no_match"
    return record


def compare_entry(entry: dict[str, Any], sources: list[dict[str, Any]]) -> dict[str, Any]:
    fields = entry["fields"]
    title = strip_latex(fields.get("title", ""))
    bib_authors = split_bib_authors(fields.get("author", ""))
    usable = [
        source
        for source in sources
        if source
        and source.get("status") == "ok"
        and source.get("metadata")
        and source.get("title_similarity", 0)
        >= (0.55 if source.get("source") == "Official page" else 0.72)
    ]
    issues: list[dict[str, str]] = []

    def add(severity: str, field: str, message: str) -> None:
        issues.append({"severity": severity, "field": field, "message": message})

    if not usable:
        return {
            **entry,
            "sources": sources,
            "status": "unverifiable",
            "issues": [
                {
                    "severity": "critical",
                    "field": "record",
                    "message": "No matching metadata record was recovered.",
                }
            ],
            "best_title_similarity": 0.0,
            "confirmed_sources": 0,
        }

    best_score = max(source["title_similarity"] for source in usable)
    if best_score < 0.88:
        add("critical", "title", f"Best title similarity is only {best_score:.3f}.")
    elif best_score < 0.96:
        add("warning", "title", f"Best title similarity is {best_score:.3f}.")

    remote_first = [
        source["metadata"]["authors"][0]
        for source in usable
        if source["metadata"].get("authors")
    ]
    if bib_authors and remote_first and not any(
        same_family(bib_authors[0], author) for author in remote_first
    ):
        add(
            "critical",
            "author",
            f"First author {bib_authors[0]!r} does not match {remote_first!r}.",
        )

    remote_counts = {
        len(source["metadata"]["authors"])
        for source in usable
        if source["metadata"].get("authors")
    }
    # Checking only the first author misses swaps of middle authors.
    same_length_lists = [source["metadata"]["authors"] for source in usable
                         if len(source["metadata"].get("authors", [])) == len(bib_authors)
                         and bib_authors]
    if same_length_lists and not any(
        all(same_family(local, remote) for local, remote in zip(bib_authors, names))
        for names in same_length_lists
    ):
        add("critical", "author_order", "No equal-length source agrees with the complete author sequence.")
    if bib_authors and remote_counts and len(bib_authors) not in remote_counts:
        add(
            "warning",
            "author",
            f"BibTeX has {len(bib_authors)} authors; sources report {sorted(remote_counts)}.",
        )

    remote_years = {
        source["metadata"]["year"]
        for source in usable
        if source["metadata"].get("year")
    }
    bib_year = strip_latex(fields.get("year", ""))
    if remote_years and bib_year not in remote_years:
        close_year = False
        if bib_year.isdigit():
            close_year = any(
                year.isdigit() and abs(int(year) - int(bib_year)) <= 1
                for year in remote_years
            )
        add(
            "warning" if close_year else "critical",
            "year",
            (
                f"BibTeX year {bib_year!r}; sources report "
                f"{sorted(remote_years)!r}"
                + (" (online/preprint/issue-year boundary)." if close_year else ".")
            ),
        )
    elif len(remote_years) > 1:
        add(
            "warning",
            "year",
            f"Sources disagree on online/issue year: {sorted(remote_years)!r}.",
        )

    if fields.get("doi"):
        priority = {
            "Crossref": 0,
            "Semantic Scholar": 1,
            "OpenAlex": 2,
            "Official page": 3,
        }
    else:
        priority = {
            "Official page": 0,
            "Semantic Scholar": 1,
            "OpenAlex": 2,
            "Crossref": 3,
        }
    ordered = sorted(usable, key=lambda source: priority.get(source["source"], 9))
    for field, severity in (
        ("volume", "critical"),
        ("issue", "warning"),
    ):
        remote = next(
            (
                source["metadata"].get(field, "")
                for source in ordered
                if source["metadata"].get(field)
            ),
            "",
        )
        local = strip_latex(fields.get("number" if field == "issue" else field, ""))
        if remote and local and normalized(remote) != normalized(local):
            add(
                severity,
                field,
                f"BibTeX {local!r}; preferred source reports {remote!r}.",
            )

    local_pages = strip_latex(fields.get("pages", ""))
    local_article = strip_latex(fields.get("articleno", ""))
    remote_pagination, remote_pagination_field = next(
        (
            (
                (source["metadata"].get("pages", ""), "pages")
                if source["metadata"].get("pages")
                else (
                    source["metadata"].get("article_number", ""),
                    "article_number",
                )
            )
            for source in ordered
            if source["metadata"].get("pages")
            or source["metadata"].get("article_number")
        ),
        ("", ""),
    )
    remote_article = next(
        (
            source["metadata"].get("article_number", "")
            for source in ordered
            if source["metadata"].get("article_number")
        ),
        "",
    )
    if local_pages:
        if (
            remote_pagination
            and normalized(local_pages) != normalized(remote_pagination)
        ):
            add(
                "critical",
                "pages",
                (
                    f"BibTeX {local_pages!r}; preferred source reports "
                    f"{remote_pagination!r}."
                ),
            )
    elif local_article:
        if (
            remote_article
            and normalized(local_article) != normalized(remote_article)
        ):
            add(
                "critical",
                "article_number",
                (
                    f"BibTeX {local_article!r}; preferred source reports "
                    f"{remote_article!r}."
                ),
            )
    elif remote_pagination_field == "article_number":
        add(
            "critical",
            "article_number",
            (
                "BibTeX is missing article number; source reports "
                f"{remote_pagination!r}."
            ),
        )
    elif (
        remote_pagination
        and not fields.get("numpages")
        and entry.get("type") != "inproceedings"
    ):
        add(
            "critical",
            "pages",
            f"BibTeX is missing pages; source reports {remote_pagination!r}.",
        )

    local_doi = strip_latex(fields.get("doi", "")).casefold()
    remote_dois = {
        source["metadata"].get("doi", "").removeprefix("https://doi.org/").casefold()
        for source in usable
        if source["metadata"].get("doi")
    }
    if local_doi and remote_dois and local_doi not in remote_dois:
        add(
            "critical",
            "doi",
            f"BibTeX DOI {local_doi!r}; sources report {sorted(remote_dois)!r}.",
        )

    confirmed = sum(
        source["title_similarity"] >= 0.90
        for source in usable
    )
    if (
        confirmed == 0
        and any(
            source["source"] == "Official page"
            and source["status"] == "ok"
            and source["title_similarity"] >= 0.55
            for source in usable
        )
    ):
        confirmed = 1
    if confirmed < 2:
        add(
            "warning",
            "evidence",
            f"Only {confirmed} source record matched at title similarity >= 0.90.",
        )

    severities = {issue["severity"] for issue in issues}
    status = "verified"
    if "critical" in severities:
        status = "needs_fix"
    elif issues:
        status = "warning"
    return {
        **entry,
        "sources": sources,
        "status": status,
        "issues": issues,
        "best_title_similarity": round(best_score, 4),
        "confirmed_sources": confirmed,
        "normalized_title": normalized(title),
    }


def verify_one(
    entry: dict[str, Any],
    refresh: bool,
    semantic_batch: dict[str, dict[str, Any] | None],
    semantic_batch_url: str,
    selected_sources: tuple[str, ...] = ("crossref", "semantic-scholar", "openalex", "official"),
) -> dict[str, Any]:
    sources = []
    if "crossref" in selected_sources:
        sources.append(query_crossref(entry, refresh))
    if "semantic-scholar" in selected_sources:
        sources.append(query_semantic_scholar(entry, refresh, semantic_batch, semantic_batch_url))
    if "openalex" in selected_sources:
        sources.append(query_openalex(entry, refresh))
    if "official" in selected_sources:
        official = query_official(entry, refresh)
        if official:
            sources.append(official)
    return compare_entry(entry, sources)


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Automated reference metadata verification report",
        "",
        f"- Audit date: {report['audit_date']}",
        f"- BibTeX file: paper/references.bib",
        f"- Entries parsed: {report['entry_count']}",
        (
            "- Status totals: "
            + ", ".join(f"{key}={value}" for key, value in summary.items())
        ),
        "- Sources requested: " + ", ".join(report.get("requested_sources", [])),
        "- Fresh requests (ignore prior caches): " + str(report.get("fresh_requests", False)),
        "- Scope: automated comparison of available author, title, year, volume, issue, pagination and DOI metadata; not a full-text or citation-claim audit.",
        "- Source counts are matching records, not proof of independent provenance or verification of every field. Human resolutions are retained in the project archive (reference_corrections_2026-09-11.md).",
        "",
        "Status meanings: verified means two or more matching records and no discrepancy in the implemented comparisons, not all-field certification; warning means identity is supported but metadata or coverage needs human attention; needs_fix means a detected substantive mismatch; unverifiable means no adequate matching record. Matching records may share provenance.",
        "",
        "| # | BibTeX key | Status | Confirmed sources | Best title match | Findings |",
        "|---:|---|---|---:|---:|---|",
    ]
    for index, item in enumerate(report["entries"], 1):
        findings = "; ".join(issue["message"] for issue in item["issues"]) or "None"
        findings = findings.replace("|", "/").replace("\n", " ")
        lines.append(
            f"| {index} | {item['key']} | {item['status']} | "
            f"{item['confirmed_sources']} | {item['best_title_similarity']:.3f} | "
            f"{findings} |"
        )

    lines.extend(["", "## Per-entry evidence", ""])
    for item in report["entries"]:
        fields = item["fields"]
        lines.extend(
            [
                f"### {item['key']}",
                "",
                f"- BibTeX identity: {strip_latex(fields.get('author', ''))}; "
                f"{strip_latex(fields.get('title', ''))}; "
                f"{strip_latex(fields.get('year', ''))}.",
                f"- Status: {item['status']}.",
            ]
        )
        if item["issues"]:
            lines.append(
                "- Findings: "
                + " ".join(
                    f"[{issue['severity']}/{issue['field']}] {issue['message']}"
                    for issue in item["issues"]
                )
            )
        else:
            lines.append("- Findings: no compared-field discrepancy.")
        for source in item["sources"]:
            metadata = source.get("metadata") or {}
            evidence_url = source.get("evidence_url") or ""
            lines.append(
                f"- {source['source']} ({source['lookup']}): "
                f"status={source['status']}, title_similarity={source['title_similarity']:.3f}, "
                f"title={metadata.get('title', '')!r}, year={metadata.get('year', '')!r}, "
                f"DOI={metadata.get('doi', '')!r}, evidence={evidence_url}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Ignore cached API results.")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--sources", nargs="+", choices=("crossref", "official", "openalex", "semantic-scholar"), default=["crossref", "official", "openalex", "semantic-scholar"])
    args = parser.parse_args()

    entries = parse_bibtex(BIB_PATH.read_text(encoding="utf-8"))
    if len(entries) != EXPECTED_ENTRIES:
        raise SystemExit(
            f"Expected {EXPECTED_ENTRIES} references but parsed {len(entries)}"
        )

    semantic_batch, semantic_batch_url = ({}, "")
    if "semantic-scholar" in args.sources:
        semantic_batch, semantic_batch_url = query_semantic_scholar_batch(entries, args.refresh)
    verified: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                verify_one,
                entry,
                args.refresh,
                semantic_batch,
                semantic_batch_url,
                tuple(args.sources),
            ): entry["key"]
            for entry in entries
        }
        for future in as_completed(futures):
            key = futures[future]
            verified[key] = future.result()
            print(f"verified {len(verified):02d}/{len(entries)} {key}")

    ordered = [verified[entry["key"]] for entry in entries]
    statuses = ("verified", "warning", "needs_fix", "unverifiable")
    summary = {
        status: sum(item["status"] == status for item in ordered)
        for status in statuses
    }
    report = {
        "schema_version": 1,
        "audit_date": REPORT_DATE,
        "requested_sources": args.sources,
        "fresh_requests": args.refresh,
        "entry_count": len(ordered),
        "bib_sha256": hashlib.sha256(BIB_PATH.read_bytes()).hexdigest(),
        "summary": summary,
        "entries": ordered,
    }
    JSON_OUTPUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MARKDOWN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    MARKDOWN_OUTPUT.write_text(markdown_report(report), encoding="utf-8")
    print(
        f"Wrote {JSON_OUTPUT.relative_to(ROOT)} and "
        f"{MARKDOWN_OUTPUT.relative_to(ROOT)}: {summary}"
    )
    return 1 if summary["needs_fix"] or summary["unverifiable"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
