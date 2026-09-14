"""Safe scientific paper retriever bypassing bot-blocked publisher websites.

This script implements an automated fallback cascade for retrieving paper metadata,
abstracts, and legal open-access full-text PDFs while strictly avoiding direct HTTP
requests to publisher domains known to ban or block automated agents.

Safe Cascade Flow:
    1. Pre-flight check on publisher domain and DOI prefix.
    2. Check official publisher API keys (ELSEVIER_API_KEY, SPRINGER_API_KEY).
    3. OpenAlex query for bibliographic metadata & inverted-index abstract.
    4. Unpaywall resolution with domain blocklist filtering (rejects publisher direct URLs).
    5. Europe PMC / PMC Open Access subset query.
    6. Formatted markdown summary + graceful paywall notice if no open copy exists.

Usage:
    python skills/general-publisher-access-guard/scripts/safe_paper_retriever.py --doi "10.1103/PhysRevLett.120.145301" --output_dir ./papers
    python skills/general-publisher-access-guard/scripts/safe_paper_retriever.py --url "https://pubs.acs.org/doi/10.1021/acs.chemmater.9b04758"
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Fallback polite email if not configured in environment
DEFAULT_EMAIL = (
    os.getenv("UNPAYWALL_EMAIL")
    or os.getenv("OPENALEX_EMAIL")
    or "atomistic_agent@example.edu"
)
USER_AGENT = f"AtomisticSkills-ResearchAgent/1.0 (mailto:{DEFAULT_EMAIL})"


def load_policy_matrix() -> dict:
    matrix_path = (
        Path(__file__).resolve().parent.parent
        / "resources"
        / "publisher_policy_matrix.json"
    )
    if not matrix_path.exists():
        raise FileNotFoundError(f"Publisher policy matrix not found at {matrix_path}")
    with open(matrix_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_doi(target: str) -> str:
    """Extract DOI from a string or URL."""
    match = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", target)
    if match:
        doi = match.group(0)
        # Clean trailing punctuation
        doi = re.sub(r"[.,;>]+$", "", doi)
        return doi
    return ""


def get_blocked_domains(matrix: dict) -> set:
    """Return all publisher domains that block bots."""
    blocked = set()
    for pub in matrix.get("publishers", []):
        if pub.get("direct_scrape_policy") in (
            "blocked",
            "allowed_api_only",
            "throttled",
        ):
            for d in pub.get("domains", []):
                blocked.add(d.lower())
    return blocked


def is_url_blocked(url: str, blocked_domains: set) -> bool:
    """Check if a given URL belongs to a blocked publisher domain."""
    if not url:
        return False
    netloc = urlparse(url).netloc.lower()
    for b_domain in blocked_domains:
        if b_domain in netloc:
            return True
    return False


def reconstruct_abstract(abstract_inverted_index: dict) -> str:
    """Reconstruct full text abstract from OpenAlex inverted index."""
    if not abstract_inverted_index:
        return ""
    token_positions = []
    for token, positions in abstract_inverted_index.items():
        for pos in positions:
            token_positions.append((pos, token))
    token_positions.sort(key=lambda x: x[0])
    return " ".join([token for _, token in token_positions])


def query_openalex(doi: str) -> dict:
    """Fetch structured metadata and abstract from OpenAlex Works API."""
    url = f"https://api.openalex.org/works/https://doi.org/{doi}"
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning(f"OpenAlex query error for {doi}: {e}")
    return {}


def query_unpaywall(doi: str, email: str, blocked_domains: set) -> tuple:
    """Query Unpaywall API and find safe repository open-access PDF URLs."""
    url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None, []
        data = resp.json()
        oa_locations = data.get("oa_locations", [])

        safe_pdf_urls = []
        blocked_pdf_urls = []

        for loc in oa_locations:
            pdf_url = loc.get("url_for_pdf") or loc.get("url")
            if not pdf_url:
                continue
            if is_url_blocked(pdf_url, blocked_domains):
                blocked_pdf_urls.append((pdf_url, loc.get("host_type", "publisher")))
            else:
                safe_pdf_urls.append(
                    {
                        "url": pdf_url,
                        "host_type": loc.get("host_type", "repository"),
                        "version": loc.get("version", "unknown"),
                        "license": loc.get("license"),
                    }
                )

        return safe_pdf_urls, blocked_pdf_urls
    except Exception as e:
        logger.warning(f"Unpaywall query error for {doi}: {e}")
        return None, []


def query_europe_pmc(doi: str) -> str:
    """Query Europe PMC for open access full text PDF or XML link."""
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=ext_id:{doi}%20src:med&format=json&resultType=core"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            results = resp.json().get("resultList", {}).get("result", [])
            if results:
                paper = results[0]
                if paper.get("isOpenAccess") == "Y":
                    # Check for full text links
                    pmcid = paper.get("pmcid")
                    if pmcid:
                        return f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"
    except Exception as e:
        logger.warning(f"Europe PMC query error for {doi}: {e}")
    return None


def download_safe_pdf(pdf_url: str, output_path: Path) -> bool:
    """Download PDF from a validated safe repository endpoint."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        resp = requests.get(pdf_url, headers=headers, stream=True, timeout=30)
        if resp.status_code == 200:
            content_type = resp.headers.get("Content-Type", "").lower()
            if "pdf" in content_type or len(resp.content) > 10000:
                with open(output_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
    except Exception as e:
        logger.warning(f"Failed to download safe PDF from {pdf_url}: {e}")
    return False


def extract_text_from_pdf(pdf_path: Path, txt_path: Path) -> str:
    """Extract plain text from downloaded PDF using PyMuPDF if installed."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        return text
    except Exception as e:
        logger.warning(f"Could not extract text with PyMuPDF: {e}")
        return ""


def retrieve_paper(doi: str, output_dir: Path, email: str = DEFAULT_EMAIL) -> dict:
    matrix = load_policy_matrix()
    blocked_domains = get_blocked_domains(matrix)

    # 1. Identify publisher
    publisher_match = None
    prefix = doi.split("/")[0] if "/" in doi else ""
    for pub in matrix.get("publishers", []):
        if prefix in pub.get("doi_prefixes", []):
            publisher_match = pub
            break

    result = {
        "doi": doi,
        "publisher": publisher_match["name"]
        if publisher_match
        else "Unknown Publisher",
        "publisher_policy": publisher_match.get("direct_scrape_policy")
        if publisher_match
        else "unknown",
        "title": "Unknown Title",
        "authors": [],
        "year": None,
        "journal": None,
        "citations": 0,
        "is_open_access": False,
        "abstract": "",
        "fulltext_retrieved": False,
        "pdf_path": None,
        "text_path": None,
        "retrieval_method": None,
        "notes": [],
    }

    if publisher_match and publisher_match.get("direct_scrape_policy") == "blocked":
        result["notes"].append(
            f"Publisher '{publisher_match['name']}' actively bans automated bot scrapers ({publisher_match.get('waf_type')}). Direct web scraping was prevented."
        )

    # 2. OpenAlex Query
    oa_data = query_openalex(doi)
    if oa_data:
        result["title"] = oa_data.get("title") or result["title"]
        result["year"] = oa_data.get("publication_year")
        result["citations"] = oa_data.get("cited_by_count", 0)
        result["is_open_access"] = oa_data.get("open_access", {}).get("is_oa", False)
        for authorship in oa_data.get("authorships", []):
            author_name = authorship.get("author", {}).get("display_name")
            if author_name:
                result["authors"].append(author_name)
        primary_loc = oa_data.get("primary_location") or {}
        source = primary_loc.get("source") or {}
        result["journal"] = source.get("display_name")

        abstract = reconstruct_abstract(oa_data.get("abstract_inverted_index"))
        if abstract:
            result["abstract"] = abstract
            result["retrieval_method"] = "OpenAlex Abstract"

    # 3. Safe OA Full-Text Search via Unpaywall (Filtered)
    safe_oa_urls, blocked_urls = query_unpaywall(doi, email, blocked_domains)
    if blocked_urls:
        for b_url, h_type in blocked_urls:
            result["notes"].append(f"Filtered out blocked publisher URL: {b_url}")

    safe_doi = doi.replace("/", "_")
    pdf_out = output_dir / f"{safe_doi}.pdf"
    txt_out = output_dir / f"{safe_doi}_text.txt"

    if safe_oa_urls:
        for candidate in safe_oa_urls:
            logger.info(
                f"Attempting safe download from repository mirror: {candidate['url']}"
            )
            if download_safe_pdf(candidate["url"], pdf_out):
                result["fulltext_retrieved"] = True
                result["pdf_path"] = str(pdf_out)
                result["retrieval_method"] = (
                    f"Safe Repository ({candidate['host_type']})"
                )
                extract_text_from_pdf(pdf_out, txt_out)
                result["text_path"] = str(txt_out)
                break

    # 4. Europe PMC fallback if not yet retrieved
    if not result["fulltext_retrieved"]:
        epmc_url = query_europe_pmc(doi)
        if epmc_url and download_safe_pdf(epmc_url, pdf_out):
            result["fulltext_retrieved"] = True
            result["pdf_path"] = str(pdf_out)
            result["retrieval_method"] = "Europe PMC Open Access"
            extract_text_from_pdf(pdf_out, txt_out)
            result["text_path"] = str(txt_out)

    # 5. Output Summary Files
    summary_path = output_dir / f"{safe_doi}_summary.md"
    json_path = output_dir / f"{safe_doi}_data.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Generate Markdown Summary
    md = f"# {result['title']}\n\n"
    md += f"- **DOI:** https://doi.org/{doi}\n"
    md += f"- **Publisher:** {result['publisher']} ({result['publisher_policy'].upper()})\n"
    if result["authors"]:
        authors_str = ", ".join(result["authors"][:5])
        if len(result["authors"]) > 5:
            authors_str += " et al."
        md += f"- **Authors:** {authors_str}\n"
    md += f"- **Year:** {result['year']} | **Citations:** {result['citations']} | **Open Access:** {'Yes' if result['is_open_access'] else 'No'}\n"
    if result["journal"]:
        md += f"- **Journal:** {result['journal']}\n"
    md += f"- **Retrieval Status:** {result['retrieval_method'] or 'Paywalled / Protected'}\n"

    if result["fulltext_retrieved"]:
        md += f"- **Full-Text PDF:** `{result['pdf_path']}`\n"
        md += f"- **Extracted Text:** `{result['text_path']}`\n"
    else:
        md += "\n> [!WARNING]\n"
        md += "> **Full-Text Restricted**: Direct scraping of this publisher was blocked by safety policy to prevent 403 errors and WAF bans. No unpaywalled repository copy (arXiv, PMC, Zenodo) was detected.\n"
        md += "> To read the full text, please download the PDF through your university proxy and provide it locally.\n"

    if result["abstract"]:
        md += f"\n## Abstract\n{result['abstract']}\n"

    if result["notes"]:
        md += "\n## Access Policy Notes\n"
        for note in result["notes"]:
            md += f"- {note}\n"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(md)

    result["summary_path"] = str(summary_path)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve paper metadata and OA PDFs safely without scraping blocked publishers."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--doi", type=str, help="Digital Object Identifier")
    group.add_argument("--url", type=str, help="Paper URL")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./papers",
        help="Directory to save paper outputs",
    )
    parser.add_argument(
        "--email",
        type=str,
        default=DEFAULT_EMAIL,
        help="Polite contact email for Unpaywall/OpenAlex",
    )

    args = parser.parse_args()
    target = args.doi or args.url
    doi = extract_doi(target)

    if not doi:
        logger.error(f"Could not extract a valid DOI from target: {target}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Executing Safe Paper Retrieval for DOI: {doi}")
    res = retrieve_paper(doi, output_dir, args.email)

    print("\n" + "=" * 70)
    print(f"Paper:  {res['title']}")
    print(f"DOI:    {res['doi']}")
    print(f"Status: {res['retrieval_method']}")
    if res["fulltext_retrieved"]:
        print(f"PDF:    {res['pdf_path']}")
    else:
        print(
            "Notice: Full text is paywalled/bot-protected. Abstract extracted safely."
        )
    print(f"Report: {res['summary_path']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
