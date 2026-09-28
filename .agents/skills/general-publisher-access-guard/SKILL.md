---
name: general-publisher-access-guard
description: Avoid bot-blocking publisher websites by routing paper retrieval through legal open-access APIs and mirrors.
metadata:
  category: [general]
---

# General Publisher Access Guard

## Goal
To prevent autonomous research agents from being blocked by scientific journal publisher web application firewalls (Cloudflare Turnstile, Radware CAPTCHA, Akamai, Imperva) and HTTP 403/429 error loops, by auditing publisher anti-bot policies and automatically routing paper retrieval through legal open-access APIs, repository mirrors, and reconstructed abstract services.

## Instructions

When the user asks you to fetch, read, summarize, or analyze scientific literature from specific URLs or DOIs, **NEVER** use direct web scraping (`read_url_content`, `curl`, or raw `requests.get`) on known bot-blocking publisher domains. Follow the three steps below:

### 1. Pre-Flight Publisher Assessment
Before attempting to download or scrape any paper URL or DOI, inspect the domain against the curated publisher policy matrix.

```bash
# Env: base-agent
python .agents/skills/general-publisher-access-guard/scripts/check_publisher.py --url "https://pubs.acs.org/doi/10.1021/acs.chemmater.9b04758"
```
Or with DOI:
```bash
# Env: base-agent
python .agents/skills/general-publisher-access-guard/scripts/check_publisher.py --doi "10.1103/PhysRevLett.120.145301" --json
```

**Decision Rule:**
- If `direct_scraping_safe: false` (status `BLOCKED` or `API ONLY`), you **MUST NOT** scrape or curl the publisher web page. Proceed immediately to Step 2.
- Known blocked domains include: `pubs.acs.org`, `journals.aps.org`, `link.aps.org`, `sciencedirect.com`, `link.springer.com`, `nature.com`, `onlinelibrary.wiley.com`, `pubs.rsc.org`, `iopscience.iop.org`, `pubs.aip.org`, `ieeexplore.ieee.org`, `science.org`, `pnas.org`, `tandfonline.com`, `academic.oup.com`, and `mdpi.com`.

### 2. Execute the Safe Retrieval Cascade
Run the safe paper retriever to query legitimate APIs in priority order:

1. **Official Publisher APIs**: If credentials (`ELSEVIER_API_KEY`, `SPRINGER_API_KEY`) are present in environment variables.
2. **OpenAlex Works API**: Retrieves complete bibliographic metadata (authors, journal, citations, publication year) and reconstructs the full abstract from the inverted index.
3. **Filtered Unpaywall API**: Searches for legal Green/Gold Open Access copies hosted in institutional and disciplinary repositories (e.g. arXiv, PubMed Central, Zenodo, university `.edu` repositories), **automatically discarding any link pointing back to a blocked publisher domain**.
4. **Europe PMC API**: Direct open access XML/PDF retrieval for biomedical and chemistry literature.
5. **Preprint APIs**: Queries arXiv API or ChemRxiv API with required rate throttling (minimum 3 seconds delay).

```bash
# Env: base-agent
python .agents/skills/general-publisher-access-guard/scripts/safe_paper_retriever.py \
    --doi "10.1103/PhysRevLett.120.145301" \
    --output_dir ./papers
```

The script outputs:
- `<safe_doi>_summary.md`: Structured markdown report with title, authors, citations, reconstructed abstract, and access notes.
- `<safe_doi>.pdf`: Downloaded open-access PDF (if available in a safe repository).
- `<safe_doi>_text.txt`: Plaintext extracted from the PDF via PyMuPDF.

### 3. Graceful Handling of Paywalled Literature
If no open-access copy exists across repository mirrors:
- **DO NOT** retry scraping the publisher website.
- Read and synthesize the abstract provided in `<safe_doi>_summary.md`.
- Inform the user that the full text is restricted by publisher paywalls/WAFs, and request them to upload the institutional PDF to the workspace if full-text extraction is required.

### 4. Auditing Publisher Defenses
To test live publisher WAF defenses or verify if a publisher's bot policy has updated:

```bash
# Env: base-agent
python .agents/skills/general-publisher-access-guard/scripts/audit_publishers.py --publisher acs
```
Or audit all major publishers:
```bash
# Env: base-agent
python .agents/skills/general-publisher-access-guard/scripts/audit_publishers.py --all
```

## Examples

### Example 1: Bypassing ACS Bot Block for Perovskite Battery Paper
Handling an ACS paper (`10.1021/acs.jpclett.7b00189`) where direct web scraping returns Cloudflare Turnstile 403:
```bash
# Env: base-agent
python .agents/skills/general-publisher-access-guard/scripts/safe_paper_retriever.py \
    --doi "10.1021/acs.jpclett.7b00189" \
    --output_dir examples/acs-blocked-paper
```
See [examples/acs-blocked-paper/README.md](examples/acs-blocked-paper/README.md) for full execution traces and validation against published literature.

### Example 2: Resolving APS Paper via Legal Open-Access Repository Mirror
Handling an APS Physical Review Letters paper (`10.1103/PhysRevLett.120.145301`) by bypassing blocked `link.aps.org/pdf` and fetching the author manuscript from OSTI/MIT DSpace:
```bash
# Env: base-agent
python .agents/skills/general-publisher-access-guard/scripts/safe_paper_retriever.py \
    --doi "10.1103/PhysRevLett.120.145301" \
    --output_dir examples/oa-repository-bypass
```
See [examples/oa-repository-bypass/README.md](examples/oa-repository-bypass/README.md) for step-by-step instructions.

## Constraints
- **Environment**: Requires `base-agent` conda environment (contains `requests`, `pymupdf` / `fitz`).
- **Zero Direct Scrape**: Never execute `curl`, `read_url_content`, or browser automation on blocked publisher domains.
- **Unpaywall Polite Pool**: Provide an email address in `UNPAYWALL_EMAIL` or `OPENALEX_EMAIL` environment variable to access high-throughput API pools.
- **Preprint Rate Limits**: arXiv requests must maintain a minimum 3-second crawl delay.
- **Institutional Access**: Paywalled papers without green open-access copies cannot be legally decrypted without subscriber credentials or manual PDF upload.

## References
- Priem, J., Piwowar, H., & Orr, R., "OpenAlex: A fully-open index of scholarly works, authors, venues, institutions, and concepts", *arXiv:2205.01833*, 2022. [DOI: 10.48550/arXiv.2205.01833](https://doi.org/10.48550/arXiv.2205.01833)
- Piwowar, H. et al., "The state of OA: a large-scale analysis of the prevalence and impact of Open Access articles", *PeerJ*, 6:e4375, 2018. [DOI: 10.7717/peerj.4375](https://doi.org/10.7717/peerj.4375)
- Ferguson, C. et al., "Europe PMC: a stocktake by the Europe PMC Consortium", *Nucleic Acids Research*, 49(D1):D1507-D1514, 2021. [DOI: 10.1093/nar/gkaa994](https://doi.org/10.1093/nar/gkaa994)
- Cloudflare AI Crawl Control & Bot Management Whitepaper, 2024.
