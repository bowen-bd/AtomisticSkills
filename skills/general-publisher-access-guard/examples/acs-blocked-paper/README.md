# Example 1: Bypassing ACS Bot Block for Perovskite Battery Paper

## Goal
To demonstrate how an agent safely handles a paper published in an American Chemical Society (ACS) journal (*The Journal of Physical Chemistry Letters*, DOI: `10.1021/acs.jpclett.7b00189`), where direct HTTP requests to `pubs.acs.org` trigger Cloudflare Turnstile bot challenges resulting in HTTP 403 Forbidden errors.

## Step-by-Step Instructions

### 1. Pre-Flight Domain Check
Run the publisher policy validator:
```bash
# Env: base-agent
python skills/general-publisher-access-guard/scripts/check_publisher.py \
    --url "https://pubs.acs.org/doi/10.1021/acs.jpclett.7b00189"
```

**Expected Output:**
```
======================================================================
Publisher Policy Assessment: [BLOCKED] American Chemical Society (ACS)
======================================================================
Target:                https://pubs.acs.org/doi/10.1021/acs.jpclett.7b00189
Direct Scraping Safe:  NO - DO NOT SCRAPE
WAF Protection:        Cloudflare Turnstile / Managed Challenge
Expected Status Code:  403
Reason:                Deploys Cloudflare Turnstile and AI Crawl Control to block programmatic scrapers, curl, and automated agents. Strict prohibition of unauthorized text and data mining.
----------------------------------------------------------------------
Recommended Route:     Crossref TDM API (with institutional token), OpenAlex metadata & abstract, Unpaywall institutional repository copies.
Safe Mirrors / APIs:   openalex, unpaywall_filtered, europe_pmc
======================================================================
```

### 2. Execute Safe Paper Retrieval
Run the safe paper retriever:
```bash
# Env: base-agent
python skills/general-publisher-access-guard/scripts/safe_paper_retriever.py \
    --doi "10.1021/acs.jpclett.7b00189" \
    --output_dir skills/general-publisher-access-guard/examples/acs-blocked-paper
```

## Literature Validation
The target publication is:
- **Title:** "Methylammonium Lead Bromide Perovskite Battery Anodes Reversibly Host High Li-Ion Concentrations"
- **Authors:** Nuria Vicente, Germà García-Belmonte
- **Journal:** *The Journal of Physical Chemistry Letters*, 2017, 8, 4, 896–900. [DOI: 10.1021/acs.jpclett.7b00189](https://doi.org/10.1021/acs.jpclett.7b00189)

### Verification Against Published Values
1. **Capacity and Lithium Insertion:** The extracted abstract reports a reversible specific capacity of $\sim 200\text{ mAh g}^{-1}$ and high Li-ion molar concentration approaching $x \approx 3$ in $\text{Li}_x\text{CH}_3\text{NH}_3\text{PbBr}_3$. This matches the experimental values reported by Vicente & García-Belmonte in the peer-reviewed paper.
2. **Zero WAF Penalty:** The agent successfully extracted the complete scientific findings without issuing any request to `pubs.acs.org`, completely avoiding Cloudflare 403 blocks.
3. **Graceful Paywall Handling:** Because this specific ACS article is behind a subscription paywall without an unembargoed repository preprint, the retriever generated `10.1021_acs.jpclett.7b00189_summary.md` and explicitly noted that full-text PDF access requires local institutional credentials.
