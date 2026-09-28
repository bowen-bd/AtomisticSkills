# Example 2: Resolving APS Paper via Legal Open-Access Repository Mirror

## Goal
To demonstrate how an agent retrieves the full text of an American Physical Society (APS) article (*Physical Review Letters*, DOI: `10.1103/PhysRevLett.120.145301`) without being blocked by APS's Cloudflare WAF on `link.aps.org`, by filtering out the blocked publisher URL and fetching the author's open-access green manuscript from an approved institutional repository.

## Step-by-Step Instructions

### 1. Pre-Flight Domain Check
```bash
# Env: base-agent
python .agents/skills/general-publisher-access-guard/scripts/check_publisher.py \
    --doi "10.1103/PhysRevLett.120.145301"
```

**Expected Output:**
```
======================================================================
Publisher Policy Assessment: [BLOCKED] American Physical Society (APS)
======================================================================
Target:                10.1103/PhysRevLett.120.145301
Direct Scraping Safe:  NO - DO NOT SCRAPE
WAF Protection:        Cloudflare Turnstile
Expected Status Code:  403
Reason:                Blocks bots with 403 Forbidden. Robots.txt explicitly disallows AI scrapers (Amazonbot, Applebot-Extended, Bytespider, CCBot). Direct PDF links on link.aps.org will fail.
----------------------------------------------------------------------
Recommended Route:     APS Physical Review REST API (token required), arXiv preprints, OpenAlex.
Safe Mirrors / APIs:   arxiv, openalex, unpaywall_filtered
======================================================================
```

### 2. Execute Safe Paper Retrieval
```bash
# Env: base-agent
python .agents/skills/general-publisher-access-guard/scripts/safe_paper_retriever.py \
    --doi "10.1103/PhysRevLett.120.145301" \
    --output_dir .agents/skills/general-publisher-access-guard/examples/oa-repository-bypass
```

**Execution Trace:**
```
INFO: Executing Safe Paper Retrieval for DOI: 10.1103/PhysRevLett.120.145301
INFO: Skipping blocked publisher URL to prevent 403 / WAF ban: http://link.aps.org/pdf/10.1103/PhysRevLett.120.145301
INFO: Attempting safe download from repository mirror: https://www.osti.gov/servlets/purl/1524040
INFO: Successfully extracted text from OA PDF for 10.1103/PhysRevLett.120.145301 to 10.1103_PhysRevLett.120.145301_text.txt
```

## Literature Validation
The target paper is:
- **Title:** "Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties"
- **Authors:** Tian Xie, Jeffrey C. Grossman
- **Journal:** *Physical Review Letters*, 2018, 120, 14, 145301. [DOI: 10.1103/PhysRevLett.120.145301](https://doi.org/10.1103/PhysRevLett.120.145301)

### Verification Against Published Values
1. **Model Dataset and Accuracy:** The extracted full text confirms that CGCNN was trained on $10^4$ DFT-calculated materials from the Materials Project database to predict 8 different properties (formation energy, band gap, Fermi energy, bulk moduli, shear moduli, Poisson ratio, etc.), with a formation energy test MAE of $0.039\text{ eV/atom}$. This precisely validates the scientific authenticity of the extracted text.
2. **Cloudflare Avoidance:** Unpaywall initially returned `http://link.aps.org/pdf/10.1103/PhysRevLett.120.145301` as the primary OA location. Direct requests to that link fail with HTTP 403 Forbidden. The safe retriever identified the blocked domain, skipped it, and retrieved the authorized open-access green manuscript from the US Department of Energy OSTI repository / MIT DSpace without error.
