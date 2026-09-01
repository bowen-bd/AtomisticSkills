---
name: general-fair-data-review
description: Review a manuscript or code repository for FAIR data compliance (Findable, Accessible, Interoperable, Reusable), producing a structured report with pass/fail per principle and actionable remediation steps.
metadata:
  category: [general]
---

# General FAIR Data Review

## Goal

Assess whether a manuscript submission or standalone code/data repository satisfies the FAIR Guiding Principles (Wilkinson et al., *Sci. Data* 2016). The output is a structured reviewer report — analogous to a peer-review report — that scores each FAIR sub-principle, identifies gaps, and provides concrete remediation steps the authors can act on before publication.

This skill is complementary to [general-peer-review](../general-peer-review/SKILL.md), which focuses on scientific methodology. Run both in sequence for a complete review.

---

## Prerequisites

- A manuscript (PDF or markdown) **and/or** a link to a code/data repository (GitHub, Zenodo, Figshare, etc.)
- Read access to any supplementary files, Data Availability Statements (DAS), or README files provided by the authors

---

## Instructions

### 1. Identify Review Scope

Determine what artifacts are under review. Three modes exist:

| Mode | Input | Focus |
|------|-------|-------|
| **Manuscript + data/code** | PDF + repo URL | Full FAIR review |
| **Manuscript only** | PDF | DAS quality, metadata richness, identifier presence |
| **Code/data repo only** | Repo URL / directory | Repository-level FAIR compliance |

State the mode explicitly at the start of the review report.

---

### 2. Read the Manuscript and Data Availability Statement

Load the manuscript. Locate and extract:
- The **Data Availability Statement** (DAS) — usually a dedicated section near the end.
- Any **Code Availability Statement**.
- All **data/code repository URLs** or DOIs mentioned.

If no DAS exists, flag immediately as a **Critical Finding** (fails F4, A1, R1.1).

---

### 3. Inspect the Data/Code Repository

For each repository URL found, check the following. If no repository exists, mark all sub-principles below as **Fail**.

```
Repository inspection checklist:
- Does a persistent identifier (DOI, Handle) exist? → F1
- Is metadata present and rich (title, authors, description, keywords, license)? → F2, R1
- Does the metadata explicitly reference the dataset/code identifier? → F3
- Is the repository indexed in a searchable resource (Zenodo, Figshare, OSF, etc.)? → F4
- Can the data/code be accessed via a standard protocol (HTTP/HTTPS, FTP)? → A1
- Is the protocol open and free (no proprietary portal login required)? → A1.1
- If restricted, is there a documented access procedure? → A1.2
- Does metadata remain accessible even if data is removed? → A2
- Are standard, community-recognized formats used (CIF, JSON, CSV, HDF5, not .xlsx or proprietary)? → I1
- Are domain ontologies or controlled vocabularies used for metadata fields? → I2
- Are cross-references to related datasets or publications included? → I3
- Is a clear, machine-readable license present (CC-BY, MIT, Apache 2.0, etc.)? → R1.1
- Is provenance documented (how data was generated, software versions, parameters)? → R1.2
- Do files conform to domain community standards (e.g., CIF for crystal structures, SMILES for molecules, HDF5 for trajectories)? → R1.3
```

---

### 4. Score Each FAIR Sub-Principle

For every sub-principle (F1–F4, A1–A2, I1–I3, R1–R1.3) assign:

- **Pass** — requirement fully met
- **Partial** — requirement partially met; improvement needed
- **Fail** — requirement not met or artifact absent
- **N/A** — not applicable to this submission type

---

### 5. Atomistic/Computational Science Specific Checks

In addition to the generic FAIR checklist, evaluate the following domain-specific criteria:

**Structures & Trajectories**
- Crystal structures deposited as `.cif` (not as images or in supplementary PDF tables)
- MD trajectories deposited in open formats (`.xyz`, `.extxyz`, `.h5md`, `.lammpsdump`) with a `README` specifying units, timestep, ensemble
- Force field / MLIP checkpoints deposited with version, training set provenance, and validation metrics

**Computational Parameters**
- DFT: INCAR/POTCAR/KPOINTS or equivalent included or described with exact values (not "similar to ref. X")
- MLIP: model architecture, training hyperparameters, and train/val/test split recorded
- MD: timestep, thermostat/barostat settings, equilibration protocol documented

**Software Environment**
- `environment.yml` or `requirements.txt` present with pinned versions
- Scripts runnable from the deposited repository without undocumented external dependencies

---

### 6. Generate the Structured Review Report

Produce a report in the following format:

```markdown
# FAIR Data Review Report

**Manuscript title:** [title]
**Review date:** [date]
**Reviewer:** AI FAIR Data Reviewer (general-fair-data-review skill)
**Review mode:** [Manuscript + data/code | Manuscript only | Code/data repo only]

---

## Summary

[2–4 sentences: overall FAIRness level, most critical gaps, overall recommendation: Ready / Minor Revisions / Major Revisions / Not Acceptable]

---

## FAIR Scorecard

| Principle | Sub-principle | Status | Evidence / Gap |
|-----------|--------------|--------|----------------|
| **Findable** | F1: Persistent identifier | Pass/Partial/Fail | ... |
| | F2: Rich metadata | | |
| | F3: Metadata references data ID | | |
| | F4: Indexed in searchable resource | | |
| **Accessible** | A1: Retrievable via standard protocol | | |
| | A1.1: Protocol open and free | | |
| | A1.2: Auth procedure documented | | |
| | A2: Metadata accessible if data removed | | |
| **Interoperable** | I1: Formal/shared knowledge representation | | |
| | I2: FAIR vocabularies used | | |
| | I3: Qualified references to other data | | |
| **Reusable** | R1: Rich, accurate, relevant attributes | | |
| | R1.1: Clear data usage license | | |
| | R1.2: Detailed provenance | | |
| | R1.3: Domain community standards met | | |

---

## Major Concerns

[Number sequentially. For each: state issue → why problematic → actionable fix.]

1. **[Issue title]**
   - *Problem:* ...
   - *Impact:* ...
   - *Fix:* ...

---

## Minor Concerns

- ...

---

## Atomistic/Computational Specific Findings

[Report on structure formats, trajectory deposits, software environments, parameter completeness.]

---

## Questions for Authors

1. ...

---

## Recommended Repositories (if none provided)

If no repository was deposited, suggest domain-appropriate options:

| Data type | Recommended repository |
|-----------|----------------------|
| Crystal structures | CCDC, ICSD, Materials Cloud, Zenodo |
| Molecular dynamics trajectories | Materials Cloud, Zenodo, NOMAD |
| ML models / checkpoints | Hugging Face, Zenodo, MACE-Models |
| General datasets | Zenodo, Figshare, Dryad |
| Code | GitHub + Zenodo DOI via Zenodo GitHub integration |
```

---

## Document-Specific Workflows

### Manuscript Review (PDF)

> [!WARNING]
> Read the PDF text directly. Do **not** assume data exists unless a DOI or repository URL is explicitly present in the manuscript body or supplement.

Steps:
1. Extract DAS and Code Availability Statement verbatim.
2. Resolve any DOIs or URLs found.
3. If DOIs resolve to a live repository, proceed with repository inspection (Step 3).
4. If only "data available upon request" is stated: flag as **Fail** for F1, F4, A1, R1.1 — this does not meet FAIR standards.

### Code/Data Repository Review Only

Steps:
1. Read top-level `README.md`.
2. Check for `LICENSE`, `environment.yml`/`requirements.txt`, `CITATION.cff`.
3. Inspect directory structure for data files and their formats.
4. Check metadata on the repository platform (Zenodo record, GitHub About section, etc.).

---

## Constraints

- **Scope**: Focus strictly on data/code FAIRness. Scientific methodology critique belongs in [general-peer-review](../general-peer-review/SKILL.md).
- **Tone**: Objective and constructive. Every Fail must include a concrete, actionable fix.
- **"Data available upon request"**: Always flag as non-FAIR. Not acceptable per RSC Digital Discovery and FAIR principles.
- **Proprietary formats**: `.xlsx`, `.mat`, Gaussian `.chk`, VASP `WAVECAR` without open alternatives are I1/R1.3 failures.
- **License absence**: Unlicensed ≠ open. Always flag missing licenses as R1.1 Fail.

---

## References

- Wilkinson, M. D. et al., "The FAIR Guiding Principles for scientific data management and stewardship", *Sci. Data* **3**, 160018 (2016). [DOI: 10.1038/sdata.2016.18](https://doi.org/10.1038/sdata.2016.18)
- RSC Digital Discovery Data Review Guidelines. [rsc.org/publishing](https://www.rsc.org/publishing/publish-with-us/publish-a-journal-article/digital-discovery)

## See Also

- [general-peer-review](../general-peer-review/SKILL.md)
- [general-deep-research](../general-deep-research/SKILL.md)

---

**Author:** Magdalena Lederbauer
**Contact:** [GitHub @mlederbauer](https://github.com/mlederbauer)
