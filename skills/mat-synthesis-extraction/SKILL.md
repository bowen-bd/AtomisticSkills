---
name: mat-synthesis-extraction
description: Extract structured synthesis procedures from a folder of PDFs using the LeMat-Synth GeneralSynthesisOntology schema, producing one JSON file per paper with per-material synthesis records.
metadata:
  category: [materials]
---

# mat-synthesis-extraction

## Goal

Given a folder of scientific paper PDFs, extract all synthesis procedures described in each paper and structure them according to the **GeneralSynthesisOntology** developed in LeMat-Synth [1]. Output is one JSON file per paper containing a list of per-material synthesis records.

The ontology captures: target compound, compound type, synthesis method, starting materials (with amounts/units/purity), sequential process steps (with actions, conditions, equipment), and overall equipment list.

---

## Instructions

### Step 1 — Parse PDFs to text

Run the PDF parser to extract plain text from all PDFs in the input folder.

```bash
# Env: base-agent
python skills/mat-synthesis-extraction/scripts/parse_pdfs.py \
    --pdf-dir /path/to/pdf_folder \
    --output-dir /path/to/output/texts
```

This produces:
- One `.txt` file per PDF (named `<paper_stem>.txt`)
- A `parse_summary.json` listing extraction status and character counts

Inspect `parse_summary.json` to confirm all PDFs extracted successfully. Papers with `"status": "empty"` are likely scanned images — skip them or obtain a text-layer PDF.

---

### Step 2 — Extract synthesized material names

For each `.txt` file produced in Step 1, identify which materials are synthesized in the paper. Read the paper text and extract a comma-separated list of synthesized compound names.

**System prompt to use:**
```
You are a materials science expert. Given the full text of a scientific paper, identify ALL distinct materials that are synthesized (not just characterized or used as reagents). Return ONLY a comma-separated list of chemical names or formulas (e.g. "NiCo2O4, CoFe2O4, Fe3O4"). If no synthesis is described, return an empty string.
```

**Input:** full paper text from Step 1
**Output:** comma-separated string of material names → split into a Python list

---

### Step 3 — Extract GeneralSynthesisOntology per material

For each (paper_text, material_name) pair from Step 2, extract the structured synthesis ontology. Use the system prompt and JSON schema below.

**System prompt:**
```
You are a helpful assistant that extracts the structured synthesis for a specific material from the paper text.

Focus ONLY on the synthesis procedure for the specified material. Search through the entire paper text to find the synthesis procedure that describes how this specific material is made.

IMPORTANT: You must output ONLY a valid JSON object with a "structured_synthesis" field. Do not include any reasoning, explanations, or markdown formatting.

If you cannot find a synthesis procedure for the specified material, return a minimal structure with the material name and an empty synthesis.
```

**User message template:**
```
Paper text:
<PAPER_TEXT>

Extract the synthesis procedure for: <MATERIAL_NAME>
```

**Required JSON output schema (GeneralSynthesisOntology):**
```json
{
  "structured_synthesis": {
    "target_compound": "string (required) — composition and description of the target",
    "target_compound_type": "one of: 'metals & alloys' | 'ceramics & glasses' | 'polymers & soft matter' | 'composites' | 'semiconductors & electronic' | 'nanomaterials' | 'two-dimensional materials' | 'framework & porous materials' | 'biomaterials & biological' | 'liquid materials' | 'hybrid & organic-inorganic' | 'functional materials & catalysts' | 'energy & sustainability' | 'smart & responsive materials' | 'emerging & quantum materials' | 'other'",
    "synthesis_method": "one of: 'PVD' | 'CVD' | 'arc discharge' | 'ball milling' | 'spray pyrolysis' | 'electrospinning' | 'sol-gel' | 'hydrothermal' | 'solvothermal' | 'precipitation' | 'coprecipitation' | 'combustion' | 'microwave-assisted' | 'sonochemical' | 'template-directed' | 'solid-state' | 'flux growth' | 'float zone & Bridgman' | 'arc melting & induction melting' | 'spark plasma sintering' | 'electrochemical deposition' | 'chemical bath deposition' | 'liquid-phase epitaxy' | 'self-assembly' | 'atomic layer deposition' | 'molecular beam epitaxy' | 'pulsed laser deposition' | 'ion implantation' | 'lithographic patterning' | 'wet impregnation' | 'incipient wetness impregnation' | 'mechanical mixing' | 'solution-based' | 'mechanochemical' | 'other'",
    "starting_materials": [
      {
        "name": "string",
        "amount": "number or null",
        "unit": "string or null — e.g. 'g', 'mL', 'mmol', 'wt%'",
        "purity": "string or null — e.g. '99.9%', 'ACS grade'",
        "vendor": "string or null"
      }
    ],
    "steps": [
      {
        "step_number": "integer",
        "action": "one of: 'add' | 'mix' | 'heat' | 'cool' | 'reflux' | 'age' | 'filter' | 'wash' | 'dry' | 'reduce' | 'calcine' | 'dissolve' | 'precipitate' | 'centrifuge' | 'sonicate' | 'anneal' | 'ion exchange' | 'impregnate'",
        "description": "string or null",
        "materials": [{"name": "string", "amount": "number or null", "unit": "string or null", "purity": "string or null", "vendor": "string or null"}],
        "equipment": [{"name": "string", "instrument_vendor": "string or null", "settings": "string or null"}],
        "conditions": {
          "temperature": "number or null",
          "temp_unit": "string or null — 'C', 'K', or 'F'",
          "duration": "number or null",
          "time_unit": "string or null — 'h', 'min', 's', 'days'",
          "pressure": "number or null",
          "pressure_unit": "string or null",
          "atmosphere": "string or null — e.g. 'air', 'N2', 'Ar'",
          "stirring": "boolean or null",
          "stirring_speed": "number or null",
          "ph": "number or null"
        }
      }
    ],
    "equipment": [{"name": "string", "instrument_vendor": "string or null", "settings": "string or null"}],
    "notes": "string or null"
  }
}
```

**Retry strategy:** If extraction fails or JSON is invalid, retry with slightly increased temperature (0.3, then 0.5). If all retries fail, write a minimal record:
```json
{"target_compound": "<material_name>", "target_compound_type": "other", "synthesis_method": "other", "starting_materials": [], "steps": [], "equipment": [], "notes": "Extraction failed."}
```

---

### Step 4 — Collect and write output JSON

After extracting all materials for a paper, write one JSON output file per paper.

**Output file:** `<output_dir>/<paper_stem>_synthesis.json`

**Output format:**
```json
{
  "paper": "<paper_stem>",
  "source_pdf": "<original_pdf_filename>",
  "materials_found": ["Material A", "Material B"],
  "syntheses": [
    {
      "material": "Material A",
      "synthesis": { ... }
    },
    {
      "material": "Material B",
      "synthesis": { ... }
    }
  ]
}
```

Write all output JSONs to the same `--output-dir` used in Step 1 (or a dedicated subdirectory).

---

### Step 5 — Review outputs

Inspect the extracted JSONs. Key things to verify:

- `target_compound` matches the material identified in Step 2
- `synthesis_method` is not `"other"` unless genuinely ambiguous
- `steps` list is non-empty for papers that clearly describe synthesis
- `starting_materials` includes amounts/units where reported in the paper

Flag papers where `steps` is empty and `notes` contains `"Extraction failed"` for manual review.

---

## Examples

See [examples/pt-cu-alloy-co-oxidation/](examples/pt-cu-alloy-co-oxidation/) for a worked example using a catalysis paper from ChemRxiv.

---

## Constraints

- **Environment**: Step 1 (`parse_pdfs.py`) requires `base-agent` (pymupdf installed).
- **Scanned PDFs**: PyMuPDF extracts embedded text only. Scanned-image PDFs produce empty output — use a PDF with a text layer.
- **Figure text**: PyMuPDF may extract figure captions and table text. The LLM extraction prompt instructs to focus on synthesis procedure sections only.
- **One JSON per paper**: Output aggregates all materials for a paper into a single file.
- **No DSPy / no separate LLM env**: LLM extraction is done by the agent directly using the schema and prompts above. No additional conda environment needed beyond `base-agent` for the parsing script.
- **Input configs**: Save any run-specific settings (model used, temperatures, pdf_dir, output_dir) to `input_configs.yaml` in the output directory for reproducibility.

---

## References

[1] Lederbauer et al., "LeMat-Synth: a multi-modal toolbox to curate broad synthesis procedure databases from scientific literature", *arXiv*, 2025. [arXiv:2510.26824](https://arxiv.org/abs/2510.26824)

---

**Author:** Magdalena Lederbauer
**Contact:** [GitHub @mlederbauer](https://github.com/mlederbauer)
