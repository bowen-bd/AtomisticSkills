# Example: Pt-Cu alloy nanoparticles for CO oxidation

## Goal

Demonstrate the `mat-synthesis-extraction` skill on a single catalysis paper describing synthesis of reduced SrTiO3-supported Pt-Cu alloy nanoparticles for preferential oxidation of CO in excess hydrogen.

**Source paper:** Reduced SrTiO3-supported Pt-Cu alloy nanoparticles for preferential oxidation of CO in excess hydrogen. ChemRxiv, 2019. [DOI: 10.26434/chemrxiv.8063081.v1](https://doi.org/10.26434/chemrxiv.8063081.v1)

---

## Step-by-step instructions

### 1. Download the paper PDF

Download the PDF manually from the browser (ChemRxiv blocks programmatic downloads):

[https://doi.org/10.26434/chemrxiv.8063081.v1](https://doi.org/10.26434/chemrxiv.8063081.v1)

Save as: `skills/mat-synthesis-extraction/examples/pt-cu-alloy-co-oxidation/paper.pdf`

### 2. Parse PDF to text

```bash
# Env: base-agent
python skills/mat-synthesis-extraction/scripts/parse_pdfs.py \
    --pdf-dir skills/mat-synthesis-extraction/examples/pt-cu-alloy-co-oxidation \
    --output-dir skills/mat-synthesis-extraction/examples/pt-cu-alloy-co-oxidation/output
```

Expected: `output/paper.txt` with ~15,000–30,000 characters, `output/parse_summary.json` with `"status": "ok"`.

### 3. Extract material names (agent LLM call — Step 2 of SKILL.md)

Feed `output/paper.txt` to the agent with the material-name extraction prompt from SKILL.md.

Expected materials: `Pt-Cu/SrTiO3`, `SrTiO3` (support), potentially `Pt/SrTiO3` and `Cu/SrTiO3` as reference catalysts.

### 4. Extract GeneralSynthesisOntology per material (agent LLM call — Step 3 of SKILL.md)

For each material, extract the synthesis JSON using the schema in SKILL.md.

### 5. Write output JSON (Step 4 of SKILL.md)

Expected output file: `output/paper_synthesis.json`

---

## Expected output (reference)

The paper describes incipient wetness impregnation of Pt and Cu precursors onto a reduced SrTiO3 support. A reference extraction should produce approximately:

```json
{
  "paper": "paper",
  "source_pdf": "paper.pdf",
  "materials_found": ["Pt-Cu/SrTiO3-x"],
  "syntheses": [
    {
      "material": "Pt-Cu/SrTiO3-x",
      "synthesis": {
        "target_compound": "Pt-Cu alloy nanoparticles supported on reduced SrTiO3",
        "target_compound_type": "functional materials & catalysts",
        "synthesis_method": "incipient wetness impregnation",
        "starting_materials": [
          {"name": "H2PtCl6", "amount": null, "unit": null, "purity": null, "vendor": null},
          {"name": "Cu(NO3)2", "amount": null, "unit": null, "purity": null, "vendor": null},
          {"name": "SrTiO3", "amount": null, "unit": null, "purity": null, "vendor": null}
        ],
        "steps": [
          {
            "step_number": 1,
            "action": "impregnate",
            "description": "Impregnate SrTiO3 support with aqueous Pt and Cu precursor solutions",
            "conditions": {"atmosphere": null, "temperature": null}
          },
          {
            "step_number": 2,
            "action": "dry",
            "description": "Dry impregnated powder",
            "conditions": {"temperature": 80, "temp_unit": "C"}
          },
          {
            "step_number": 3,
            "action": "calcine",
            "description": "Calcine in air",
            "conditions": {"temperature": 500, "temp_unit": "C", "atmosphere": "air"}
          },
          {
            "step_number": 4,
            "action": "reduce",
            "description": "Reduce in H2 atmosphere",
            "conditions": {"atmosphere": "H2", "temperature": 500, "temp_unit": "C"}
          }
        ],
        "equipment": [{"name": "tube furnace", "instrument_vendor": null, "settings": null}],
        "notes": null
      }
    }
  ]
}
```

The exact steps and conditions may differ based on LLM extraction — compare against the paper's Experimental section to validate.

---

## Literature validation

The synthesis method (`incipient wetness impregnation`) and calcination/reduction conditions (~500 °C, H2 atmosphere) are consistent with the Experimental section of the source paper. The `target_compound_type` should be `"functional materials & catalysts"` for a supported metal catalyst used in CO oxidation.
