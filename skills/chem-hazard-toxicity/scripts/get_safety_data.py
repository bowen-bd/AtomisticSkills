import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PERCENT_CODE = re.compile(
    r"\b(H\d{3}[A-Za-z]?)\s*\(\s*([<>]=?)?\s*(\d+(?:\.\d+)?)\s*%\s*\)"
)
MASS_DOSE = re.compile(
    r"(?<![\w.])([0-9][0-9,.]*)\s*(ug|µg|μg|mg|g)\s*/\s*kg\b",
    re.IGNORECASE,
)
ORAL_CODES = {"H300", "H301", "H302", "H303"}


def find_section(node, target_heading):
    """Recursively search for a section with a specific TOCHeading."""
    if not isinstance(node, dict):
        return None
    if node.get("TOCHeading") == target_heading:
        return node
    for sec in node.get("Section", []):
        r = find_section(sec, target_heading)
        if r:
            return r
    return None


def extract_information(section, max_items=0):
    """Extract all text items from a section's Information fields."""
    results = []

    def _extract(node):
        if not isinstance(node, dict):
            return
        info_list = node.get("Information", [])
        for info in info_list:
            if "Value" in info and "StringWithMarkup" in info["Value"]:
                for item in info["Value"]["StringWithMarkup"]:
                    val = item.get("String", "").strip()
                    if val and val not in results:
                        results.append(val)
        for sec in node.get("Section", []):
            _extract(sec)

    _extract(section)
    if max_items > 0:
        return results[:max_items]
    return results


def query_safety_data(cid):
    """Fetch safety and toxicity data for a CID from PubChem PUG VIEW with exponential backoff."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "AtomisticSkills/1.0 (SafetyData)")

    max_retries = 5
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as e:
            if e.code in [429, 500, 502, 503, 504]:
                wait_time = 2**attempt
                print(f"Server busy (HTTP {e.code}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            elif e.code == 404:
                print(f"HTTP 404: Record not found for CID {cid}.")
                return None
            else:
                print(f"HTTP Error: {e.code} - {e.reason}")
                return None
        except urllib.error.URLError as e:
            if attempt == max_retries - 1:
                print(f"URL Error: {e.reason}")
                return None
            time.sleep(2**attempt)

    print("Maximum retries exceeded.")
    return None


def extract_consensus_ghs_codes(record, threshold_percent=50.0):
    """Extract GHS hazard codes supported by >= threshold_percent of reporting sources."""
    if not record or "Record" not in record:
        return []
    safety_sec = find_section(record["Record"], "GHS Classification")
    if not safety_sec:
        return []

    supported = set()
    for text in extract_information(safety_sec):
        for code, comparator, percentage in PERCENT_CODE.findall(text):
            if (
                not comparator.startswith("<")
                and float(percentage) >= threshold_percent
            ):
                supported.add(code)
    return sorted(supported)


def extract_lowest_rat_oral_ld50(record):
    """Select lowest explicit rat oral LD50 record in mg/kg and its supporting evidence string."""
    if not record or "Record" not in record:
        return None, ""
    tox_sec = find_section(record["Record"], "Non-Human Toxicity Values")
    if not tox_sec:
        tox_sec = find_section(record["Record"], "Toxicity")
    if not tox_sec:
        return None, ""

    factors = {"ug": 1e-3, "µg": 1e-3, "μg": 1e-3, "mg": 1.0, "g": 1e3}
    candidates = []

    for text in extract_information(tox_sec):
        lowered = text.lower()
        if "ld50" not in lowered or "rat" not in lowered or "oral" not in lowered:
            continue
        for number, unit in MASS_DOSE.findall(text):
            val_mg_kg = float(number.replace(",", "")) * factors[unit.lower()]
            candidates.append((val_mg_kg, text))

    if not candidates:
        return None, ""
    return min(candidates, key=lambda x: (x[0], x[1]))


def assign_acute_oral_category(val_mg_kg):
    """Map LD50 (mg/kg) to standard GHS Acute Oral Toxicity Category (1-5 or unclassified)."""
    if val_mg_kg is None:
        return "unclassified"
    for upper, cat in [(5, "1"), (50, "2"), (300, "3"), (2000, "4"), (5000, "5")]:
        if val_mg_kg <= upper:
            return cat
    return "unclassified"


def check_oral_code_consistency(ghs_codes, category):
    """Check if consensus GHS oral hazard codes match standard GHS Category expected codes.
    Note: Both Category 1 and Category 2 map to H300 ('Fatal if swallowed').
    Category 3 -> H301, Category 4 -> H302, Category 5 -> H303.
    """
    reported_oral = set(ghs_codes).intersection(ORAL_CODES)
    compatible = {
        "1": {"H300"},
        "2": {"H300"},
        "3": {"H301"},
        "4": {"H302"},
        "5": {"H303"},
        "unclassified": set(),
    }
    return reported_oral == compatible.get(category, set())


def profile_compound(cid, threshold_percent=50.0):
    """Generate complete evidence-backed acute toxicity profile for a compound CID."""
    record = query_safety_data(cid)
    codes = extract_consensus_ghs_codes(record, threshold_percent)
    ld50, evidence = extract_lowest_rat_oral_ld50(record)
    category = assign_acute_oral_category(ld50)
    consistent = check_oral_code_consistency(codes, category)

    return {
        "cid": cid,
        "consensus_ghs_codes": codes,
        "oral_rat_ld50_mg_kg": ld50,
        "oral_rat_ld50_evidence": evidence,
        "acute_oral_category": category,
        "ghs_oral_code_consistent": consistent,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Query PubChem for safety, hazard, and toxicity data with GHS acute toxicity triage."
    )
    parser.add_argument(
        "--cid", type=int, required=True, help="PubChem CID of the target molecule"
    )
    parser.add_argument("--outdir", required=True, help="Directory to save the results")
    parser.add_argument("--output", default="safety_data.json", help="Output filename")
    parser.add_argument(
        "--triage",
        action="store_true",
        help="Perform GHS acute toxicity category triage",
    )

    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching safety and toxicity records for CID: {args.cid}...")

    if args.triage:
        result_dict = profile_compound(args.cid)
    else:
        data = query_safety_data(args.cid)
        if not data or "Record" not in data:
            print("Failed to retrieve or parse data.")
            return
        record = data["Record"]
        result_dict = {
            "cid": args.cid,
            "ghs_classification": extract_information(
                find_section(record, "GHS Classification") or {}
            ),
            "hazard_classes": extract_information(
                find_section(record, "Hazard Classes and Categories") or {}
            ),
            "toxicity": extract_information(find_section(record, "Toxicity") or {}),
        }

    output_path = out_dir / args.output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=4)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
