"""Check scientific journal publisher bot-blocking policies and suggest safe APIs.

This script checks a target URL, domain, or DOI against the publisher policy matrix
to determine whether automated scraping/fetching will be blocked (e.g., by Cloudflare,
Akamai, Radware, or 403/429 status codes) and outputs the recommended legal API route.

Usage:
    python skills/general-publisher-access-guard/scripts/check_publisher.py --url "https://pubs.acs.org/doi/10.1021/jacs.3c01234"
    python skills/general-publisher-access-guard/scripts/check_publisher.py --doi "10.1103/PhysRevLett.120.145301"
    python skills/general-publisher-access-guard/scripts/check_publisher.py --domain "sciencedirect.com"
"""

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse


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


def extract_doi_prefix(doi_or_url: str) -> str:
    doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", doi_or_url)
    if doi_match:
        full_doi = doi_match.group(0)
        return full_doi.split("/")[0]
    return ""


def identify_publisher(target: str, matrix: dict) -> dict:
    # 1. Check if domain in URL
    parsed = urlparse(target)
    netloc = parsed.netloc.lower() or target.lower()

    # Match domain
    for pub in matrix.get("publishers", []):
        for domain in pub.get("domains", []):
            if domain in netloc:
                return pub

    # 2. Check DOI prefix
    prefix = extract_doi_prefix(target)
    if prefix:
        for pub in matrix.get("publishers", []):
            if prefix in pub.get("doi_prefixes", []):
                return pub

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Check if a scientific publisher blocks bots and find safe API alternatives."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", type=str, help="Full paper URL or landing page")
    group.add_argument(
        "--doi", type=str, help="Digital Object Identifier (e.g. 10.1021/jacs.3c01234)"
    )
    group.add_argument(
        "--domain", type=str, help="Publisher domain (e.g. pubs.acs.org)"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output results in JSON format"
    )

    args = parser.parse_args()
    target = args.url or args.doi or args.domain

    matrix = load_policy_matrix()
    publisher = identify_publisher(target, matrix)

    if not publisher:
        result = {
            "target": target,
            "matched": False,
            "status": "unknown",
            "message": f"Target '{target}' does not match any known blocked publisher. Standard precautions still apply.",
            "direct_scraping_safe": True,
        }
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(
                f"[OK] Target '{target}' is not in the known bot-blocking publisher blocklist."
            )
        sys.exit(0)

    policy = publisher.get("direct_scrape_policy", "unknown")
    is_blocked = policy in ("blocked", "allowed_api_only")

    result = {
        "target": target,
        "matched": True,
        "publisher_name": publisher["name"],
        "publisher_id": publisher["id"],
        "policy": policy,
        "direct_scraping_safe": not is_blocked,
        "status_code_on_bot": publisher.get("status_code_on_bot"),
        "waf_type": publisher.get("waf_type"),
        "blocking_reason": publisher.get("reason"),
        "recommended_api": publisher.get("recommended_api"),
        "safe_open_mirrors": publisher.get("safe_open_mirrors", []),
        "env_api_key_variable": publisher.get("env_api_key"),
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        status_tag = (
            "BLOCKED"
            if policy == "blocked"
            else ("API ONLY" if policy == "allowed_api_only" else "THROTTLED")
        )
        print("=" * 70)
        print(f"Publisher Policy Assessment: [{status_tag}] {publisher['name']}")
        print("=" * 70)
        print(f"Target:                {target}")
        print(
            f"Direct Scraping Safe:  {'NO - DO NOT SCRAPE' if is_blocked else 'YES (with rate limits)'}"
        )
        print(f"WAF Protection:        {publisher.get('waf_type')}")
        print(f"Expected Status Code:  {publisher.get('status_code_on_bot')}")
        print(f"Reason:                {publisher.get('reason')}")
        print("-" * 70)
        print(f"Recommended Route:     {publisher.get('recommended_api')}")
        print(
            f"Safe Mirrors / APIs:   {', '.join(publisher.get('safe_open_mirrors', []))}"
        )
        if publisher.get("env_api_key"):
            print(f"Official API Env Key:  {publisher.get('env_api_key')}")
        print("=" * 70)

    sys.exit(1 if is_blocked else 0)


if __name__ == "__main__":
    main()
