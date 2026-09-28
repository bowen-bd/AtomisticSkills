"""Audit publisher domains and WAF defenses to verify bot-blocking behavior.

This script scans target publisher domains or URLs, tests HTTP responses against
bot headers, detects WAF signatures (Cloudflare, Akamai, Radware, Imperva), and
outputs a diagnostic status report.

Usage:
    python .agents/skills/general-publisher-access-guard/scripts/audit_publishers.py --all
    python .agents/skills/general-publisher-access-guard/scripts/audit_publishers.py --publisher acs
    python .agents/skills/general-publisher-access-guard/scripts/audit_publishers.py --url "https://pubs.acs.org"
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests


def load_policy_matrix() -> dict:
    matrix_path = (
        Path(__file__).resolve().parent.parent
        / "resources"
        / "publisher_policy_matrix.json"
    )
    with open(matrix_path, "r", encoding="utf-8") as f:
        return json.load(f)


def probe_endpoint(url: str, timeout: int = 10) -> dict:
    bot_headers = {"User-Agent": "python-requests/2.31.0"}
    browser_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    result = {
        "url": url,
        "bot_status": None,
        "browser_status": None,
        "waf_detected": [],
        "block_signals": [],
        "is_blocked": False,
    }

    # 1. Bot Probe
    try:
        r_bot = requests.get(
            url, headers=bot_headers, timeout=timeout, allow_redirects=True
        )
        result["bot_status"] = r_bot.status_code
        server = r_bot.headers.get("Server", "").lower()

        if "cf-ray" in r_bot.headers or "cloudflare" in server:
            result["waf_detected"].append("Cloudflare")
        if "x-amz-cf-id" in r_bot.headers:
            result["waf_detected"].append("AWS CloudFront")
        if "akamai" in server or "x-akamai" in str(r_bot.headers).lower():
            result["waf_detected"].append("Akamai")
        if "validate.perfdrive.com" in r_bot.url:
            result["waf_detected"].append("Radware Bot Manager")
            result["block_signals"].append("CAPTCHA Redirect")

        text = r_bot.text[:3000].lower()
        if "turnstile" in text or "just a moment" in text:
            result["block_signals"].append("Cloudflare Turnstile Challenge")
        if "access denied" in text or r_bot.status_code == 403:
            result["block_signals"].append("403 Forbidden / Access Denied")
        if r_bot.status_code == 429:
            result["block_signals"].append("429 Too Many Requests")

    except Exception as e:
        result["bot_status"] = f"Error: {e}"

    # 2. Browser Probe
    try:
        r_browser = requests.get(
            url, headers=browser_headers, timeout=timeout, allow_redirects=True
        )
        result["browser_status"] = r_browser.status_code
    except Exception as e:
        result["browser_status"] = f"Error: {e}"

    result["is_blocked"] = (
        result["bot_status"] in (403, 429)
        or bool(result["block_signals"])
        or (
            result["bot_status"] == 200
            and "turnstile" in str(result["block_signals"]).lower()
        )
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Audit academic publisher anti-bot defenses."
    )
    parser.add_argument(
        "--all", action="store_true", help="Audit all publishers in the policy matrix"
    )
    parser.add_argument(
        "--publisher",
        type=str,
        help="Specific publisher ID to audit (e.g. acs, aps, elsevier)",
    )
    parser.add_argument("--url", type=str, help="Custom URL to probe")
    parser.add_argument(
        "--json", action="store_true", help="Output results in JSON format"
    )

    args = parser.parse_args()
    matrix = load_policy_matrix()

    targets = []
    if args.url:
        targets.append({"name": "Custom URL", "id": "custom", "url": args.url})
    elif args.publisher:
        pub = next(
            (
                p
                for p in matrix.get("publishers", [])
                if p["id"] == args.publisher.lower()
            ),
            None,
        )
        if not pub:
            print(f"Publisher ID '{args.publisher}' not found in matrix.")
            sys.exit(1)
        targets.append(
            {
                "name": pub["name"],
                "id": pub["id"],
                "url": f"https://{pub['domains'][0]}",
            }
        )
    elif args.all:
        for pub in matrix.get("publishers", []):
            targets.append(
                {
                    "name": pub["name"],
                    "id": pub["id"],
                    "url": f"https://{pub['domains'][0]}",
                }
            )
    else:
        parser.print_help()
        sys.exit(0)

    print(f"Starting audit of {len(targets)} targets...\n")
    audit_results = []

    for t in targets:
        print(f"Probing {t['name']} ({t['url']})...")
        probe = probe_endpoint(t["url"])
        probe["name"] = t["name"]
        probe["id"] = t["id"]
        audit_results.append(probe)
        time.sleep(0.5)

    if args.json:
        print(json.dumps(audit_results, indent=2))
    else:
        print("\n" + "=" * 85)
        print(
            f"{'Publisher':<35} | {'Bot HTTP':<10} | {'Browser':<10} | {'WAF / Blocks'}"
        )
        print("-" * 85)
        for r in audit_results:
            waf_info = ", ".join(r["waf_detected"] + r["block_signals"]) or "None"
            print(
                f"{r['name'][:35]:<35} | {str(r['bot_status']):<10} | {str(r['browser_status']):<10} | {waf_info[:25]}"
            )
        print("=" * 85)


if __name__ == "__main__":
    main()
