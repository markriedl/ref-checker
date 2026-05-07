#!/usr/bin/env python3
"""Check PDF references using Minimax via OpenRouter (batch: one call per PDF)."""

import re
import os
import argparse
import requests
import time
import json
import getpass
import unicodedata
from openai import OpenAI
import pymupdf4llm

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search/match"
CONFIG_FILE = os.path.expanduser("~/.ref_checker_config")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "minimax/minimax-m2.5:free"

SYSTEM_PROMPT = (
    "You are a precise academic reference parser. "
    "You will receive a references section from an academic paper. "
    "For each reference, extract the paper title and list of author names. "
    "Return ONLY a valid JSON array with one object per reference, in the same order they appear, "
    "where each object has exactly two keys:\n"
    '  "title": the paper title as a string, or null if not identifiable\n'
    '  "authors": a list of author name strings as they appear in the reference '
    "(full names when available), or [] if no authors can be identified\n"
    "Do not include any explanation, markdown formatting, or code blocks. Output raw JSON only."
)


def get_openrouter_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            for line in f:
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.strip().split("=", 1)[1]
    key = getpass.getpass("Enter your OpenRouter API key: ").strip()
    if not key:
        raise SystemExit("No API key provided.")
    with open(CONFIG_FILE, "a") as f:
        f.write(f"OPENROUTER_API_KEY={key}\n")
    os.chmod(CONFIG_FILE, 0o600)
    print(f"API key saved to {CONFIG_FILE}")
    return key


client = OpenAI(api_key=get_openrouter_key(), base_url=OPENROUTER_BASE_URL)


def extract_all_refs(refs, ref_section, max_retries=3):
    """Send the raw ref section in one call; return list of (title, authors) tuples."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": ref_section},
                ],
            )
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 15 * (attempt + 1)
                print(f"  Rate limited, retrying in {wait}s...")
                time.sleep(wait)
                continue
            print(f"  API error: {e}")
            return [(None, [])] * len(refs)

        if not response.choices:
            print("  Empty response from model")
            return [(None, [])] * len(refs)

        text = response.choices[0].message.content
        if not text:
            return [(None, [])] * len(refs)
        text = text.strip()
        break
    else:
        return [(None, [])] * len(refs)

    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        results = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                results = json.loads(match.group())
            except json.JSONDecodeError:
                print("  Could not parse JSON array from response")
                return [(None, [])] * len(refs)
        else:
            print("  No JSON array found in response")
            return [(None, [])] * len(refs)

    if not isinstance(results, list):
        print("  Response was not a JSON array")
        return [(None, [])] * len(refs)

    # Pad or truncate to match ref count in case model dropped/added entries
    while len(results) < len(refs):
        results.append({"title": None, "authors": []})
    results = results[: len(refs)]

    return [(r.get("title"), r.get("authors") or []) for r in results]


def get_paper_from_semantic_scholar(title):
    params = {"query": title, "fields": "title,authors"}
    response = requests.get(SEMANTIC_SCHOLAR_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        if data.get("data"):
            return data["data"][0]
    return None


def normalize(s):
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii").lower()


def last_name(name):
    parts = re.sub(r"\s+", " ", name).strip().split()
    return normalize(parts[-1]) if parts else ""


def check_authors(ref_authors, ss_authors):
    ref_last_names = {last_name(a) for a in ref_authors if a.strip()}
    ss_last_names = {last_name(a["name"]) for a in ss_authors}
    missing = [a["name"] for a in ss_authors if last_name(a["name"]) not in ref_last_names]
    extra = [a for a in ref_authors if last_name(a) not in ss_last_names]
    return missing, extra


def log_not_found(log_file, pdf_name, ref_num, title, ref_text):
    with open(log_file, "a") as f:
        f.write(f"PDF: {pdf_name}\n")
        f.write(f"Ref #{ref_num}\n")
        f.write(f"Title: {title}\n")
        f.write(f"Raw: {ref_text[:300]}\n")
        f.write("-" * 60 + "\n")


def get_ref_section(md_text):
    m = re.search(r"# [0-9 ]*\*\*References\*\*([a-zA-Z0-9 \(\)\.\,\;]*)", md_text)
    if m is not None:
        after = md_text[m.span()[1]:]
        m2 = re.search(r"# \*\*", after)
        if m2 is not None:
            return after[: m2.span()[0]].strip()
        return after
    return None


def check_refs(filename, sleep=1, pdf_has_line_numbers=False, log_file="not_found.log"):
    print(f"\n{'='*70}")
    print(f"PDF: {filename}")
    print(f"{'='*70}")

    md_text = pymupdf4llm.to_markdown(filename)
    ref_section = get_ref_section(md_text)
    if ref_section is None:
        print("ERROR: Could not find references section.")
        return {"title_found": 0, "ss_found": 0, "no_title": 0}

    refs = [r.strip() for r in ref_section.replace("_", "").split("\n\n") if r.strip()]
    if pdf_has_line_numbers:
        refs = [re.sub(r"\d+", "", r) for r in refs]

    print(f"Found {len(refs)} reference blocks. Sending batch to {MODEL}...\n")
    parsed = extract_all_refs(refs, ref_section)

    pdf_name = os.path.basename(filename)
    stats = {"title_found": 0, "ss_found": 0, "no_title": 0}

    for n, (ref, (title, ref_authors)) in enumerate(zip(refs, parsed)):
        preview = ref[:100].replace("\n", " ")
        print(f"[{n}] {preview}{'...' if len(ref) > 100 else ''}")

        if not title:
            stats["no_title"] += 1
            print("  -> NO TITLE\n")
            continue

        stats["title_found"] += 1
        author_str = ", ".join(ref_authors[:3])
        if len(ref_authors) > 3:
            author_str += f" +{len(ref_authors)-3} more"
        print(f"  title: {title}")
        if ref_authors:
            print(f"  authors: {author_str}")

        paper = get_paper_from_semantic_scholar(title)
        time.sleep(sleep)

        if paper is None:
            print("  -> NOT FOUND in Semantic Scholar")
            log_not_found(log_file, pdf_name, n, title, ref)
        else:
            stats["ss_found"] += 1
            ss_authors = paper.get("authors", [])
            missing, extra = check_authors(ref_authors, ss_authors)
            if not missing and not extra:
                print("  -> FOUND | Authors OK")
            else:
                print("  -> FOUND | Author issues:")
                for name in missing:
                    print(f"     MISSING (in SS, not in ref): {name}")
                for name in extra:
                    print(f"     EXTRA (in ref, not in SS): {name}")
        print()

    return stats


def print_summary(all_stats):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = {"title_found": 0, "ss_found": 0, "no_title": 0}
    for s in all_stats.values():
        for k in total:
            total[k] += s.get(k, 0)
    refs_total = total["title_found"] + total["no_title"]
    print(f"Model:        {MODEL}")
    print(f"References:   {refs_total}")
    print(f"Title found:  {total['title_found']}  ({100*total['title_found']//refs_total if refs_total else 0}%)")
    print(f"In SS:        {total['ss_found']}  ({100*total['ss_found']//refs_total if refs_total else 0}%)")
    print(f"No title:     {total['no_title']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Check PDF references via Minimax on OpenRouter (batch mode)."
    )
    parser.add_argument("filenames", nargs="+", help="PDF file(s) to check")
    parser.add_argument(
        "--sleep", type=float, default=1,
        help="Seconds between Semantic Scholar calls (default: 1)",
    )
    parser.add_argument(
        "--line-numbers", action="store_true",
        help="Strip embedded line numbers from reference text",
    )
    parser.add_argument(
        "--log", default="not_found.log",
        help="Log file for titles not found in Semantic Scholar",
    )
    args = parser.parse_args()

    all_stats = {}
    for filename in args.filenames:
        all_stats[filename] = check_refs(
            filename,
            sleep=args.sleep,
            pdf_has_line_numbers=args.line_numbers,
            log_file=args.log,
        )

    print_summary(all_stats)


if __name__ == "__main__":
    main()
