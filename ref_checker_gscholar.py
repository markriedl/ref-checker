#!/usr/bin/env python3
"""Check PDF references for hallucinated titles and authors via Google Scholar + Semantic Scholar."""

import re
import os
import argparse
import requests
import time
import pymupdf4llm
import unidecode
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search/match"
GOOGLE_SCHOLAR_URL = "https://scholar.google.com/scholar"
GSCHOLAR_TIMEOUT = 15  # seconds per request
GSCHOLAR_429_WAIT = 60  # seconds to wait on a 429 before retrying
GSCHOLAR_MAX_RETRIES = 3

_UA = UserAgent()
_SESSION = requests.Session()


# ---------------------------------------------------------------------------
# PDF → reference blocks
# ---------------------------------------------------------------------------

def get_ref_section(md_text):
    """Extract the references section from markdown-formatted PDF text."""
    m = re.search(r"# [0-9 ]*\*\*References\*\*([a-zA-Z0-9 \(\)\.\,\;]*)", md_text)
    if m is not None:
        after = md_text[m.span()[1]:]
        m2 = re.search(r"# \*\*", after)
        if m2 is not None:
            return after[: m2.span()[0]].strip()
        return after
    return None


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize(text):
    """Lowercase, strip accents, collapse punctuation/whitespace for fuzzy matching."""
    text = unidecode.unidecode(text).lower()
    text = re.sub(r"[-]", " ", text)       # hyphens → spaces
    text = re.sub(r"[^\w\s]", " ", text)   # remaining punctuation → spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Google Scholar
# ---------------------------------------------------------------------------

def search_google_scholar(ref_text):
    """Search Google Scholar with the full reference text.

    Uses a persistent session with randomized user-agents and retries on 429.
    Returns the title string of the first hit, or None if not found / blocked.
    """
    params = {"q": ref_text, "hl": "en"}

    for attempt in range(1, GSCHOLAR_MAX_RETRIES + 1):
        _SESSION.headers.update({
            "User-Agent": _UA.random,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://scholar.google.com/",
        })
        try:
            response = _SESSION.get(
                GOOGLE_SCHOLAR_URL, params=params, timeout=GSCHOLAR_TIMEOUT
            )
        except requests.RequestException as e:
            print(f"  WARNING: Google Scholar request failed ({e})")
            return None

        if response.status_code == 429:
            if attempt < GSCHOLAR_MAX_RETRIES:
                print(f"  WARNING: Google Scholar rate limited (429) — waiting {GSCHOLAR_429_WAIT}s before retry {attempt}/{GSCHOLAR_MAX_RETRIES - 1}...")
                time.sleep(GSCHOLAR_429_WAIT)
                continue
            else:
                print(f"  WARNING: Google Scholar rate limited (429) — giving up after {GSCHOLAR_MAX_RETRIES} attempts. Try a longer --sleep or run later.")
                return None

        if response.status_code != 200:
            print(f"  WARNING: Google Scholar returned HTTP {response.status_code}")
            return None

        lc = response.text.lower()
        if "unusual traffic" in lc or 'id="captcha"' in lc or "recaptcha" in lc:
            print("  WARNING: Google Scholar CAPTCHA — try a longer --sleep or run later.")
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        result = soup.find(class_="gs_ri")
        if not result:
            return None

        title_el = result.find("h3", class_="gs_rt")
        if not title_el:
            return None

        a = title_el.find("a")
        if a:
            return a.get_text(strip=True)
        return re.sub(r"^\s*\[.*?\]\s*", "", title_el.get_text(strip=True))


def title_in_ref(gs_title, ref_text):
    """Return True if the normalized GS title is a substring of the normalized reference."""
    return normalize(gs_title) in normalize(ref_text)


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------

def get_paper_from_semantic_scholar(title):
    """Search Semantic Scholar for a paper by title. Returns the top match or None."""
    params = {"query": title, "fields": "title,authors"}
    response = requests.get(SEMANTIC_SCHOLAR_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        if data.get("data"):
            return data["data"][0]
    return None


# ---------------------------------------------------------------------------
# Author checking
# ---------------------------------------------------------------------------

def _last_name(name):
    """Return the accent-stripped, lowercased last token of a name string."""
    parts = name.strip().split()
    return unidecode.unidecode(parts[-1]).lower() if parts else ""


def check_authors(ref_text, ss_authors):
    """Check whether each Semantic Scholar author's last name appears in the reference text.

    Both the author last name and the reference text are accent-stripped before comparison.
    Returns a list of author names that are absent from the reference.
    """
    norm_ref = unidecode.unidecode(ref_text).lower()
    missing = []
    for author in ss_authors:
        ln = _last_name(author["name"])
        if ln and ln not in norm_ref:
            missing.append(author["name"])
    return missing


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_not_found(log_file, pdf_name, ref_num, ref_text):
    with open(log_file, "a") as f:
        f.write(f"PDF: {pdf_name}\n")
        f.write(f"Ref #{ref_num}\n")
        f.write(f"Raw: {ref_text[:300]}\n")
        f.write("-" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main check loop
# ---------------------------------------------------------------------------

def check_refs(filename, sleep=15, pdf_has_line_numbers=False, log_file="not_found.log"):
    print(f"Converting {filename} to markdown...")
    md_text = pymupdf4llm.to_markdown(filename)

    ref_section = get_ref_section(md_text)
    if ref_section is None:
        print("ERROR: Could not find references section in this PDF.")
        return

    refs = [r.strip() for r in ref_section.replace("_", "").split("\n\n") if r.strip()]
    print(f"Found {len(refs)} reference blocks.\n")

    pdf_name = os.path.basename(filename)

    for n, ref in enumerate(refs):
        if pdf_has_line_numbers:
            ref = re.sub(r"\b\d{1,3}\b", "", ref)

        print(f"[{n}] {ref.replace(chr(10), ' ')}")

        # Step 1: ask Google Scholar what the top hit is for this reference
        gs_title = search_google_scholar(ref)

        if gs_title is None:
            print("  -> NOT FOUND in Google Scholar")
            log_not_found(log_file, pdf_name, n, ref)
            print()
            time.sleep(sleep)
            continue

        # Step 2: verify the GS title actually appears in the original reference text
        if not title_in_ref(gs_title, ref):
            print(f"  -> TITLE NOT VERIFIED: Google Scholar returned \"{gs_title}\" but it is not in the reference")
            log_not_found(log_file, pdf_name, n, ref)
            print()
            time.sleep(sleep)
            continue

        print(f"  -> TITLE FOUND: {gs_title}")

        # Step 3: use the verified title to fetch authors from Semantic Scholar
        paper = get_paper_from_semantic_scholar(gs_title)
        if paper is None:
            print("  -> NOT FOUND in Semantic Scholar (cannot verify authors)")
            print()
            time.sleep(sleep)
            continue

        ss_authors = paper.get("authors", [])

        # Step 4: check each SS author's last name appears in the reference text
        missing = check_authors(ref, ss_authors)
        if not missing:
            print("  Authors: OK")
        else:
            for name in missing:
                print(f"  MISSING AUTHOR (in Semantic Scholar, not in ref): {name}")

        print()
        time.sleep(sleep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Check PDF references for hallucinated titles and authors."
    )
    parser.add_argument("filename", help="Path to the PDF file to check")
    parser.add_argument(
        "--sleep",
        type=float,
        default=15,
        help="Seconds to sleep between Google Scholar requests (default: 15)",
    )
    parser.add_argument(
        "--line-numbers",
        action="store_true",
        help="Strip line numbers embedded in reference text",
    )
    parser.add_argument(
        "--log",
        default="not_found.log",
        help="Log file for titles not found (default: not_found.log)",
    )
    args = parser.parse_args()

    check_refs(
        args.filename,
        sleep=args.sleep,
        pdf_has_line_numbers=args.line_numbers,
        log_file=args.log,
    )


if __name__ == "__main__":
    main()
