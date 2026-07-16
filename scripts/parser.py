"""
Parses articles from Wikipedia and saves them to data/articles.json.

Article text/title/url come from the wikimedia/wikipedia dataset on
HuggingFace Hub (streamed, no per-article API calls), because the plain
MediaWiki API approach hits Wikipedia's anonymous rate limit hard once you
need content for thousands of articles - empirically, ~120 requests/min was
enough to trigger sustained 429s that made bulk collection impractically
slow. Streaming the dataset avoids that entirely for the bulk of the work.

Categories are still fetched from the MediaWiki API (prop=categories, no
extracts - a much lighter call), but only for the few thousand articles
actually kept, so total API usage stays small enough to stay under the rate
limit even at modest concurrency.

Usage: python scripts/parser.py
"""

import concurrent.futures
import hashlib
import json
import os
import threading
import time

import requests
from datasets import load_dataset

API_URL = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "SearcherMVP/1.0 (educational project; python-requests)"
}

WIKI_DATASET = "wikimedia/wikipedia"
WIKI_DATASET_CONFIG = "20231101.en"
SHUFFLE_BUFFER = 10000  # streaming shuffle window, so we don't just take
                        # articles in dump order

TARGET_DOCS = 6000    # buffer above the 5000 minimum required
MIN_FINAL_DOCS = 5000

CATEGORY_BATCH = 20     # max titles per prop=categories request
MAX_WORKERS = 3         # anonymous MediaWiki API access has a fairly tight
                        # rate limit; going much above this causes persistent
                        # 429s (verified empirically)
REQUEST_PACE_SECONDS = 1.0  # per-worker delay between requests
MIN_CONTENT_LENGTH = 200
MAX_CONTENT_LENGTH = 4000
MAX_CATEGORIES = 5    # categories stored per article, for search filtering

OUTPUT_PATH = "data/articles.json"


def _get(params: dict, max_retries: int = 5) -> dict | None:
    params = {**params, "format": "json"}
    for attempt in range(max_retries):
        try:
            resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=20)
            if resp.status_code == 429:
                wait = 15 * (2 ** attempt)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            print(f"  Error (attempt {attempt + 1}): {e}")
            time.sleep(5)
    print(f"  Giving up on this request after {max_retries} attempts.")
    return None


def collect_articles(target: int) -> list[dict]:
    """Stream articles from the HF Wikipedia dataset until `target` distinct,
    long-enough articles are collected."""
    print(f"Streaming articles from {WIKI_DATASET} ({WIKI_DATASET_CONFIG})...")
    ds = load_dataset(WIKI_DATASET, WIKI_DATASET_CONFIG, split="train", streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=SHUFFLE_BUFFER)

    articles = []
    seen_hashes = set()
    duplicates_skipped = 0
    scanned = 0

    for row in ds:
        scanned += 1
        content = row["text"].strip()
        if len(content) >= MIN_CONTENT_LENGTH:
            content = content[:MAX_CONTENT_LENGTH]
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                duplicates_skipped += 1
            else:
                seen_hashes.add(content_hash)
                articles.append({
                    "title": row["title"],
                    "url": row["url"],
                    "content": content,
                    "source": "Wikipedia",
                    "categories": [],
                })

        if scanned % 1000 == 0:
            print(f"[{scanned} scanned] collected {len(articles)} articles "
                  f"({duplicates_skipped} duplicates skipped)")

        if len(articles) >= target:
            break

    print(f"\nCollected {len(articles)} articles from {scanned} scanned rows "
          f"({duplicates_skipped} duplicates skipped).")
    return articles


def _fetch_categories_batch(titles: list[str]) -> dict[str, list[str]] | None:
    """Fetch categories for up to CATEGORY_BATCH titles at once.

    Returns None (not {}) when the request itself failed after retries, so
    callers can retry instead of silently leaving articles without categories.
    """
    params = {
        "action": "query",
        "titles": "|".join(titles),
        "prop": "categories",
        "redirects": 1,
        "cllimit": "max",
        "clshow": "!hidden",  # skip maintenance/tracking categories (e.g. "CS1 errors")
    }
    data = _get(params)
    if data is None:
        return None

    pages = data.get("query", {}).get("pages", {})
    result = {}
    for page in pages.values():
        if "missing" in page:
            continue
        title = page.get("title", "")
        result[title] = [
            c["title"].split(":", 1)[-1] for c in page.get("categories", [])
        ][:MAX_CATEGORIES]
    return result


def _fetch_categories_paced(titles: list[str]) -> dict[str, list[str]] | None:
    result = _fetch_categories_batch(titles)
    time.sleep(REQUEST_PACE_SECONDS)
    return result


def fill_categories(articles: list[dict]) -> None:
    """Look up categories for each article's title and fill them in place."""
    by_title = {a["title"]: a for a in articles}
    titles = list(by_title.keys())
    print(f"\nFetching categories for {len(titles)} articles in batches of "
          f"{CATEGORY_BATCH} using {MAX_WORKERS} parallel workers...\n")

    pending = [titles[i:i + CATEGORY_BATCH] for i in range(0, len(titles), CATEGORY_BATCH)]
    lock = threading.Lock()
    completed_titles = 0
    retry_round = 0

    while pending:
        retry_round += 1
        if retry_round > 1:
            print(f"\nRetrying {len(pending)} category batches that failed to "
                  f"fetch (retry round {retry_round})...\n")
        failed = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_fetch_categories_paced, batch): batch for batch in pending}
            for future in concurrent.futures.as_completed(futures):
                batch = futures[future]
                result = future.result()

                if result is None:
                    failed.append(batch)
                    continue

                with lock:
                    for title, cats in result.items():
                        if title in by_title:
                            by_title[title]["categories"] = cats
                    completed_titles += len(batch)
                    print(f"[{completed_titles}/{len(titles)} titles processed for categories]")

        pending = failed
        if pending and retry_round >= 5:
            print(f"\nWARNING: {len(pending)} category batches "
                  f"({sum(len(b) for b in pending)} titles) kept failing after "
                  f"{retry_round} retry rounds; those articles will have no categories.")
            break


def save(articles: list[dict]) -> None:
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)


def main():
    articles = collect_articles(TARGET_DOCS)
    save(articles)  # checkpoint content before touching the API at all

    fill_categories(articles)
    save(articles)

    print(f"\nDone: {len(articles)} articles saved to {OUTPUT_PATH}")
    if len(articles) < MIN_FINAL_DOCS:
        print(
            f"WARNING: only {len(articles)} documents collected, below the "
            f"required {MIN_FINAL_DOCS}. Increase TARGET_DOCS in this "
            f"script and rerun."
        )


if __name__ == "__main__":
    main()
