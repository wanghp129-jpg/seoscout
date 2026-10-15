#!/usr/bin/env python3
"""
Step 2: Collect content

Read search_results.json, extract YouTube transcripts and web page content,
output per-keyword JSON files in collected/.
"""

import asyncio
import argparse
import os
import re
from datetime import datetime
from collections import defaultdict

from .core.config import Config
from .core.youtube import YouTube
from .core.web import Web
from .core.models import YouTubeItem, WebItem
from .core.utils import load_json, save_json, ensure_dir


def keyword_to_filename(keyword: str) -> str:
    return re.sub(r'[^a-z0-9_]', '', keyword.lower().replace(' ', '_').replace('/', '_'))


def deduplicate_items(items_by_keyword, item_type="video"):
    url_map = {}

    for keyword, items in items_by_keyword.items():
        for item_dict in items:
            url = item_dict.get('url')
            if not url:
                continue

            if url not in url_map:
                url_map[url] = {
                    'item': item_dict,
                    'keywords': []
                }

            if keyword not in url_map[url]['keywords']:
                url_map[url]['keywords'].append(keyword)

    unique_items = []
    url_to_keywords = {}

    for url, data in url_map.items():
        unique_items.append(data['item'])
        url_to_keywords[url] = data['keywords']

    return unique_items, url_to_keywords


async def run_collect(project: str, entries=None):
    """Run the extract step programmatically."""
    Config.init(project)

    print("=" * 70)
    print(f"  Step 2: Collect [{project}]")
    print("=" * 70)

    input_file = f"{Config.OUT_DIR}/search_results.json"
    try:
        data = load_json(input_file)
    except FileNotFoundError:
        print(f"❌ Not found: {input_file}")
        print(f"Run `seoscout search` first")
        return

    yt_items_by_keyword = {}
    web_items_by_keyword = {}

    collected_dir = f"{Config.OUT_DIR}/collected"
    skipped_count = 0

    rows = data["keywords"]
    if entries is not None:
        identities = {(entry.keyword, entry.category) for entry in entries}
        rows = [row for row in rows if (row['keyword'], row.get('category', '')) in identities]
    for kw_data in rows:
        keyword = kw_data["keyword"]
        category = kw_data.get("category", "")

        # Build collected path with optional category subdirectory
        keyword_file = keyword_to_filename(keyword)
        if category:
            category_slug = category.lower().replace(' ', '-')
            collected_path = f"{collected_dir}/{category_slug}/{keyword_file}.json"
        else:
            collected_path = f"{collected_dir}/{keyword_file}.json"
        if os.path.exists(collected_path):
            skipped_count += 1
            continue

        if "youtube" in kw_data and kw_data["youtube"].get("items"):
            selected_yt = [
                item_dict for item_dict in kw_data["youtube"]["items"]
                if item_dict.get("selected", True)
            ]
            yt_items_by_keyword[keyword] = selected_yt[:Config.YOUTUBE_EXTRACT_TOP_K]
        else:
            yt_items_by_keyword[keyword] = []

        if "web" in kw_data and kw_data["web"].get("items"):
            selected_web = [
                item_dict for item_dict in kw_data["web"]["items"]
                if item_dict.get("selected", True)
            ]
            web_items_by_keyword[keyword] = selected_web[:Config.WEB_EXTRACT_TOP_K]
        else:
            web_items_by_keyword[keyword] = []

    if skipped_count > 0:
        print(f"\n⏭️  Skipped {skipped_count} keywords (collected file exists)")

    print("\n📋 Before dedup:")
    total_yt_before = sum(len(items) for items in yt_items_by_keyword.values())
    total_web_before = sum(len(items) for items in web_items_by_keyword.values())
    print(f"  - YouTube: {total_yt_before} videos")
    print(f"  - Web:     {total_web_before} pages")

    unique_yt_dicts, yt_url_to_keywords = deduplicate_items(yt_items_by_keyword, "video")
    unique_web_dicts, web_url_to_keywords = deduplicate_items(web_items_by_keyword, "page")

    print("\n📋 After dedup:")
    print(f"  - YouTube: {len(unique_yt_dicts)} unique (removed {total_yt_before - len(unique_yt_dicts)} dupes)")
    print(f"  - Web:     {len(unique_web_dicts)} unique (removed {total_web_before - len(unique_web_dicts)} dupes)")

    yt_items = [YouTubeItem(**item_dict) for item_dict in unique_yt_dicts]
    web_items = [WebItem(**item_dict) for item_dict in unique_web_dicts]

    print(f"\n📋 To collect:")
    print(f"  - YouTube: {len(yt_items)} videos")
    print(f"  - Web:     {len(web_items)} pages")

    if not yt_items and not web_items:
        print("\n⚠️  Nothing to collect")
        return

    yt = YouTube()
    web = Web()

    print("\n" + "=" * 70)
    print("  Collecting content...")
    print("=" * 70)

    yt_results, web_results = await asyncio.gather(
        yt.extract_batch(yt_items),
        web.extract_batch(web_items)
    )

    results = defaultdict(lambda: {"youtube": [], "web": []})

    for item, content in yt_results:
        if content:
            url = item.url
            keywords = yt_url_to_keywords.get(url, [])
            for keyword in keywords:
                results[keyword]["youtube"].append({
                    "type": "youtube",
                    "title": item.title,
                    "url": item.url,
                    "content": content
                })

    for item, content in web_results:
        if content:
            url = item.url
            keywords = web_url_to_keywords.get(url, [])
            for keyword in keywords:
                results[keyword]["web"].append({
                    "type": "web",
                    "title": item.title,
                    "url": item.url,
                    "content": content
                })

    print("\n" + "=" * 70)
    print("  Saving results")
    print("=" * 70)

    ensure_dir(collected_dir)

    total_saved = 0
    for kw_data in rows:
        keyword = kw_data["keyword"]
        category = kw_data.get("category", "")
        keyword_result = results.get(keyword, {"youtube": [], "web": []})

        if not keyword_result["youtube"] and not keyword_result["web"]:
            continue

        keyword_file = keyword_to_filename(keyword)
        if category:
            category_slug = category.lower().replace(' ', '-')
            output_file = f"{collected_dir}/{category_slug}/{keyword_file}.json"
        else:
            output_file = f"{collected_dir}/{keyword_file}.json"

        output = {
            "keyword": keyword,
            "category": category,
            "collected_at": datetime.now().isoformat(),
            "sources": {
                "youtube": {
                    "count": len(keyword_result["youtube"]),
                    "videos": keyword_result["youtube"]
                },
                "web": {
                    "count": len(keyword_result["web"]),
                    "pages": keyword_result["web"]
                }
            },
            "total_sources": len(keyword_result["youtube"]) + len(keyword_result["web"])
        }

        save_json(output, output_file)
        total_saved += 1
        print(f"  ✓ {keyword_file}.json")

    print("\n" + "=" * 70)
    print("  ✅ Collect complete")
    print("=" * 70)

    yt_success = sum(1 for _, content in yt_results if content)
    yt_failed = len(yt_results) - yt_success
    print(f"YouTube: {yt_success}/{len(yt_results)} succeeded")
    if yt_failed > 0:
        print(f"  Failed:")
        for item, content in yt_results:
            if not content:
                print(f"    - {item.video_id}: {item.title}")

    web_success = sum(1 for _, content in web_results if content)
    web_failed = len(web_results) - web_success
    print(f"\nWeb: {web_success}/{len(web_results)} succeeded")
    if web_failed > 0:
        print(f"  Failed:")
        for item, content in web_results:
            if not content:
                print(f"    - {item.url}: {item.title}")

    print(f"\nSaved: {total_saved} files → {collected_dir}/")
    print("=" * 70)


async def main():
    parser = argparse.ArgumentParser(description="Step 2: Collect content")
    parser.add_argument("--project", "--domain", required=True,
                        help="Project name")
    args = parser.parse_args()

    await run_collect(args.project)


if __name__ == "__main__":
    asyncio.run(main())
