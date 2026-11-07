"""
Pipeline: fetch AI news (RSS) -> filter & rank -> GPT rewrite -> auto-post.

Usage:
    python pipeline.py                     # dry-run (shows what would be posted)
    python pipeline.py --post              # actually post the top candidate
    python pipeline.py --post --top 3      # post top 3 (spaced out)
    python pipeline.py --post --approve    # ask y/n before posting each tweet
"""

import argparse
import hashlib
import html
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib import error, request

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from post import post_tweet, refresh_access_token  # reuse posting logic

USER_ACCESS_TOKEN = os.getenv("USER_ACCESS_TOKEN")
REFRESH_TOKEN = os.getenv("REFRESH_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

ROOT = os.path.dirname(os.path.abspath(__file__))
TOPICS_PATH = os.path.join(ROOT, "topics.json")
POSTED_PATH = os.path.join(ROOT, "posted.json")

OPENAI_URL = "https://api.openai.com/v1/chat/completions"


# ---------- helpers ----------

def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def http_json(method: str, url: str, *, headers: dict, body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = request.Request(url=url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        msg = exc.read().decode("utf-8") if exc.fp else exc.reason
        raise RuntimeError(f"{exc.code} {exc.reason}: {msg}") from exc


# ---------- 1. fetch news ----------

def fetch_rss(url: str) -> list[dict]:
    req = request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with request.urlopen(req, timeout=30) as resp:
        root = ET.fromstring(resp.read())

    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        source = (item.findtext("source") or "").strip()
        # Google News titles end with " - Source"; strip it
        if source and title.endswith(f" - {source}"):
            title = title[: -len(f" - {source}")].strip()
        desc = html.unescape(re.sub(r"<[^>]+>", " ", item.findtext("description") or ""))
        desc = re.sub(r"\s+", " ", desc).strip()
        pub = item.findtext("pubDate")
        try:
            published = parsedate_to_datetime(pub) if pub else None
        except (TypeError, ValueError):
            published = None
        items.append({
            "id": (item.findtext("guid") or item.findtext("link") or title).strip(),
            "title": html.unescape(title),
            "source": source,
            "description": desc,
            "link": (item.findtext("link") or "").strip(),
            "published": published,
        })
    return items


# ---------- 2. filter & rank ----------

def score_item(item: dict, cfg: dict) -> float:
    text = item["title"].lower()

    # too-short headlines ("Introducing dots") give GPT nothing to work with
    if len(text.split()) < cfg["min_title_words"]:
        return -1

    for word in cfg["exclude_keywords"]:
        if word.lower() in text:
            return -1

    age_hours = None
    if item["published"]:
        age_hours = (datetime.now(timezone.utc) - item["published"]).total_seconds() / 3600
        if age_hours > cfg["max_age_hours"]:
            return -1

    boost = sum(1 for w in cfg["boost_keywords"] if w.lower() in text)
    source_bonus = 5 if item["source"] in cfg["priority_sources"] else 0
    freshness = max(0.0, cfg["max_age_hours"] - age_hours) / cfg["max_age_hours"] * 5 if age_hours is not None else 0
    return boost * 2 + source_bonus + freshness


def gather_candidates(cfg: dict) -> list[dict]:
    seen = set()
    candidates = []
    for url in cfg["rss_feeds"]:
        print(f"  feed: {url[:80]}...")
        try:
            items = fetch_rss(url)
        except (error.URLError, ET.ParseError) as exc:
            print(f"    ! feed failed: {exc}")
            continue
        print(f"    {len(items)} items")
        for it in items:
            # dedup by id and by near-identical headline (same story from many outlets)
            title_key = re.sub(r"\W+", "", it["title"].lower())[:60]
            if it["id"] in seen or title_key in seen:
                continue
            seen.update({it["id"], title_key})
            score = score_item(it, cfg)
            if score < 0:
                continue
            it["_score"] = score
            candidates.append(it)
    candidates.sort(key=lambda it: it["_score"], reverse=True)
    return candidates


# ---------- 3. GPT rewrite ----------

def rewrite_with_gpt(item: dict, cfg: dict) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing in environment/.env")

    user_msg = (
        f"Headline: {item['title']}\n"
        f"Source: {item['source'] or 'unknown'}\n"
        f"Published: {item['published'].strftime('%Y-%m-%d') if item['published'] else 'unknown'}\n\n"
        f"Allowed hashtags: {' '.join(cfg['hashtags'])}\n"
        f"Max length: {cfg['max_post_length']} characters.\n\n"
        f"Write ONE tweet about this news. Output only the tweet."
    )
    body = {
        "model": cfg["openai_model"],
        "messages": [
            {"role": "system", "content": cfg["system_prompt"]},
            {"role": "user", "content": user_msg},
        ],
    }
    result = http_json("POST", OPENAI_URL, headers={
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }, body=body)
    text = result["choices"][0]["message"]["content"].strip()
    # strip wrapping quotes if the model adds them
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    # links cost $0.20/post on X API — never post them
    text = re.sub(r"\s*https?://\S+", "", text).strip()
    if len(text) > cfg["max_post_length"]:
        text = text[: cfg["max_post_length"] - 1].rstrip() + "…"
    return text


# ---------- 4. dedup ----------

def dedup_key(item: dict) -> str:
    return hashlib.sha1(item["id"].encode()).hexdigest()


# ---------- 5. main ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--post", action="store_true", help="Actually post (otherwise dry-run)")
    parser.add_argument("--top", type=int, default=1, help="Number of tweets to post")
    parser.add_argument("--approve", action="store_true", help="Ask y/n before posting each tweet")
    args = parser.parse_args()

    cfg = load_json(TOPICS_PATH, None)
    if cfg is None:
        sys.exit("topics.json not found")

    posted = load_json(POSTED_PATH, {"ids": []})
    posted_ids = set(posted["ids"])

    print("Step 1: fetching news...")
    candidates = gather_candidates(cfg)
    print(f"  -> {len(candidates)} candidates after filter")

    fresh = [c for c in candidates if dedup_key(c) not in posted_ids]
    print(f"  -> {len(fresh)} after dedup")
    if not fresh:
        sys.exit("No fresh candidates. Try again later or relax filters.")

    selected = fresh[: args.top]
    print(f"\nStep 2: rewriting top {len(selected)} with {cfg['openai_model']}...\n")

    for i, src in enumerate(selected, 1):
        print(f"=== Candidate {i}/{len(selected)}  score={src['_score']:.1f}  source={src['source']} ===")
        print(f"HEADLINE: {src['title']}")
        try:
            tweet = rewrite_with_gpt(src, cfg)
        except RuntimeError as exc:
            print(f"! GPT failed: {exc}")
            continue
        print(f"REWRITE ({len(tweet)} chars): {tweet}\n")

        if not args.post:
            print("(dry-run, not posting)\n")
            continue

        if args.approve and input("Post this? [y/N]: ").strip().lower() != "y":
            print("skipped\n")
            continue

        access_token = USER_ACCESS_TOKEN
        try:
            result = post_tweet(tweet, access_token)
        except RuntimeError as exc:
            if "401" in str(exc) and REFRESH_TOKEN:
                print("Refreshing token...")
                access_token, _ = refresh_access_token(REFRESH_TOKEN)
                result = post_tweet(tweet, access_token)
            else:
                print(f"! Post failed: {exc}")
                continue

        tweet_id = result.get("data", {}).get("id")
        print(f"POSTED: https://x.com/i/web/status/{tweet_id}\n")
        posted_ids.add(dedup_key(src))
        save_json(POSTED_PATH, {"ids": list(posted_ids)})
        time.sleep(2)  # spacing between posts


if __name__ == "__main__":
    main()
