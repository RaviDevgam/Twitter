"""
Delete your own posts, replies and reposts on X.

Usage:
    python cleanup.py                              # dry-run: list latest ~3200 via API, delete nothing
    python cleanup.py --archive data/tweets.js     # dry-run using your downloaded X archive (free, complete)
    python cleanup.py --before 2026-05-01          # only items created before this date
    python cleanup.py --archive data/tweets.js --confirm   # actually delete

Progress is saved to deleted.json, so you can stop and re-run safely.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from urllib import error, parse, request

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from post import refresh_access_token

ROOT = os.path.dirname(os.path.abspath(__file__))
DELETED_PATH = os.path.join(ROOT, "deleted.json")
API = "https://api.x.com/2"
COST_PER_DELETE = 0.005

state = {
    "access": os.getenv("USER_ACCESS_TOKEN"),
    "refresh": os.getenv("REFRESH_TOKEN"),
}


# ---------- http ----------

def api(method: str, path: str, params: dict | None = None) -> dict:
    """Authorized X API call with token refresh on 401 and waiting on 429."""
    url = f"{API}{path}" + (f"?{parse.urlencode(params)}" if params else "")
    for _ in range(5):
        req = request.Request(url, method=method,
                              headers={"Authorization": f"Bearer {state['access']}"})
        try:
            with request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8") or "{}")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8") if exc.fp else exc.reason
            if exc.code == 401 and state["refresh"]:
                print("  token expired, refreshing...")
                state["access"], state["refresh"] = refresh_access_token(state["refresh"])
                continue
            if exc.code == 429:
                reset = int(exc.headers.get("x-rate-limit-reset", time.time() + 60))
                wait = max(5, reset - int(time.time()) + 2)
                print(f"  rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            raise RuntimeError(f"{exc.code} {exc.reason}: {body}") from exc
    raise RuntimeError(f"gave up on {method} {path}")


# ---------- collect ----------

def from_archive(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # file looks like: window.YTD.tweets.part0 = [ ... ]
    data = json.loads(text[text.index("["):])
    items = []
    for entry in data:
        t = entry.get("tweet", entry)
        created = datetime.strptime(t["created_at"], "%a %b %d %H:%M:%S %z %Y")
        items.append({"id": t["id_str"], "created": created, "text": t.get("full_text", "")})
    return items


def from_api() -> list[dict]:
    me = api("GET", "/users/me")["data"]
    print(f"  account: @{me['username']}")
    items, token = [], None
    while True:
        params = {"max_results": 100, "tweet.fields": "created_at"}
        if token:
            params["pagination_token"] = token
        page = api("GET", f"/users/{me['id']}/tweets", params)
        for t in page.get("data", []):
            created = datetime.fromisoformat(t["created_at"].replace("Z", "+00:00"))
            items.append({"id": t["id"], "created": created, "text": t["text"]})
        token = page.get("meta", {}).get("next_token")
        print(f"  fetched {len(items)}...")
        if not token:
            return items


def kind(item: dict) -> str:
    return "repost" if item["text"].startswith("RT @") else "post/reply"


# ---------- main ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", help="Path to tweets.js from your X data archive")
    parser.add_argument("--before", help="Only delete items created before YYYY-MM-DD")
    parser.add_argument("--confirm", action="store_true", help="Actually delete (otherwise dry-run)")
    args = parser.parse_args()

    if not state["access"]:
        sys.exit("USER_ACCESS_TOKEN missing — run `python post.py` once to log in.")

    print("Collecting your posts...")
    items = from_archive(args.archive) if args.archive else from_api()

    if args.before:
        cutoff = datetime.strptime(args.before, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        items = [it for it in items if it["created"] < cutoff]

    done = set(json.load(open(DELETED_PATH)) if os.path.exists(DELETED_PATH) else [])
    todo = [it for it in items if it["id"] not in done]

    reposts = sum(1 for it in todo if kind(it) == "repost")
    print(f"\n{len(todo)} to delete ({len(todo) - reposts} posts/replies, {reposts} reposts)"
          f"  |  already deleted: {len(done)}")
    print(f"Estimated cost: ~${len(todo) * COST_PER_DELETE:.2f}\n")

    for it in todo[:10]:
        print(f"  {it['created']:%Y-%m-%d}  [{kind(it)}]  {it['text'][:70]!r}")
    if len(todo) > 10:
        print(f"  ... and {len(todo) - 10} more")

    if not args.confirm:
        print("\nDry-run. Nothing deleted. Add --confirm to delete.")
        return

    if input(f"\nType DELETE to permanently remove {len(todo)} items: ").strip() != "DELETE":
        sys.exit("Cancelled.")

    failed = 0
    for i, it in enumerate(todo, 1):
        try:
            # works for posts, replies and reposts (a repost's own id undoes the repost)
            api("DELETE", f"/tweets/{it['id']}")
            done.add(it["id"])
            print(f"  [{i}/{len(todo)}] deleted {it['id']}")
        except RuntimeError as exc:
            if "404" in str(exc):  # already gone
                done.add(it["id"])
            else:
                failed += 1
                print(f"  [{i}/{len(todo)}] ! failed {it['id']}: {exc}")
        if i % 20 == 0:
            json.dump(sorted(done), open(DELETED_PATH, "w"))
    json.dump(sorted(done), open(DELETED_PATH, "w"))
    print(f"\nDone. Deleted {len(todo) - failed}, failed {failed}.")


if __name__ == "__main__":
    main()
