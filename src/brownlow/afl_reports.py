"""Official AFL match reports from afl.com.au.

The public ``aflapi`` match index maps our Champion Data match ids to the
numeric afl.com.au match pages. Each match page's "Match Report" tab carries
the report article id, and the article page is server-rendered prose. We store
a short excerpt plus the headline and link, keyed by our match id.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from html import unescape as html_unescape

import pandas as pd

INDEX_URL = "https://aflapi.afl.com.au/afl/v2/matches"
MATCH_URL = "https://www.afl.com.au/afl/matches/{match_id}"
NEWS_URL = "https://www.afl.com.au/news/{article_id}"
SOURCE = "AFL.com.au"
MAX_SUMMARY_CHARS = 480
PAGE_SIZE = 100

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36 brownlow-research"
)
_TAGS = re.compile(r"<[^>]+>")
_ARTICLE_ID = re.compile(r'data-ui-tab="Match Report".*?data-article-id="(\d+)"', re.DOTALL)
_HEADING = re.compile(r'<h1[^>]*class="[^"]*article__heading[^"]*"[^>]*>(.*?)</h1>', re.DOTALL)


def _get(url: str, retries: int = 3, pause: float = 1.0) -> str:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception as error:  # noqa: BLE001 - network scrape, retry any failure
            last_error = error
            time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def fetch_match_index(season_prefix: str = "CD_M2026", pages: range | None = None) -> dict[str, int]:
    """Map Champion Data match ids to afl.com.au numeric match ids.

    Matches are chronological, so the target season sits at the end of the
    public index; only the last ``pages`` pages are scanned by default.
    """
    page_range = pages or range(70, 86)
    mapping: dict[str, int] = {}
    for page in page_range:
        url = f"{INDEX_URL}?" + urllib.parse.urlencode({"page": page, "pageSize": PAGE_SIZE})
        try:
            payload = json.loads(_get(url))
        except RuntimeError:
            continue
        for match in payload.get("matches", []):
            provider_id = str(match.get("providerId", ""))
            if provider_id.startswith(season_prefix) and match.get("id") is not None:
                mapping[provider_id] = int(match["id"])
    return mapping


def match_page_article_id(match_html: str) -> str | None:
    """Article id from the Match Report tab of a match page."""
    found = _ARTICLE_ID.search(match_html)
    return found.group(1) if found else None


def article_summary(article_html: str) -> tuple[str, str]:
    """Headline and lead paragraph of a report article."""
    heading = _HEADING.search(article_html)
    headline = " ".join(html_unescape(_TAGS.sub(" ", heading.group(1))).split()) if heading else ""
    start = article_html.find('class="article__body"')
    if start < 0:
        start = article_html.find('class="article-body"')
    region = article_html[start : start + 20000] if start >= 0 else article_html[:200000]
    for chunk in re.findall(r"<p[^>]*>(.*?)</p>", region, re.DOTALL):
        text = " ".join(html_unescape(_TAGS.sub(" ", chunk)).split())
        if len(text) < 60 or any(marker in text for marker in ("function(", "var ", "-->")):
            continue
        if len(text) > MAX_SUMMARY_CHARS:
            cut = text[:MAX_SUMMARY_CHARS]
            boundary = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
            text = cut[: boundary + 1] if boundary > 120 else cut.rstrip() + "..."
        return headline, text
    return headline, ""


def fetch_reports(
    fixtures: pd.DataFrame,
    existing: dict[str, dict] | None = None,
    *,
    rounds: list[int] | None = None,
    pause: float = 0.4,
) -> dict[str, dict]:
    """Fetch report excerpts for every fixture that lacks one.

    ``fixtures`` needs ``match_id``, ``round``, ``season``, ``home`` and
    ``away`` columns; results are keyed by ``match_id``.
    """
    reports = dict(existing or {})
    work = fixtures.copy()
    if rounds is not None:
        work = work[work["round"].isin(rounds)]
    prefixes = sorted({f"CD_M{int(row.season)}" for row in work.itertuples(index=False)})
    index: dict[str, int] = {}
    for prefix in prefixes:
        index.update(fetch_match_index(prefix))

    for row in work.sort_values(["round", "match_id"]).itertuples(index=False):
        match_id = str(row.match_id)
        if match_id in reports and reports[match_id].get("text"):
            continue
        numeric_id = index.get(match_id)
        if numeric_id is None:
            print(f"  {row.round}: {row.home} v {row.away} -> no afl.com.au match id")
            continue
        try:
            match_html = _get(MATCH_URL.format(match_id=numeric_id))
            article_id = match_page_article_id(match_html)
            if not article_id:
                print(f"  {row.round}: {row.home} v {row.away} -> no report article")
                continue
            article_html = _get(NEWS_URL.format(article_id=article_id))
            headline, text = article_summary(article_html)
        except RuntimeError:
            print(f"  {row.round}: {row.home} v {row.away} -> fetch failed")
            continue
        if not text:
            print(f"  {row.round}: {row.home} v {row.away} -> empty report")
            continue
        reports[match_id] = {
            "source": SOURCE,
            "url": NEWS_URL.format(article_id=article_id),
            "headline": headline,
            "round": int(row.round),
            "text": text,
        }
        print(f"  {row.round}: {row.home} v {row.away} -> {headline[:70]}")
        time.sleep(pause)
    return reports
