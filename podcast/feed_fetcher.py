"""RSS ingestion for AI industry news."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List

import feedparser

logger = logging.getLogger(__name__)

# Five reliable AI-industry RSS feeds. Each entry is (label, url) so log
# output and article attribution stay readable even if a feed's own title
# field is inconsistent.
RSS_FEEDS = [
    ("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/"),
    ("The Verge AI", "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml"),
    ("VentureBeat AI", "https://venturebeat.com/category/ai/feed/"),
    ("Ars Technica AI", "https://arstechnica.com/ai/feed/"),
    ("MIT Technology Review AI", "https://www.technologyreview.com/topic/artificial-intelligence/feed"),
]

CADENCE_WINDOWS = {
    "daily": timedelta(hours=24),
    "weekly": timedelta(days=7),
}


@dataclass
class Article:
    source: str
    title: str
    link: str
    summary: str
    published: datetime


class FeedFetcher:
    """Fetches and filters recent articles from a hardcoded list of RSS feeds.

    Designed to degrade gracefully: a single feed timing out, returning
    malformed XML, or being temporarily unreachable never aborts the run --
    it's logged and skipped so the pipeline continues with whatever feeds
    did work.
    """

    def __init__(
        self,
        feeds: List[tuple] | None = None,
        timeout_seconds: int = 15,
        per_feed_limit: int = 2,
    ):
        self.feeds = feeds or RSS_FEEDS
        self.timeout_seconds = timeout_seconds
        # Cap how many recent items each individual feed contributes, so a
        # high-volume source (e.g. TechCrunch) can't crowd out every other
        # source once results are pooled and sorted by recency.
        self.per_feed_limit = per_feed_limit

    def fetch_recent(self, cadence: str = "daily") -> List[Article]:
        window = CADENCE_WINDOWS.get(cadence, CADENCE_WINDOWS["daily"])
        cutoff = datetime.now(timezone.utc) - window

        articles: List[Article] = []
        for label, url in self.feeds:
            try:
                articles.extend(self._fetch_one(label, url, cutoff))
            except Exception as exc:  # noqa: BLE001 - modular error handling by design
                logger.warning("Skipping feed '%s' (%s) after error: %s", label, url, exc)
                continue

        articles.sort(key=lambda a: a.published, reverse=True)
        logger.info(
            "Fetched %d recent article(s) across %d feed(s) (max %d per feed)",
            len(articles),
            len(self.feeds),
            self.per_feed_limit,
        )
        return articles

    def _fetch_one(self, label: str, url: str, cutoff: datetime) -> List[Article]:
        start = time.monotonic()
        parsed = feedparser.parse(url)
        elapsed = time.monotonic() - start

        if parsed.get("bozo") and not parsed.get("entries"):
            raise ValueError(f"unparseable feed (bozo_exception={parsed.get('bozo_exception')})")

        results: List[Article] = []
        for entry in parsed.get("entries", []):
            published = self._extract_published(entry)
            if published is None or published < cutoff:
                continue
            results.append(
                Article(
                    source=label,
                    title=entry.get("title", "Untitled").strip(),
                    link=entry.get("link", ""),
                    summary=self._clean_summary(entry),
                    published=published,
                )
            )

        results.sort(key=lambda a: a.published, reverse=True)
        total_recent = len(results)
        results = results[: self.per_feed_limit]

        logger.info(
            "Fetched %d recent item(s) from %s in %.2fs (%d available, capped to %d)",
            len(results),
            label,
            elapsed,
            total_recent,
            self.per_feed_limit,
        )
        return results

    @staticmethod
    def _extract_published(entry) -> datetime | None:
        for field in ("published_parsed", "updated_parsed"):
            value = entry.get(field)
            if value:
                return datetime.fromtimestamp(time.mktime(value), tz=timezone.utc)
        return None

    @staticmethod
    def _clean_summary(entry) -> str:
        raw = entry.get("summary", "") or entry.get("description", "")
        # Strip any embedded HTML tags without pulling in a full HTML parser dependency.
        import re

        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:1200]
