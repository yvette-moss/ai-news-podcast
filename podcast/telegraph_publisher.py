"""Publishes the episode script as a Telegraph (telegra.ph) article.

Telegram opens telegra.ph links in Instant View, so a long script reads as a
native in-app article instead of being split across several messages.
API docs: https://telegra.ph/api
"""

from __future__ import annotations

import json
import logging
from typing import List

import requests

from .feed_fetcher import Article

logger = logging.getLogger(__name__)

TELEGRAPH_API_BASE = "https://api.telegra.ph/{method}"
AUTHOR_NAME = "AI News Podcast"


class TelegraphPublishError(RuntimeError):
    pass


class TelegraphPublisher:
    def __init__(self, timeout_seconds: int = 30):
        self.timeout_seconds = timeout_seconds

    def publish(self, title: str, script_text: str, source_articles: List[Article]) -> str:
        """Create a Telegraph page and return its URL."""
        # No signup needed: an anonymous account is created on the fly.
        token = self._call("createAccount", {"short_name": "AINewsPodcast", "author_name": AUTHOR_NAME})["access_token"]
        page = self._call(
            "createPage",
            {
                "access_token": token,
                "title": title[:256],
                "author_name": AUTHOR_NAME,
                "content": json.dumps(self._build_content(script_text, source_articles)),
                "return_content": "false",
            },
        )
        logger.info("Script published to Telegraph -> %s", page["url"])
        return page["url"]

    @staticmethod
    def _build_content(script_text: str, source_articles: List[Article]) -> list:
        # Telegraph content is a list of DOM-like nodes; plain strings are auto-escaped.
        nodes: list = [
            {"tag": "p", "children": [paragraph.strip()]}
            for paragraph in script_text.strip().split("\n\n")
            if paragraph.strip()
        ]
        if source_articles:
            nodes.append({"tag": "h4", "children": ["Sources"]})
            nodes.append(
                {
                    "tag": "ol",
                    "children": [
                        {
                            "tag": "li",
                            "children": [
                                {"tag": "a", "attrs": {"href": a.link}, "children": [a.title]},
                                f" — {a.source}",
                            ],
                        }
                        for a in source_articles
                    ],
                }
            )
        return nodes

    def _call(self, method: str, data: dict) -> dict:
        url = TELEGRAPH_API_BASE.format(method=method)
        try:
            response = requests.post(url, data=data, timeout=self.timeout_seconds)
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise TelegraphPublishError(f"Telegraph {method} request failed: {exc}") from exc
        if not payload.get("ok"):
            raise TelegraphPublishError(f"Telegraph {method} rejected: {payload.get('error', payload)}")
        return payload["result"]
