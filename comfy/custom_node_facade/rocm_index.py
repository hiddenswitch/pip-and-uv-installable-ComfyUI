"""Allowlisted proxy for AMD's TheRock Python package index.

TheRock's ``whl-next`` repository is a PEP 503 simple index today.  Keeping
the proxy explicit (rather than accepting arbitrary upstream URLs) avoids
turning the package facade into an SSRF/open-proxy service.  Distribution
files remain hosted by AMD; only the small HTML index pages pass through the
facade.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from urllib.parse import urljoin

import aiohttp


ROCM_SIMPLE_INDEX_URL = "https://stable.repo.amd.com/rocm/whl-next/"
_HREF_RE = re.compile(r"(?P<prefix>\bhref\s*=\s*)(?P<quote>[\"'])(?P<url>.*?)(?P=quote)", re.IGNORECASE)


def _absolute_links(body: str, base_url: str) -> str:
    """Make relative PEP 503 links absolute while preserving all attributes."""

    def replace(match: re.Match[str]) -> str:
        target = match.group("url")
        if target.startswith(("http://", "https://", "data:", "#")):
            return match.group(0)
        absolute = urljoin(base_url, target)
        return f'{match.group("prefix")}{match.group("quote")}{html.escape(absolute, quote=True)}{match.group("quote")}'

    return _HREF_RE.sub(replace, body)


@dataclass(frozen=True)
class RocmSimpleIndexProxy:
    """Proxy AMD's stable all-architecture TheRock simple repository."""

    root_url: str = ROCM_SIMPLE_INDEX_URL

    async def render_root(self, session: aiohttp.ClientSession) -> str:
        return await self._fetch(session, self.root_url, self.root_url)

    async def render_project(self, session: aiohttp.ClientSession, project: str) -> str:
        project_url = urljoin(self.root_url, f"{project.rstrip('/')}/")
        return await self._fetch(session, project_url, project_url)

    @staticmethod
    async def _fetch(session: aiohttp.ClientSession, url: str, link_base: str) -> str:
        async with session.get(url) as response:
            response.raise_for_status()
            return _absolute_links(await response.text(), link_base)
