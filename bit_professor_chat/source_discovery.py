from __future__ import annotations

import re
from collections.abc import Sequence
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from .ingestion_models import ProfessorListing


LISTING_URL = "https://isc.bit.edu.cn/schools/csat/knowingprofessors5/index.htm"


def build_requests_session(user_agent: str = "agents-tutorial/0.1") -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    return session


def fetch_soup(url: str, session: requests.Session) -> BeautifulSoup:
    response = session.get(url, timeout=30)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    return BeautifulSoup(response.text, "html.parser")


def discover_listing_pages(listing_url: str, session: requests.Session) -> list[str]:
    soup = fetch_soup(listing_url, session)
    pages = {listing_url}
    for anchor in soup.select("a[href$='.htm']"):
        href = anchor.get("href", "").strip()
        if not href:
            continue
        if re.fullmatch(r"index\d*\.htm", href):
            pages.add(urljoin(listing_url, href))

    def page_key(url: str) -> tuple[int, str]:
        match = re.search(r"index(\d*)\.htm$", url)
        suffix = match.group(1) if match else ""
        return (0 if suffix == "" else int(suffix), url)

    return sorted(pages, key=page_key)


def collect_professor_links_from_page(
    listing_url: str, session: requests.Session
) -> list[ProfessorListing]:
    soup = fetch_soup(listing_url, session)
    professor_items: list[ProfessorListing] = []
    seen: set[str] = set()
    for anchor in soup.select("a[href$='.htm']"):
        name = anchor.get_text(" ", strip=True)
        href = anchor.get("href", "").strip()
        if not name or not re.fullmatch(r"b\d+\.htm", href):
            continue
        detail_url = urljoin(listing_url, href)
        if detail_url in seen:
            continue
        seen.add(detail_url)
        professor_items.append(ProfessorListing(name=name, detail_url=detail_url))
    return professor_items


def collect_professor_links(
    listing_pages: Sequence[str], session: requests.Session
) -> list[ProfessorListing]:
    collected: list[ProfessorListing] = []
    seen: set[str] = set()
    for page_url in listing_pages:
        for item in collect_professor_links_from_page(page_url, session):
            if item.detail_url in seen:
                continue
            seen.add(item.detail_url)
            collected.append(item)
    return collected


def find_professor_listing_by_detail_url(
    detail_url: str,
    *,
    session: requests.Session,
    listing_url: str = LISTING_URL,
) -> ProfessorListing | None:
    normalized_detail_url = detail_url.strip()
    listing_pages = discover_listing_pages(listing_url, session)
    for listing in collect_professor_links(listing_pages, session):
        if listing.detail_url.strip() == normalized_detail_url:
            return listing
    return None


def extract_image_urls(detail_url: str, session: requests.Session) -> list[str]:
    soup = fetch_soup(detail_url, session)
    image_urls: list[str] = []
    for image in soup.select("img[src]"):
        src = image.get("src", "").strip()
        if not src:
            continue
        absolute_url = urljoin(detail_url, src)
        if "/images/" not in absolute_url:
            continue
        if absolute_url not in image_urls:
            image_urls.append(absolute_url)
    return image_urls
