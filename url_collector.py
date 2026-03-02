# url_collector.py
"""
Collects all individual source URLs from MBFC category listing pages.
Each category page lists sources as links — we extract all of them.
"""

import asyncio
import re
from typing import List, Set
from playwright.async_api import async_playwright, Browser, Page

from logger import bot_logger
from browserless_connection import connect_browser, USER_AGENT

BASE_URL = "https://mediabiasfactcheck.com"

# All MBFC category listing pages
CATEGORY_PAGES = [
    f"{BASE_URL}/center/",
    f"{BASE_URL}/left-center/",
    f"{BASE_URL}/left/",
    f"{BASE_URL}/right-center/",
    f"{BASE_URL}/right/",
    f"{BASE_URL}/conspiracy/",
    f"{BASE_URL}/fake-news/",
    f"{BASE_URL}/pro-science/",
    f"{BASE_URL}/satire/",
    f"{BASE_URL}/fact-checking-sources/",
]

# URL patterns that are NOT individual source pages — skip these
SKIP_PATTERNS = [
    "/category/",
    "/tag/",
    "/page/",
    "/news/",
    "/about/",
    "/methodology/",
    "/contact/",
    "/membership",
    "/donate",
    "/search/",
    "/appsextensions/",
    "/re-evaluated-sources/",
    "/filtered-search/",
    "/country-freedom-map/",
    "/fact-checks/",
    "?",
    "#",
    "mailto:",
    "javascript:",
]

# Known non-source top-level slugs to exclude
SKIP_SLUGS = {
    "center", "left-center", "left", "right-center", "right",
    "conspiracy", "fake-news", "pro-science", "satire",
    "fact-checking-sources", "news", "about", "methodology",
    "contact", "membership-account", "donate", "search",
    "appsextensions", "re-evaluated-sources", "filtered-search",
    "country-freedom-map", "fact-checks", "us-senators-ratings",
    "journalists", "politicians", "countries",
}


def _is_source_url(href: str) -> bool:
    """Return True if this href looks like an individual MBFC source page."""
    if not href or not href.startswith(BASE_URL):
        return False

    for pattern in SKIP_PATTERNS:
        if pattern in href:
            return False

    # Extract slug — the part after the base URL
    slug = href.replace(BASE_URL, "").strip("/").split("/")[0]

    if not slug:
        return False

    if slug in SKIP_SLUGS:
        return False

    # Must look like a reasonable slug: letters, numbers, hyphens
    if not re.match(r'^[a-z0-9][a-z0-9\-]+[a-z0-9]$', slug):
        return False

    return True


async def _scrape_category_page(page: Page, url: str) -> List[str]:
    """Scrape a single category page and return all source URLs found."""
    found = []
    try:
        bot_logger.logger.info(f"Collecting URLs from: {url}")
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)

        # Wait for article list to load
        try:
            await page.wait_for_selector("article a, .entry-content a, td a", timeout=10000)
        except Exception:
            pass

        # Extract all links on the page
        links = await page.evaluate("""
            () => {
                const anchors = document.querySelectorAll('a[href]');
                return Array.from(anchors).map(a => a.href);
            }
        """)

        for href in links:
            href = href.strip().split("?")[0].rstrip("/")
            # Normalise — ensure trailing slash removed for consistency
            if _is_source_url(href):
                found.append(href)

        bot_logger.logger.info(f"  -> Found {len(found)} source URLs on {url}")

    except Exception as e:
        bot_logger.logger.error(f"Failed to scrape category page {url}: {e}")

    return found


async def collect_all_urls(progress_callback=None) -> List[str]:
    """
    Visit all MBFC category pages and collect every individual source URL.
    Returns a deduplicated, sorted list.

    Args:
        progress_callback: Optional async callable(message: str) for status updates
    """
    all_urls: Set[str] = set()

    async with async_playwright() as pw:
        browser: Browser = await connect_browser(pw)
        context = await browser.new_context(user_agent=USER_AGENT)

        page = await context.new_page()

        for i, category_url in enumerate(CATEGORY_PAGES, 1):
            urls = await _scrape_category_page(page, category_url)
            all_urls.update(urls)

            msg = f"Category {i}/{len(CATEGORY_PAGES)}: {category_url.split('/')[-2]} -> {len(urls)} URLs found"
            bot_logger.logger.info(msg)

            if progress_callback:
                await progress_callback(msg)

            # Small delay between category pages
            await asyncio.sleep(2)

        await browser.close()

    result = sorted(list(all_urls))
    bot_logger.logger.info(f"URL collection complete. Total unique source URLs: {len(result)}")
    return result