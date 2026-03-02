# batch_scraper.py
"""
Runs a batch of MBFC page scrapes concurrently.
Processes up to CONCURRENT_PAGES pages at a time within the batch.
Returns results for state tracking.
"""

import asyncio
import os
from typing import List, Tuple, Optional
from playwright.async_api import async_playwright, Browser, BrowserContext

from mbfc_scraper import MBFCScraper
from browserless_connection import connect_browser
from supabase_writer import SupabaseWriter
from logger import bot_logger

# How many pages to scrape in parallel within a batch
CONCURRENT_PAGES = int(os.getenv("CONCURRENT_PAGES", "5"))


class BatchResult:
    """Summary of a completed batch."""

    def __init__(self):
        self.succeeded: List[str] = []
        self.failed: List[str] = []
        self.details: List[dict] = []  # per-URL result details

    @property
    def total(self) -> int:
        return len(self.succeeded) + len(self.failed)

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return len(self.succeeded) / self.total * 100

    def summary_text(self, batch_number: int) -> str:
        lines = [
            f"Batch {batch_number} complete",
            f"  Scraped:  {self.total} pages",
            f"  Success:  {len(self.succeeded)}",
            f"  Failed:   {len(self.failed)}",
            f"  Rate:     {self.success_rate:.0f}%",
        ]
        if self.failed:
            lines.append(f"\nFailed URLs (first 5):")
            for url in self.failed[:5]:
                lines.append(f"  - {url}")
            if len(self.failed) > 5:
                lines.append(f"  ... and {len(self.failed) - 5} more")
        return "\n".join(lines)


async def _scrape_single(
    url: str,
    scraper: MBFCScraper,
    writer: SupabaseWriter,
    browser: Browser,
) -> Tuple[str, bool]:
    """
    Scrape one URL and write to Supabase.
    Returns (url, success: bool).
    """
    context: Optional[BrowserContext] = None
    try:
        # Each page gets its own context for isolation
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()

        extracted = await scraper.scrape_page(page, url)

        if not extracted:
            bot_logger.logger.warning(f"No data extracted from {url}")
            return (url, False)

        success = await writer.write(mbfc_url=url, extracted_data=extracted)
        return (url, success)

    except Exception as e:
        bot_logger.logger.error(f"Error processing {url}: {e}")
        return (url, False)

    finally:
        if context:
            try:
                await context.close()
            except Exception:
                pass


async def run_batch(
    urls: List[str],
    batch_number: int = 1,
    progress_callback=None,
) -> BatchResult:
    """
    Scrape a list of URLs in parallel (CONCURRENT_PAGES at a time).

    Args:
        urls: List of MBFC source page URLs to scrape
        batch_number: For logging purposes
        progress_callback: Optional async callable(message: str)

    Returns:
        BatchResult with succeeded/failed URL lists
    """
    result = BatchResult()

    if not urls:
        bot_logger.logger.warning("run_batch called with empty URL list.")
        return result

    bot_logger.logger.info(
        f"Starting batch {batch_number}: {len(urls)} URLs, "
        f"{CONCURRENT_PAGES} concurrent"
    )

    scraper = MBFCScraper()
    writer = SupabaseWriter()

    if not writer.enabled:
        raise RuntimeError("Supabase is not configured. Cannot run batch.")

    async with async_playwright() as pw:
        browser: Browser = await connect_browser(pw)

        # Process URLs in chunks of CONCURRENT_PAGES
        total = len(urls)
        done = 0

        for i in range(0, total, CONCURRENT_PAGES):
            chunk = urls[i: i + CONCURRENT_PAGES]

            tasks = [
                _scrape_single(url, scraper, writer, browser)
                for url in chunk
            ]

            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

            for res in chunk_results:
                if isinstance(res, Exception):
                    bot_logger.logger.error(f"Unexpected exception in chunk: {res}")
                    continue
                url, success = res
                if success:
                    result.succeeded.append(url)
                else:
                    result.failed.append(url)
                done += 1

            # Progress update after each chunk
            pct = round(done / total * 100)
            msg = f"Batch {batch_number} progress: {done}/{total} ({pct}%)"
            bot_logger.logger.info(msg)
            if progress_callback:
                await progress_callback(msg)

            # Small pause between chunks to be polite to MBFC
            if i + CONCURRENT_PAGES < total:
                await asyncio.sleep(2)

        await browser.close()

    bot_logger.logger.info(result.summary_text(batch_number))
    return result