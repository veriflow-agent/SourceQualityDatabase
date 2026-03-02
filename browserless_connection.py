# browserless_connection.py
"""
Shared helper for connecting to Railway Browserless.
Uses chromium.connect() for Railway, falls back to chromium.launch() locally.
Mirrors the pattern from VeriFlow's BrowserlessScraper.
"""

import os
from typing import Optional
from playwright.async_api import Playwright, Browser

from logger import bot_logger

LAUNCH_ARGS = [
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _get_ws_endpoint() -> Optional[str]:
    """Build the WebSocket endpoint URL with auth token."""
    endpoint = (
        os.getenv("BROWSER_PLAYWRIGHT_ENDPOINT_PRIVATE")
        or os.getenv("BROWSER_PLAYWRIGHT_ENDPOINT")
    )
    if not endpoint:
        return None

    token = os.getenv("BROWSER_TOKEN")
    has_token_in_url = "?token=" in endpoint or "&token=" in endpoint

    if has_token_in_url:
        return endpoint
    elif token:
        sep = "&" if "?" in endpoint else "?"
        return f"{endpoint}{sep}token={token}"
    else:
        bot_logger.logger.warning(
            "BROWSER_TOKEN not set and endpoint has no embedded token — connection may fail."
        )
        return endpoint


async def connect_browser(pw: Playwright) -> Browser:
    """
    Connect to Railway Browserless if available, otherwise launch locally.
    Returns a Browser instance ready to use.
    """
    ws_endpoint = _get_ws_endpoint()

    if ws_endpoint:
        bot_logger.logger.info("Connecting to Railway Browserless...")
        browser = await pw.chromium.connect(ws_endpoint, timeout=30000)
        bot_logger.logger.info("Connected to Railway Browserless.")
        return browser
    else:
        bot_logger.logger.info("No Browserless endpoint found — launching local Chromium.")
        browser = await pw.chromium.launch(
            headless=True,
            args=LAUNCH_ARGS,
        )
        return browser
