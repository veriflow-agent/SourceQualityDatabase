# mbfc_scraper.py for MBFC scraping service
"""
MBFC page scraper with aggressive ad blocking.
Adapted from VeriFlow's mbfc_scraper.py.
"""

import asyncio
import json
import re
import os
from typing import Optional, List
from pydantic import BaseModel, Field
from playwright.async_api import Browser, Page

from logger import bot_logger

try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    LLM_AVAILABLE = True
    bot_logger.logger.info("LangChain available - using AI extraction")
except ImportError:
    LLM_AVAILABLE = False
    bot_logger.logger.warning("LangChain not available - falling back to regex extraction")


class MBFCExtractedData(BaseModel):
    """Structured data extracted from an MBFC page."""
    publication_name: str = Field(description="Name of the publication")
    source_domain: Optional[str] = Field(
        default=None,
        description="The publication's actual website domain, e.g. cnn.com or bbc.co.uk"
    )
    bias_rating: Optional[str] = Field(default=None)
    bias_score: Optional[float] = Field(default=None)
    factual_reporting: Optional[str] = Field(default=None)
    factual_score: Optional[float] = Field(default=None)
    credibility_rating: Optional[str] = Field(default=None)
    country: Optional[str] = Field(default=None)
    country_freedom_rating: Optional[str] = Field(default=None)
    media_type: Optional[str] = Field(default=None)
    traffic_popularity: Optional[str] = Field(default=None)
    ownership: Optional[str] = Field(default=None)
    funding: Optional[str] = Field(default=None)
    failed_fact_checks: list = Field(default_factory=list)
    summary: Optional[str] = Field(default=None)
    special_tags: list = Field(default_factory=list)


BLOCKED_DOMAINS = [
    "googlesyndication.com", "googleadservices.com", "doubleclick.net",
    "google-analytics.com", "googletagmanager.com", "googletagservices.com",
    "pagead2.googlesyndication.com", "adngin.com", "adservice.google.com",
    "adsystem.com", "advertising.com", "adform.net", "adnxs.com",
    "adsrvr.org", "amazon-adsystem.com", "criteo.com", "outbrain.com",
    "taboola.com", "pubmatic.com", "rubiconproject.com", "openx.net",
    "casalemedia.com", "contextweb.com", "spotxchange.com", "tremorhub.com",
    "facebook.net", "connect.facebook.net", "chartbeat.com", "quantserve.com",
    "scorecardresearch.com", "newrelic.com", "nr-data.net", "segment.io",
    "segment.com", "mixpanel.com", "hotjar.com", "fullstory.com",
    "optinmonster.com", "sumo.com", "mailchimp.com", "klaviyo.com",
    "privy.com", "justuno.com", "jetpack.wordpress.com",
]

MBFC_EXTRACTION_PROMPT = """You are an expert at extracting structured data from Media Bias/Fact Check (MBFC) pages.

Given the raw text content from an MBFC page, extract all relevant information.

IMPORTANT GUIDELINES:
1. Extract EXACT values as they appear (e.g., "LEFT-CENTER", "HIGH", "MOSTLY FREE")
2. For bias_score, look for numbers in parentheses like "(-3.4)" and extract as float
3. For factual_score, look for numbers like "(1.0)" near factual reporting
4. failed_fact_checks should be a list - if "None in the Last 5 years", return empty list []
5. special_tags: include labels like "Questionable Source", "Conspiracy-Pseudoscience", "Satire", etc.
6. If a field is not found, use null
7. For source_domain: Look for the publication's ACTUAL website URL anywhere in the text.
   MBFC pages contain a clickable link to the publication's own website - it often appears as a
   URL like "https://cnn.com", "https://www.theguardian.com", etc., or labeled "Source:", "Website:".
   Extract ONLY the bare domain (e.g. "cnn.com", "bbc.co.uk", "theguardian.com").
   Strip "www.", "https://", "http://". Do NOT return "mediabiasfactcheck.com".
   If you cannot find a clear publication website URL, return null.

RAW PAGE CONTENT:
{page_content}

Respond with ONLY valid JSON matching this structure:
{{
    "publication_name": "string",
    "source_domain": "string or null - the publication's own website domain e.g. cnn.com",
    "bias_rating": "string or null",
    "bias_score": "number or null",
    "factual_reporting": "string or null",
    "factual_score": "number or null",
    "credibility_rating": "string or null",
    "country": "string or null",
    "country_freedom_rating": "string or null",
    "media_type": "string or null",
    "traffic_popularity": "string or null",
    "ownership": "string or null",
    "funding": "string or null",
    "failed_fact_checks": ["list of strings"],
    "summary": "string or null",
    "special_tags": ["list of strings"]
}}"""

# Domains to skip when hunting for source_domain in page text
_SKIP_DOMAINS = {
    "mediabiasfactcheck.com", "facebook.com", "twitter.com", "x.com",
    "instagram.com", "youtube.com", "wikipedia.org", "google.com",
    "amazon.com", "apple.com", "linkedin.com", "wordpress.com",
}


class MBFCScraper:
    """Scrapes individual MBFC source pages with ad blocking and AI extraction."""

    def __init__(self):
        self.session_cookie = os.getenv("MBFC_SESSION_COOKIE")
        self.llm = None

        if LLM_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            try:
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0
                ).bind(response_format={"type": "json_object"})
                bot_logger.logger.info("MBFC Scraper: AI extraction enabled (gpt-4o-mini)")
            except Exception as e:
                bot_logger.logger.warning(f"MBFC Scraper: Could not init LLM: {e}")

        if self.session_cookie:
            bot_logger.logger.info("MBFC Scraper: Ad-free session cookie configured")

    async def _should_block_request(self, route) -> bool:
        url = route.request.url.lower()
        resource_type = route.request.resource_type

        if resource_type in ["image", "media", "font"]:
            return True

        for domain in BLOCKED_DOMAINS:
            if domain in url:
                return True

        blocked_patterns = [
            "/ads/", "/ad/", "/advert", "pagead", "adsense",
            "adserver", "tracking", "analytics", "popup", "modal",
            ".gif", ".png", ".jpg", ".jpeg", ".webp",
        ]
        for pattern in blocked_patterns:
            if pattern in url:
                return True

        return False

    async def _setup_page(self, page: Page):
        """Configure page with ad/popup blocking before navigation."""

        async def handle_route(route):
            if await self._should_block_request(route):
                await route.abort()
            else:
                await route.continue_()

        await page.route("**/*", handle_route)

        await page.set_extra_http_headers({
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        })

        await page.add_init_script("""
            window.alert = () => {};
            window.confirm = () => true;
            window.prompt = () => '';

            const removeBlockingElements = () => {
                const selectors = [
                    '[class*="popup"]', '[class*="modal"]', '[class*="overlay"]',
                    '[id*="popup"]', '[id*="modal"]', '[id*="overlay"]',
                    '[class*="cookie"]', '[id*="cookie"]', '[class*="consent"]',
                    '[class*="newsletter"]', '[class*="subscribe"]',
                    '.adsbygoogle', '[class*="adngin"]', '[id*="adngin"]',
                    'ins.adsbygoogle', '[data-ad-slot]',
                ];
                selectors.forEach(sel => {
                    try {
                        document.querySelectorAll(sel).forEach(el => {
                            if (!el.closest('article') && !el.closest('.entry-content')) {
                                el.remove();
                            }
                        });
                    } catch(e) {}
                });
                document.body.style.overflow = 'auto';
                document.documentElement.style.overflow = 'auto';
            };

            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', removeBlockingElements);
            } else {
                removeBlockingElements();
            }
            setInterval(removeBlockingElements, 1000);
            const observer = new MutationObserver(removeBlockingElements);
            observer.observe(document.body, { childList: true, subtree: true });
        """)

    async def _cleanup_page(self, page: Page):
        try:
            await page.evaluate("""
                () => {
                    const adSelectors = [
                        '.adsbygoogle', '[class*="adngin"]', '[id*="adngin"]',
                        'ins.adsbygoogle', '[data-ad-slot]',
                    ];
                    adSelectors.forEach(sel => {
                        document.querySelectorAll(sel).forEach(el => el.remove());
                    });
                    document.querySelectorAll('*').forEach(el => {
                        const style = window.getComputedStyle(el);
                        if (style.position === 'fixed' && style.zIndex > 100) {
                            const rect = el.getBoundingClientRect();
                            if (rect.width > window.innerWidth * 0.5 || rect.height > window.innerHeight * 0.5) {
                                el.remove();
                            }
                        }
                    });
                    document.body.style.overflow = 'auto';
                    document.documentElement.style.overflow = 'auto';
                }
            """)
        except Exception:
            pass

    async def _wait_for_content(self, page: Page):
        """Wait for MBFC article content to appear."""
        for selector in ["article", ".entry-content"]:
            try:
                await page.wait_for_selector(selector, timeout=8000)
                break
            except Exception:
                pass
        try:
            await page.wait_for_selector("text=Bias Rating", timeout=5000)
        except Exception:
            pass
        await asyncio.sleep(1)
        try:
            await page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            pass

    async def _get_visible_text(self, page: Page) -> str:
        """Extract the main article text from the page."""
        try:
            text = await page.evaluate("""
                () => {
                    const selectors = [
                        'article',
                        '.entry-content.clearfix',
                        '.entry-content',
                        '#main-content',
                        '[role="main"]'
                    ];
                    for (const sel of selectors) {
                        const el = document.querySelector(sel);
                        if (el && el.innerText && el.innerText.length > 500) {
                            return el.innerText;
                        }
                    }
                    return document.body.innerText;
                }
            """)
            return self._clean_text(text) if text else ""
        except Exception as e:
            bot_logger.logger.error(f"Text extraction error: {e}")
            return ""

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        for pattern in [r'Advertisement\s*\n', r'Skip to content\s*\n', r'Search for:.*?\n']:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return text.strip()

    def _extract_source_domain_from_text(self, text: str) -> Optional[str]:
        """
        Regex fallback: find the publication's own website domain in page text.
        Tries labeled patterns first (most reliable), then bare URLs.
        """
        # Pattern 1: explicitly labeled - "Source: https://cnn.com", "Website: example.com"
        labeled = re.search(
            r'(?:Source|Website|URL|Homepage|Visit):\s*https?://(?:www\.)?'
            r'([a-zA-Z0-9][a-zA-Z0-9\-]*(?:\.[a-zA-Z0-9][a-zA-Z0-9\-]*)+)',
            text, re.IGNORECASE
        )
        if labeled:
            candidate = labeled.group(1).lower()
            if not any(skip in candidate for skip in _SKIP_DOMAINS):
                return candidate

        # Pattern 2: any https:// URL in the text that isn't a skip domain
        for match in re.finditer(
            r'https?://(?:www\.)?([a-zA-Z0-9][a-zA-Z0-9\-]*(?:\.[a-zA-Z]{2,}))(?:/|$|\s)',
            text
        ):
            candidate = match.group(1).lower()
            if not any(skip in candidate for skip in _SKIP_DOMAINS):
                return candidate

        return None

    async def _extract_with_ai(self, page_content: str) -> Optional[MBFCExtractedData]:
        if not self.llm:
            return self._extract_with_regex(page_content)
        try:
            content = page_content[:8000] if len(page_content) > 8000 else page_content
            prompt = ChatPromptTemplate.from_messages([("user", MBFC_EXTRACTION_PROMPT)])
            chain = prompt | self.llm
            response = await chain.ainvoke({"page_content": content})

            raw = response.content
            data = json.loads(raw if isinstance(raw, str) else str(raw))
            data.setdefault("failed_fact_checks", [])
            data.setdefault("special_tags", [])

            # If AI didn't find source_domain, try regex as backup
            if not data.get("source_domain"):
                data["source_domain"] = self._extract_source_domain_from_text(page_content)

            return MBFCExtractedData(**data)
        except Exception as e:
            bot_logger.logger.error(f"AI extraction failed: {e}")
            return self._extract_with_regex(page_content)

    def _extract_with_regex(self, page_content: str) -> Optional[MBFCExtractedData]:
        try:
            data = {}

            title_match = re.search(r'^([^\n]+?)(?:\s*[-]\s*Bias)', page_content, re.MULTILINE)
            if title_match:
                data["publication_name"] = title_match.group(1).strip()
            else:
                name_match = re.search(r'Overall,?\s+we\s+rate\s+([^,]+)', page_content, re.IGNORECASE)
                data["publication_name"] = name_match.group(1).strip() if name_match else "Unknown"

            # Source domain - the publication's real website
            data["source_domain"] = self._extract_source_domain_from_text(page_content)

            bias_match = re.search(r'Bias Rating:\s*([A-Z\-]+(?:\s+[A-Z\-]+)?)\s*\(?([\-\d.]+)?\)?', page_content, re.IGNORECASE)
            if bias_match:
                data["bias_rating"] = bias_match.group(1).strip()
                if bias_match.group(2):
                    try:
                        data["bias_score"] = float(bias_match.group(2))
                    except ValueError:
                        pass

            factual_match = re.search(r'Factual Reporting:\s*([A-Z\s]+)\s*\(?([\d.]+)?\)?', page_content, re.IGNORECASE)
            if factual_match:
                data["factual_reporting"] = factual_match.group(1).strip()
                if factual_match.group(2):
                    try:
                        data["factual_score"] = float(factual_match.group(2))
                    except ValueError:
                        pass

            cred_match = re.search(r'MBFC Credibility Rating:\s*([A-Z\s]+)', page_content, re.IGNORECASE)
            if cred_match:
                data["credibility_rating"] = cred_match.group(1).strip()

            country_match = re.search(r'Country:\s*([A-Za-z\s]+?)(?:\n|MBFC)', page_content)
            if country_match:
                data["country"] = country_match.group(1).strip()

            freedom_match = re.search(r'Country Freedom Rating:\s*([A-Z\s]+)', page_content, re.IGNORECASE)
            if freedom_match:
                data["country_freedom_rating"] = freedom_match.group(1).strip()

            media_match = re.search(r'Media Type:\s*([A-Za-z\s]+?)(?:\n|Traffic)', page_content)
            if media_match:
                data["media_type"] = media_match.group(1).strip()

            traffic_match = re.search(r'Traffic/Popularity:\s*([A-Za-z\s]+?)(?:\n|MBFC)', page_content)
            if traffic_match:
                data["traffic_popularity"] = traffic_match.group(1).strip()

            data["failed_fact_checks"] = []

            special_tags = []
            for tag in ["Questionable Source", "Conspiracy-Pseudoscience", "Satire", "Pro-Science", "Propaganda"]:
                if re.search(tag, page_content, re.IGNORECASE):
                    special_tags.append(tag)
            data["special_tags"] = special_tags

            if data.get("publication_name") and (data.get("bias_rating") or data.get("factual_reporting")):
                return MBFCExtractedData(**data)

            return None

        except Exception as e:
            bot_logger.logger.error(f"Regex extraction failed: {e}")
            return None

    async def scrape_page(self, page: Page, url: str) -> Optional[MBFCExtractedData]:
        """
        Scrape a single MBFC source page.
        The Page object should not yet have navigated.
        """
        try:
            await self._setup_page(page)
            await page.goto(url, wait_until="domcontentloaded", timeout=20000)
            await self._wait_for_content(page)
            await self._cleanup_page(page)

            text = await self._get_visible_text(page)
            if not text or len(text) < 200:
                bot_logger.logger.warning(f"Insufficient text on {url} ({len(text)} chars)")
                return None

            extracted = await self._extract_with_ai(text)
            if extracted:
                domain_info = f" ({extracted.source_domain})" if extracted.source_domain else " (domain not found in page)"
                bot_logger.logger.info(
                    f"Scraped: {extracted.publication_name}{domain_info} "
                    f"| Bias: {extracted.bias_rating} | Factual: {extracted.factual_reporting}"
                )
            return extracted

        except Exception as e:
            bot_logger.logger.error(f"Error scraping {url}: {e}")
            return None