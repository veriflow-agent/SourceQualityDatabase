# supabase_writer.py for MBFC scraping service
"""
Writes scraped MBFC data to the Supabase media_credibility table.
Uses the same schema and upsert logic as VeriFlow's SupabaseService.
"""

import os
import re
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
from supabase import create_client, Client
from logger import bot_logger

try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False


class SupabaseWriter:
    """Writes MBFC bulk scrape results to Supabase."""

    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.client: Optional[Client] = None
        self.enabled = False
        self.llm = None

        if not self.supabase_url or not self.supabase_key:
            bot_logger.logger.error("SUPABASE_URL and SUPABASE_KEY must be set.")
            return

        try:
            self.client = create_client(self.supabase_url, self.supabase_key)
            self.enabled = True
            bot_logger.logger.info("Supabase client connected.")
        except Exception as e:
            bot_logger.logger.error(f"Supabase connection failed: {e}")
            return

        if AI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            try:
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0
                ).bind(response_format={"type": "json_object"})
                bot_logger.logger.info("AI features enabled for tier assignment and name generation.")
            except Exception as e:
                bot_logger.logger.warning(f"Could not init LLM: {e}")

    # -------------------------------------------------------
    # DOMAIN DETECTION (fallbacks only - source_domain from
    # MBFCExtractedData is now the primary source)
    # -------------------------------------------------------

    def _extract_domain_from_url(self, mbfc_url: str) -> Optional[str]:
        """
        Try to guess the publication domain from the MBFC page URL slug.
        e.g. mediabiasfactcheck.com/bbc -> bbc.com
        This is a last-resort heuristic — only used when AI extraction fails.
        """
        slug = mbfc_url.rstrip("/").split("/")[-1]
        # Only return slug as domain if it already looks like one (has a dot)
        if "." in slug:
            return slug.lower()
        return None

    def _extract_domain_from_text(self, text: str) -> Optional[str]:
        """
        Find a domain mentioned in the page text.
        Looks for patterns like 'Source: example.com' or bare URLs.
        """
        skip = {"mediabiasfactcheck", "twitter", "facebook", "instagram", "youtube", "wikipedia"}
        patterns = [
            r'(?:Source|Website|URL|Homepage):\s*(https?://)?([a-zA-Z0-9\-]+\.[a-zA-Z]{2,})',
            r'(?:^|\s)(www\.[a-zA-Z0-9\-]+\.[a-zA-Z]{2,})',
            r'https?://(?:www\.)?([a-zA-Z0-9\-]+\.[a-zA-Z]{2,})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text[:3000], re.IGNORECASE | re.MULTILINE)
            if match:
                domain = match.group(match.lastindex).lower().strip()
                if not any(s in domain for s in skip):
                    return domain
        return None

    # -------------------------------------------------------
    # TIER ASSIGNMENT
    # -------------------------------------------------------

    def _rule_based_tier(self, mbfc_data: dict) -> dict:
        """Fast rule-based tier assignment — used when AI is not available."""
        factual = (mbfc_data.get("factual_reporting") or "").upper()
        credibility = (mbfc_data.get("credibility_rating") or "").upper()
        tags = [t.upper() for t in mbfc_data.get("special_tags", [])]

        if "CONSPIRACY-PSEUDOSCIENCE" in tags or "PROPAGANDA" in tags:
            return {"tier": 5, "reasoning": "Conspiracy or propaganda source"}
        if factual == "VERY LOW" or credibility == "LOW CREDIBILITY":
            return {"tier": 5, "reasoning": "Very low factual reporting or credibility"}
        if "QUESTIONABLE SOURCE" in tags:
            return {"tier": 4, "reasoning": "Questionable source flag"}
        if factual == "LOW":
            return {"tier": 4, "reasoning": "Low factual reporting"}
        if factual == "HIGH" and "HIGH" in credibility:
            return {"tier": 1, "reasoning": "High factual + high credibility"}
        if factual in ["MOSTLY FACTUAL", "HIGH"] and "LOW" not in credibility:
            return {"tier": 2, "reasoning": "Mostly factual reporting"}
        return {"tier": 3, "reasoning": "Mixed or unclear — default mid-tier"}

    TIER_PROMPT = """You assign credibility tiers (1-5) to media publications based on MBFC data.

TIER DEFINITIONS:
Tier 1: Highly reliable — scientific journals, major wire services, established quality press
Tier 2: Reliable — mainstream professional news with strong fact-checking track record
Tier 3: Mixed — general news with some bias but attempts factual reporting
Tier 4: Unreliable — questionable sources, heavy bias, frequent factual errors
Tier 5: Not credible — conspiracy sites, propaganda, very low factual reporting

SPECIAL RULES:
- QUESTIONABLE SOURCE tag -> Tier 4-5
- CONSPIRACY-PSEUDOSCIENCE tag -> Tier 5
- SATIRE sources -> Tier 3-4
- PRO-SCIENCE tag -> boost reliability
- State-affiliated media -> consider Tier 4-5

Return ONLY: {{"tier": 1-5, "reasoning": "brief explanation"}}"""

    async def _ai_tier(self, mbfc_data: dict, domain: str) -> dict:
        if not self.llm:
            return self._rule_based_tier(mbfc_data)
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.TIER_PROMPT),
                ("user", """Domain: {domain}
Bias Rating: {bias_rating}
Factual Reporting: {factual_reporting}
Credibility Rating: {credibility_rating}
Special Tags: {special_tags}
Failed Fact Checks: {failed_count}
Summary: {summary}

Return JSON: {{"tier": 1-5, "reasoning": "explanation"}}""")
            ])
            chain = prompt | self.llm
            result = await chain.ainvoke({
                "domain": domain,
                "bias_rating": mbfc_data.get("bias_rating", "Unknown"),
                "factual_reporting": mbfc_data.get("factual_reporting", "Unknown"),
                "credibility_rating": mbfc_data.get("credibility_rating", "Unknown"),
                "special_tags": mbfc_data.get("special_tags", []),
                "failed_count": len(mbfc_data.get("failed_fact_checks", [])),
                "summary": (mbfc_data.get("summary") or "")[:500],
            })
            parsed = json.loads(result.content)
            return {"tier": parsed.get("tier", 3), "reasoning": parsed.get("reasoning", "")}
        except Exception as e:
            bot_logger.logger.error(f"AI tier assignment failed: {e}")
            return self._rule_based_tier(mbfc_data)

    # -------------------------------------------------------
    # NAME GENERATION
    # -------------------------------------------------------

    NAME_PROMPT = """Generate a list of common name variations for a media publication so it can be found by different search terms.

Include: official name, acronyms, shortened names, names without articles (The, A), common misspellings if any.
Return ONLY: {{"names": ["name1", "name2", ...]}}"""

    async def _generate_names(self, domain: str, publication_name: str) -> List[str]:
        if not self.llm:
            return [publication_name] if publication_name else [domain]
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.NAME_PROMPT),
                ("user", "Domain: {domain}\nOfficial name: {name}\n\nReturn JSON with names list.")
            ])
            chain = prompt | self.llm
            result = await chain.ainvoke({"domain": domain, "name": publication_name})
            parsed = json.loads(result.content)
            names = parsed.get("names", [publication_name])
            return [n for n in names if n and isinstance(n, str)]
        except Exception as e:
            bot_logger.logger.error(f"Name generation failed: {e}")
            return [publication_name] if publication_name else [domain]

    # -------------------------------------------------------
    # WRITE TO SUPABASE
    # -------------------------------------------------------

    async def write(
        self,
        mbfc_url: str,
        extracted_data,  # MBFCExtractedData
        domain: Optional[str] = None,
        raw_page_text: Optional[str] = None
    ) -> bool:
        """
        Write a scraped MBFC record to Supabase.
        Returns True on success, False on failure.

        Domain resolution priority:
        1. source_domain from MBFCExtractedData (extracted by AI from page content)
        2. domain argument passed in explicitly
        3. Heuristic from MBFC URL slug (only if it contains a dot)
        4. Regex search in raw_page_text
        5. Last resort: MBFC slug + ".unknown"
        """
        if not self.enabled:
            bot_logger.logger.error("Supabase not enabled — cannot write.")
            return False

        # Priority 1: source_domain extracted from page content by the scraper
        if not domain and hasattr(extracted_data, 'source_domain') and extracted_data.source_domain:
            domain = extracted_data.source_domain

        # Priority 2: heuristic from MBFC URL slug (only when slug has a dot)
        if not domain:
            domain = self._extract_domain_from_url(mbfc_url)

        # Priority 3: regex scan of raw page text
        if not domain and raw_page_text:
            domain = self._extract_domain_from_text(raw_page_text)

        # Priority 4: last resort - use the MBFC slug so the record isn't lost
        if not domain:
            slug = mbfc_url.rstrip("/").split("/")[-1]
            domain = f"{slug}.unknown"
            bot_logger.logger.warning(
                f"Could not determine domain for {mbfc_url}, using: {domain}"
            )

        domain = domain.lower().strip()

        mbfc_dict = extracted_data.model_dump()

        # Assign tier
        tier_result = await self._ai_tier(mbfc_dict, domain)

        # Generate names
        names = await self._generate_names(domain, extracted_data.publication_name)

        # Build the record
        record: Dict[str, Any] = {
            "domain": domain,
            "names": names,
            "mbfc_bias_rating": extracted_data.bias_rating,
            "mbfc_bias_score": extracted_data.bias_score,
            "mbfc_factual_reporting": extracted_data.factual_reporting,
            "mbfc_factual_score": extracted_data.factual_score,
            "mbfc_credibility_rating": extracted_data.credibility_rating,
            "mbfc_country_freedom_rating": extracted_data.country_freedom_rating,
            "mbfc_url": mbfc_url,
            "mbfc_special_tags": extracted_data.special_tags,
            "country": extracted_data.country,
            "media_type": extracted_data.media_type,
            "ownership": extracted_data.ownership,
            "funding": extracted_data.funding,
            "traffic_popularity": extracted_data.traffic_popularity,
            "failed_fact_checks": extracted_data.failed_fact_checks,
            "mbfc_summary": extracted_data.summary,
            "assigned_tier": tier_result["tier"],
            "tier_reasoning": tier_result["reasoning"],
            "source": "mbfc_bulk",
            "is_verified": True,
            "last_verified_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }

        # Remove None values so we don't overwrite existing data with nulls
        record = {k: v for k, v in record.items() if v is not None}

        try:
            result = self.client.table("media_credibility") \
                .upsert(record, on_conflict="domain") \
                .execute()

            if result.data and len(result.data) > 0:
                bot_logger.logger.info(
                    f"Saved: {domain} | Tier {tier_result['tier']} | {extracted_data.bias_rating}"
                )
                return True
            else:
                bot_logger.logger.warning(
                    f"Upsert returned no data for {domain} — check RLS policies."
                )
                return False

        except Exception as e:
            bot_logger.logger.error(f"Supabase write failed for {domain}: {e}")
            return False

    def get_already_scraped(self) -> set:
        """
        Return set of mbfc_urls already in the database.
        Used to skip re-scraping pages we already have.
        """
        if not self.enabled:
            return set()
        try:
            result = self.client.table("media_credibility") \
                .select("mbfc_url") \
                .eq("source", "mbfc_bulk") \
                .execute()
            urls = {row["mbfc_url"] for row in (result.data or []) if row.get("mbfc_url")}
            bot_logger.logger.info(f"Already in DB: {len(urls)} mbfc_bulk records")
            return urls
        except Exception as e:
            bot_logger.logger.error(f"Could not fetch existing records: {e}")
            return set()