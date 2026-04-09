"""Contextual entity extraction for policy documents.

Extracts every meaningful named thing from a document (roles, locations,
time periods, actions, rewards, thresholds, products, abbreviations, etc.)
together with what that thing *means in context*.

The enrichment text produced by DocumentEntityMap is embedded alongside the
document summary so the document vector sits close to the natural-language
queries users would ask, not just close to raw policy prose.

The flat lookup dict produced by DocumentEntityMap is used at query time so
that any user term that matches a known entity or its synonyms automatically
pulls in the entity's full contextual meaning as expanded search terms.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from policygpt.services.base import AIService


# Recognised entity categories — open enough to cover any domain.
ENTITY_CATEGORIES: frozenset[str] = frozenset({
    "role",         # FC, EIM, ACH, channel head
    "location",     # Goa, Bali, Singapore — any named place
    "time_period",  # JFM, Q4, March 2026, contest window
    "action",       # travel, lunch, meet, achieve, qualify
    "reward",       # Rolex, Samsung Galaxy, cash, trip, voucher
    "threshold",    # 85%, ₹3,49,000, ≥₹25,000 FYFP
    "product",      # Non-UL, Term, ULIP, life insurance product
    "contest",      # Bali Bliss, Power League, Dhurandhar
    "abbreviation", # FYFP, DPPM, HDFC, ACE, ESL, T&C
    "other",
})


@dataclass
class ExtractedEntity:
    name: str           # As it appears in the document
    category: str       # One of ENTITY_CATEGORIES
    context: str        # What it means in this document (1 sentence)
    synonyms: list[str] = field(default_factory=list)  # Alt phrases users would search


@dataclass
class DocumentEntityMap:
    entities: list[ExtractedEntity] = field(default_factory=list)

    def to_enrichment_text(self) -> str:
        """Compact natural-language block included in the document embedding.

        Format keeps each entity on one line so embedding tokenisation is clean:
          Category: Name (context; also: synonym1, synonym2)
        """
        if not self.entities:
            return ""

        by_category: dict[str, list[ExtractedEntity]] = {}
        for entity in self.entities:
            by_category.setdefault(entity.category, []).append(entity)

        lines: list[str] = []
        for category, items in sorted(by_category.items()):
            label = category.replace("_", " ").title()
            parts: list[str] = []
            for e in items:
                synonyms_str = ", ".join(e.synonyms[:5])
                part = f"{e.name} ({e.context}"
                if synonyms_str:
                    part += f"; also: {synonyms_str}"
                part += ")"
                parts.append(part)
            lines.append(f"{label}: {'; '.join(parts)}")

        return "\n".join(lines)

    def to_tags(self) -> list[str]:
        """Flat deduplicated tags for metadata scoring."""
        seen: set[str] = set()
        tags: list[str] = []
        for entity in self.entities:
            for term in [entity.name] + entity.synonyms:
                key = term.lower().strip()
                if key and key not in seen:
                    seen.add(key)
                    tags.append(key)
        return tags

    def to_lookup(self) -> dict[str, ExtractedEntity]:
        """Flat map of any name/synonym (lowercased) → entity.

        Used at query time: if any token from the user question matches a key
        here, the entity's context + synonyms are injected as expanded terms.
        """
        lookup: dict[str, ExtractedEntity] = {}
        for entity in self.entities:
            for term in [entity.name] + entity.synonyms:
                key = term.lower().strip()
                if key and key not in lookup:
                    lookup[key] = entity
        return lookup

    def tags_relevant_to(self, text: str) -> list[str]:
        """Return tags whose entity name appears in *text* (case-insensitive).

        Used to annotate sections with only the entities actually mentioned
        in that section, avoiding tag pollution across the whole document.
        Entity categories that apply document-wide (roles, abbreviations,
        contests) are always included regardless of mention.
        """
        global_categories = {"role", "abbreviation", "contest", "time_period"}
        text_lower = text.lower()
        tags: list[str] = []
        seen: set[str] = set()
        for entity in self.entities:
            is_global = entity.category in global_categories
            is_mentioned = entity.name.lower() in text_lower or any(
                s.lower() in text_lower for s in entity.synonyms
            )
            if is_global or is_mentioned:
                for term in [entity.name] + entity.synonyms:
                    key = term.lower().strip()
                    if key and key not in seen:
                        seen.add(key)
                        tags.append(key)
        return tags


class EntityExtractor:
    """LLM-based contextual entity extractor — works for any domain/document."""

    def __init__(self, ai: AIService, domain_context: str = "") -> None:
        self.ai = ai
        self.domain_context = domain_context.strip()

    def _system_prompt(self) -> str:
        base = (
            "You are a structured entity extraction specialist. "
            "Your job is to identify every meaningful named entity in a document "
            "and describe what each entity means *in the context of that document*. "
            "Return only a valid JSON array — no markdown, no explanation."
        )
        if self.domain_context:
            return f"Domain context: {self.domain_context}\n{base}"
        return base

    def extract(
        self,
        title: str,
        masked_text: str,
        max_output_tokens: int = 1000,
        char_budget: int = 6000,
    ) -> DocumentEntityMap:
        excerpt = masked_text[:char_budget].strip()
        user_prompt = self._build_prompt(title, excerpt)
        try:
            raw = self.ai.llm_text(
                system_prompt=self._system_prompt(),
                user_prompt=user_prompt,
                max_output_tokens=max_output_tokens,
            )
            return self._parse(raw)
        except Exception:
            return DocumentEntityMap()

    @classmethod
    def _build_prompt(cls, title: str, excerpt: str) -> str:
        categories = ", ".join(sorted(ENTITY_CATEGORIES))
        return (
            f"Document title: {title}\n\n"
            f"{excerpt}\n\n"
            "---\n"
            "Extract ALL meaningful named entities from the document above.\n\n"
            f"Entity categories: {categories}\n\n"
            "For every entity return a JSON object with exactly these keys:\n"
            '  "name"     — the entity as it appears in the document\n'
            '  "category" — one of the categories listed above\n'
            '  "context"  — one sentence: what this entity IS and what role it plays in this document\n'
            '  "synonyms" — list of alternative phrases or terms a user might type when asking about it\n\n'
            "Rules:\n"
            "- Extract ALL meaningful entities: roles, locations, time periods, actions, "
            "rewards, thresholds, products, contest names, and abbreviations.\n"
            "- For roles: describe the role's level, responsibilities, and which contests they qualify for.\n"
            "- For locations: describe whether domestic or international, the country, "
            "and what the location represents (travel reward, event venue, etc.).\n"
            "- For rewards: describe the reward type (cash, travel, voucher, product) and its value if stated.\n"
            "- For thresholds: describe what the number means (eligibility criterion, "
            "minimum FYFP, persistency rate, slab amount) and who it applies to.\n"
            "- For time periods: describe the window, what it covers, and any deadlines within it.\n"
            "- For abbreviations: spell them out fully in context.\n"
            "- For contest names: describe the contest purpose and who can participate.\n"
            "- Synonyms must reflect how a non-expert sales agent would phrase a question "
            "(natural, conversational English — the way someone types in a chat).\n"
            "- Be exhaustive: include every entity a user in this domain might ask about.\n\n"
            "Example output (do not copy these values — extract from the document above):\n"
            '[\n'
            '  {"name":"Goa","category":"location","context":"Domestic travel reward destination '
            'in India offered to qualifying agents","synonyms":["goa trip","india trip",'
            '"domestic travel","goa event"]},\n'
            '  {"name":"FYFP","category":"abbreviation","context":"First Year First Premium — '
            'the total new business premium collected in the first policy year, used as the '
            'contest performance metric","synonyms":["first year premium","first year first premium",'
            '"new premium","production"]}\n'
            "]\n\n"
            "Output ONLY the JSON array. No preamble, no explanation."
        )

    @staticmethod
    def _parse(raw: str) -> DocumentEntityMap:
        """Parse LLM output, tolerating minor formatting issues."""
        # Extract the first JSON array found in the response.
        match = re.search(r"\[[\s\S]*\]", raw.strip())
        if not match:
            return DocumentEntityMap()
        try:
            items: list[Any] = json.loads(match.group())
        except json.JSONDecodeError:
            return DocumentEntityMap()

        entities: list[ExtractedEntity] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            context = str(item.get("context") or "").strip()
            if not name or not context:
                continue
            raw_cat = str(item.get("category") or "other").strip().lower()
            category = raw_cat if raw_cat in ENTITY_CATEGORIES else "other"
            raw_syns = item.get("synonyms") or []
            synonyms = (
                [str(s).strip() for s in raw_syns if str(s).strip()]
                if isinstance(raw_syns, list)
                else []
            )
            entities.append(ExtractedEntity(
                name=name,
                category=category,
                context=context,
                synonyms=synonyms,
            ))

        return DocumentEntityMap(entities=entities)
