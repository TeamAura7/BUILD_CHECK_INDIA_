"""
backend/rase/extractor.py
==========================
LLM-assisted drafting of RuntimeRuleDefinition records from retrieved
regulation clauses (RAG -> RASE).

Hard boundary (see ARCHITECTURE.md): the LLM never sets ComplianceStatus,
and nothing it drafts here becomes an authoritative RuntimeRuleDefinition
automatically. Every draft is:
  - tagged status="DRAFT"
  - forced to cite the retrieved chunk_id(s) it was built from
  - written to data/runtime_rules/<municipality>/drafts.json, a SEPARATE
    file from the live rules.json the RuleEngine reads
  - only moved into rules.json via the explicit promote_draft() step,
    which a human calls after reviewing the citation

If Groq isn't configured (no API key) or retrieval finds too few chunks
(< settings.rase_min_retrieval_chunks), drafting is refused rather than
hallucinating a threshold with no textual grounding.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from backend.config import Settings, get_settings
from backend.rag.retrieval.hybrid_retriever import hybrid_search
from backend.rase.schema import parse_condition
from backend.runtime_rules.contracts import RuntimeRuleDefinition
from backend.tools.logging_config import get_logger

logger = get_logger(__name__)


class RuleDraft(BaseModel):
    """A RuntimeRuleDefinition-shaped draft, plus the RAG provenance that
    produced it. Never consumed directly by the RuleEngine — see promote_draft."""

    rule: RuntimeRuleDefinition
    status: str = "DRAFT"
    source_chunk_ids: list[str] = Field(default_factory=list)
    drafted_from_query: str = ""
    reviewer_notes: Optional[str] = None


#: Default one-query-per-field set for bootstrapping a municipality's
#: ruleset. Used by draft_all_fields() below and by
#: backend.tools.run_full_compliance's --municipality flow. Kept here
#: (not duplicated per-caller) so there is one place to add/rename fields.
DEFAULT_FIELD_QUERIES: dict[str, str] = {
    "plot.width": "minimum plot width requirement residential building",
    "plot.depth": "minimum plot depth requirement residential building",
    "plot.area": "minimum plot area requirement residential building",
    "building.width": "minimum building width requirement residential building",
    "building.depth": "minimum building depth requirement residential building",
    "building.footprint_area": "maximum ground floor built up footprint area",
    "road.width": "minimum road width requirement for residential building access",
    "setbacks.front": "minimum front setback residential building",
    "setbacks.rear": "minimum rear setback residential building",
    "setbacks.left": "minimum left side setback residential building",
    "setbacks.right": "minimum right side setback residential building",
    "coverage": "maximum ground coverage percentage residential building",
    "far": "maximum floor area ratio FAR residential building",
}


_DRAFT_SYSTEM_PROMPT = """You are drafting a structured building-byelaw rule from Indian \
municipal regulation text. You extract, you do not invent. If the retrieved text does not \
state a concrete numeric threshold, you must refuse.

Respond with ONLY a single JSON object (no markdown fences, no preamble), matching exactly:
{
  "rule_id": "<short unique slug, e.g. bbmp-front-setback-residential>",
  "citation": "<clause/section reference exactly as it appears in the text>",
  "description": "<one-line human description of the rule>",
  "applies_when": <Condition JSON: {} for always, or {"all":[...]}/{"any":[...]}/{"field":...}>,
  "threshold": {"field": "<one of: plot.width, plot.depth, plot.area, building.width, \
building.depth, building.footprint_area, building.floor_count, road.width, setbacks.front, \
setbacks.rear, setbacks.left, setbacks.right, coverage, far>", "op": "<one of: ==, !=, <, <=, \
>, >=, in, between>", "value": <number or [low, high] for between>, "unit": "<m|sq_m|%|ratio>"}
}

If the retrieved text does not contain a clear numeric threshold for any of the allowed \
fields above, respond with exactly: {"refused": true, "reason": "<why>"}
"""


def _build_user_prompt(query: str, chunks: list[dict[str, Any]]) -> str:
    parts = [f'Draft a rule for: "{query}"\n', "Retrieved regulation text:\n"]
    for c in chunks:
        parts.append(f"[chunk_id={c.get('chunk_id')}] ({c.get('clause_ref', '')})")
        parts.append(c.get("text", ""))
        parts.append("")
    return "\n".join(parts)


def _extract_json(raw: str) -> dict[str, Any]:
    """
    Parse the model's response as a single JSON object.

    Reasoning-capable models (e.g. Groq's qwen3.x family) can prepend
    chain-of-thought text or <think>...</think> blocks even when asked for
    JSON-only output, and some wrap the answer in markdown fences. This
    strips fences first, then falls back to extracting the outermost
    {...} span so a stray preamble doesn't turn a perfectly good draft
    into a silent "could not draft" result.
    """
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Fall back: grab the outermost {...} span and try again. This
    # tolerates a reasoning preamble (including unterminated <think>
    # blocks) that precedes the actual JSON object.
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("no JSON object found in model output", raw, 0)
    return json.loads(raw[start : end + 1])


def _call_groq(system_prompt: str, user_prompt: str, settings: Settings) -> str:
    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set; RASE drafting requires a configured LLM. "
            "Set it in .env, or author RuntimeRuleDefinition JSON by hand under "
            "data/runtime_rules/<municipality>/rules.json."
        )
    from groq import Groq

    client = Groq(api_key=settings.groq_api_key)
    kwargs: dict[str, Any] = dict(
        model=settings.groq_model_name,
        temperature=settings.rase_draft_temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # JSON mode forces the model to emit a single valid JSON object as
        # `content` — critical for reasoning-capable models (qwen3.x,
        # gpt-oss, ...) which otherwise interleave chain-of-thought text
        # with the answer even when the prompt asks for JSON only.
        response_format={"type": "json_object"},
    )
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:
        # Some models/older API versions reject response_format for chat
        # completions. Retry once without it rather than failing outright —
        # _extract_json's fallback span-parse still has a shot at the result.
        logger.warning(
            "Groq call with response_format=json_object failed (%s); retrying without it.",
            exc,
        )
        kwargs.pop("response_format", None)
        response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def draft_rule(
    query: str,
    municipality: str,
    settings: Settings | None = None,
) -> Optional[RuleDraft]:
    """
    Retrieve regulation clauses for `query` (via the municipality's hybrid
    RAG index) and ask Groq to draft a single RuntimeRuleDefinition from
    them.

    Returns None if drafting was refused (too little grounding, or the
    model itself declined because no numeric threshold was present).
    Raises if Groq isn't configured at all.
    """
    settings = settings or get_settings()
    municipality = municipality.upper()

    chunks = hybrid_search(query, municipality, settings=settings)
    if len(chunks) < settings.rase_min_retrieval_chunks:
        logger.warning(
            "RASE: only %d chunks retrieved for query=%r (municipality=%s); "
            "refusing to draft (min=%d).",
            len(chunks),
            query,
            municipality,
            settings.rase_min_retrieval_chunks,
        )
        return None

    user_prompt = _build_user_prompt(query, chunks)
    raw = _call_groq(_DRAFT_SYSTEM_PROMPT, user_prompt, settings)

    try:
        payload = _extract_json(raw)
    except json.JSONDecodeError:
        logger.error("RASE: model response was not valid JSON: %r", raw[:300])
        return None

    if payload.get("refused"):
        logger.info("RASE: model refused to draft a rule: %s", payload.get("reason"))
        return None

    # Validate the condition shapes eagerly so a malformed draft fails here,
    # not later inside the deterministic engine.
    parse_condition(payload.get("applies_when") or {}) if payload.get("applies_when") else None
    parse_condition(payload["threshold"])

    rule = RuntimeRuleDefinition(
        rule_id=payload["rule_id"],
        municipality=municipality,
        citation=payload.get("citation"),
        description=payload["description"],
        applies_when=payload.get("applies_when") or {},
        threshold=payload["threshold"],
        version="0.0.0-draft",
    )

    return RuleDraft(
        rule=rule,
        source_chunk_ids=[c.get("chunk_id", "") for c in chunks],
        drafted_from_query=query,
    )


def draft_all_fields(
    municipality: str,
    field_queries: dict[str, str] | None = None,
    settings: Settings | None = None,
) -> dict[str, Optional[RuleDraft]]:
    """
    One-time bootstrap: draft one rule per field (default
    DEFAULT_FIELD_QUERIES) for a municipality's already-ingested corpus,
    saving every successful draft to drafts.json for review.

    Call this ONCE per municipality after ingestion (~1 Groq call per
    field). It is deliberately NOT something a per-plan compliance check
    should call — after you promote_draft() the ones you accept,
    backend.compliance.engine.JsonFileRuleEngine evaluates any number of
    plans against the promoted ruleset with ZERO further LLM calls.
    """
    settings = settings or get_settings()
    field_queries = field_queries or DEFAULT_FIELD_QUERIES

    results: dict[str, Optional[RuleDraft]] = {}
    for field, query in field_queries.items():
        try:
            draft = draft_rule(query, municipality, settings=settings)
        except Exception as exc:
            logger.error("draft_all_fields: %s failed: %s", field, exc)
            draft = None
        if draft is not None:
            save_draft(draft, settings=settings)
        results[field] = draft
    return results


# ─── Draft storage ───────────────────────────────────────────────────────────


def save_draft(draft: RuleDraft, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    path = settings.runtime_rule_drafts_path(draft.rule.municipality)
    path.parent.mkdir(parents=True, exist_ok=True)

    drafts: list[dict[str, Any]] = []
    if path.exists():
        drafts = json.loads(path.read_text(encoding="utf-8"))

    drafts = [d for d in drafts if d["rule"]["rule_id"] != draft.rule.rule_id]
    drafts.append(json.loads(draft.model_dump_json()))

    path.write_text(json.dumps(drafts, indent=2), encoding="utf-8")
    logger.info("Saved draft rule %s for %s -> %s", draft.rule.rule_id, draft.rule.municipality, path)


def load_drafts(municipality: str, settings: Settings | None = None) -> list[RuleDraft]:
    settings = settings or get_settings()
    path = settings.runtime_rule_drafts_path(municipality)
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [RuleDraft.model_validate(d) for d in raw]


def promote_draft(
    rule_id: str,
    municipality: str,
    reviewer_notes: Optional[str] = None,
    settings: Settings | None = None,
) -> RuntimeRuleDefinition:
    """
    Explicit human-in-the-loop step: move one drafted rule from
    drafts.json into the live rules.json the deterministic RuleEngine
    reads. This is the ONLY way a rule becomes authoritative — nothing
    upstream of this call does it automatically.
    """
    settings = settings or get_settings()
    municipality = municipality.upper()

    drafts = load_drafts(municipality, settings=settings)
    match = next((d for d in drafts if d.rule.rule_id == rule_id), None)
    if match is None:
        raise ValueError(f"No draft rule_id={rule_id!r} found for {municipality}")

    live_path = settings.runtime_rules_path(municipality)
    live_path.parent.mkdir(parents=True, exist_ok=True)
    live_rules: list[dict[str, Any]] = []
    if live_path.exists():
        live_rules = json.loads(live_path.read_text(encoding="utf-8"))

    promoted = match.rule.model_copy(update={"version": "1.0.0"})
    live_rules = [r for r in live_rules if r["rule_id"] != promoted.rule_id]
    live_rules.append(json.loads(promoted.model_dump_json()))
    live_path.write_text(json.dumps(live_rules, indent=2), encoding="utf-8")

    logger.info(
        "Promoted rule %s for %s from draft to live ruleset (%s)",
        rule_id,
        municipality,
        live_path,
    )
    return promoted
