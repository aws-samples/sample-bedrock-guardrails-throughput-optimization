#!/usr/bin/env python3
"""
Guardrail optimization helpers - the three techniques from the blog, as runnable code.

  1. apply_guardrail_batched  - content-block batching (fewer API calls)
  2. GuardrailCache / apply_with_cache - cache results for repeated content
  3. apply_selective          - skip trusted content by risk tier

These are the implementations the companion blog references. They wrap the
`ApplyGuardrail` runtime API with specific error handling.

Text-unit accounting note
-------------------------
The ApplyGuardrail response has NO single ``textUnits`` field. Consumption is
reported per policy type in the ``usage`` object:
``contentPolicyUnits``, ``topicPolicyUnits``, ``wordPolicyUnits``,
``sensitiveInformationPolicyUnits``, ``contextualGroundingPolicyUnits``, etc.
``total_text_units()`` sums the billable policy-unit fields so callers get one
comparable number. (Verified against the botocore bedrock-runtime service model.)

API limits:
  Requests per second and text units per second are both per-Region quotas and
  both are adjustable. Check the current values for your account in the Service
  Quotas console. A request that is too large is rejected with
  ValidationException, not throttled.
"""
from __future__ import annotations

import hashlib
import time

from botocore.exceptions import ClientError

# Policy-unit fields in the ApplyGuardrail `usage` object that represent billable
# text-unit consumption. Summed to produce one comparable "text units" number.
_USAGE_UNIT_FIELDS = (
    "contentPolicyUnits",
    "topicPolicyUnits",
    "wordPolicyUnits",
    "sensitiveInformationPolicyUnits",
    "contextualGroundingPolicyUnits",
)

# A guardrail text unit is 1,000 characters.
TEXT_UNIT_CHARS = 1000
# Client-side guard against oversized requests. Set this to the per-request
# limit shown for your Region in Service Quotas.
MAX_TEXT_UNITS_PER_REQUEST = 1000


def total_text_units(response: dict) -> int:
    """Sum the billable policy-unit fields from an ApplyGuardrail response.

    There is no single ``textUnits`` field - this aggregates the per-policy units.
    """
    usage = response.get("usage", {}) or {}
    return sum(int(usage.get(field, 0) or 0) for field in _USAGE_UNIT_FIELDS)


def is_throttling_error(err: ClientError) -> bool:
    """True only for real throttling - not auth, validation, or not-found errors."""
    code = err.response.get("Error", {}).get("Code", "")
    return code in ("ThrottlingException", "ThrottledException", "TooManyRequestsException")


# ── 1. Content-block batching ────────────────────────────────────────────────

def apply_guardrail_batched(client, text, guardrail_id, guardrail_version="DRAFT",
                            chunk_chars=TEXT_UNIT_CHARS):
    """Chunk text into content blocks and evaluate them in ONE ApplyGuardrail call.

    Reduces RPS: N chunks cost 1 request instead of N. Raises ValueError if the
    input exceeds the documented per-request cap (the API would reject it).

    SECURITY NOTE: fixed-width chunking can split an adversarial phrase across a
    block boundary, potentially evading detection a single-block eval would catch.
    For untrusted input, prefer overlapping windows or semantic boundaries.
    """
    chunks = [text[i:i + chunk_chars] for i in range(0, len(text), chunk_chars)]
    if len(chunks) > MAX_TEXT_UNITS_PER_REQUEST:
        raise ValueError(
            f"{len(chunks)} chunks exceeds the {MAX_TEXT_UNITS_PER_REQUEST}-text-unit "
            "per-request cap; split across multiple requests."
        )
    content_blocks = [{"text": {"text": chunk}} for chunk in chunks]
    try:
        return client.apply_guardrail(
            guardrailIdentifier=guardrail_id,
            guardrailVersion=guardrail_version,
            source="INPUT",
            content=content_blocks,
        )
    except ClientError as e:
        if is_throttling_error(e):
            raise RuntimeError("ApplyGuardrail throttled - reduce RPS or request a quota increase") from e
        raise


# ── 2. Intelligent caching ───────────────────────────────────────────────────

SYSTEM_PROMPT_TTL = 86400   # 24 hours - system prompts rarely change
RAG_CONTEXT_TTL = 3600      # 1 hour - matches a typical KB refresh cycle


class GuardrailCache:
    """In-memory TTL cache for guardrail results on repeated content.

    Cache key includes the guardrail version so a config change invalidates old
    entries automatically. NEVER cache user input - it is unique and adversarial.
    Swap the dict for Redis/ElastiCache in production.
    """

    def __init__(self, clock=time.time):
        self._store: dict[str, tuple[float, dict]] = {}
        self._clock = clock
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(text: str, guardrail_version: str) -> str:
        return f"{guardrail_version}:{hashlib.sha256(text.encode()).hexdigest()}"

    def get(self, text: str, guardrail_version: str):
        entry = self._store.get(self._key(text, guardrail_version))
        if entry and entry[0] > self._clock():
            self.hits += 1
            return entry[1]
        self.misses += 1
        return None

    def put(self, text: str, guardrail_version: str, result: dict, ttl: int):
        self._store[self._key(text, guardrail_version)] = (self._clock() + ttl, result)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def apply_with_cache(client, text, content_type, guardrail_id, version,
                     cache: GuardrailCache):
    """Evaluate content, using the cache for repeatable types.

    ``content_type`` in {"user_input", "system_prompt", "rag_context"}.
    User input is always evaluated fresh (never cached).
    """
    def _call():
        return client.apply_guardrail(
            guardrailIdentifier=guardrail_id, guardrailVersion=version,
            source="INPUT", content=[{"text": {"text": text}}],
        )

    if content_type == "user_input":
        return _call()

    cached = cache.get(text, version)
    if cached is not None:
        return cached

    result = _call()
    ttl = SYSTEM_PROMPT_TTL if content_type == "system_prompt" else RAG_CONTEXT_TTL
    cache.put(text, version, result, ttl)
    return result


# ── 3. Selective application ─────────────────────────────────────────────────

def apply_selective(client, request, guardrail_id, version, cache: GuardrailCache | None = None):
    """Evaluate only content that needs fresh evaluation.

    ``request`` = {"user_input": str, "rag_contexts": [str, ...]}.
    User input always evaluated; RAG contexts skipped if already cached; system
    prompts and trusted tool output are assumed evaluated at deploy time and skipped.

    RAG contexts sent here are not written back to the cache: the batched
    response is aggregated, so it can't give a per-context verdict. Populate the
    cache for RAG contexts separately, for example with apply_with_cache at
    knowledge base ingestion time.
    """
    blocks = [{"text": {"text": request["user_input"]}}]
    for ctx in request.get("rag_contexts", []):
        if cache is None or cache.get(ctx, version) is None:
            blocks.append({"text": {"text": ctx}})

    return client.apply_guardrail(
        guardrailIdentifier=guardrail_id, guardrailVersion=version,
        source="INPUT", content=blocks,
    )
