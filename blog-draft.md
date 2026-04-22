# Optimize Amazon Bedrock Guardrails for high-throughput enterprise workloads

by Guruprasad Seeryada | AWS Enterprise Support

**Categories**: Amazon Bedrock, Amazon Bedrock Guardrails, Best Practices, Generative AI, Intermediate (200)

---

As enterprises scale their generative AI applications from proof-of-concept to production, they often encounter throughput challenges when applying safety guardrails to every request. Organizations running centralized AI gateways that serve thousands of users across hundreds of use cases need strategies to maximize guardrail throughput while staying within service quotas.

In this post, we demonstrate three techniques to optimize [Amazon Bedrock Guardrails](https://aws.amazon.com/bedrock/guardrails/) throughput for high-volume enterprise workloads: content block batching, intelligent caching, and selective application. These patterns can reduce your API calls by 60–80% while maintaining full compliance coverage.

## Understanding the throughput challenge

Enterprise AI platforms typically route all LLM traffic through a centralized gateway or proxy. When guardrails are applied to every request, two service quotas become relevant:

- **Requests per second (RPS)** — The number of `ApplyGuardrail` API calls per second. This quota is adjustable.
- **Text units per second (TUPs)** — The volume of text processed per second, where 1 text unit equals 1,000 characters. This quota has a hard limit per region.

These are independent limits with different error messages:

| Limit | Error message |
|-------|--------------|
| RPS | `ApplyGuardrail requestId: {id} got throttled for the accountId: {account}` |
| TUPs | `Too many requests sent to ApplyGuardrail: On-demand ApplyGuardrail content filter policy text units per second limit exceeded` |

A common misconception is that these are the same throttle. They require different optimization strategies, and fixing one can worsen the other if not handled carefully. For example, chunking large inputs into smaller pieces reduces text units per call (helping TUPs) but increases the number of API calls (hurting RPS).

The techniques in this post address both limits simultaneously.

## Technique 1: Content block batching

The `ApplyGuardrail` API accepts an array of content blocks in a single request. Instead of making separate API calls for each chunk of text, you can batch multiple chunks into one call.

### How it works

The API request body accepts a `content` array:

```json
{
    "source": "INPUT",
    "content": [
        {"text": {"text": "First chunk of text to evaluate..."}},
        {"text": {"text": "Second chunk of text to evaluate..."}},
        {"text": {"text": "Third chunk of text to evaluate..."}}
    ]
}
```

This consumes 1 RPS instead of 3, while the text units are summed across all blocks.

### Implementation

The following Python example demonstrates how to chunk a large input and batch the chunks into a single `ApplyGuardrail` call:

```python
import boto3

bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

CHUNK_SIZE = 1000  # 1 text unit = 1,000 characters

def chunk_text(text, chunk_size=CHUNK_SIZE):
    """Split text into chunks aligned to text unit boundaries."""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

def apply_guardrail_batched(text, guardrail_id, guardrail_version):
    """Apply guardrail to large text using batched content blocks."""
    chunks = chunk_text(text)

    # Batch all chunks into a single API call
    content_blocks = [
        {"text": {"text": chunk}} for chunk in chunks
    ]

    response = bedrock_runtime.apply_guardrail(
        guardrailIdentifier=guardrail_id,
        guardrailVersion=guardrail_version,
        source="INPUT",
        content=content_blocks
    )

    return response

# Example: Process a 5,000-character input
large_input = "..." * 5000  # Your large input text
response = apply_guardrail_batched(
    large_input,
    guardrail_id="your-guardrail-id",
    guardrail_version="1"
)

# Check the action
if response["action"] == "GUARDRAIL_INTERVENED":
    print("Content blocked:", response["outputs"][0]["text"])
else:
    print("Content passed guardrail evaluation")
```

### When to use

Use content block batching when you have large inputs that need chunking. Without batching, a 10,000-character input chunked into 10 pieces would consume 10 RPS. With batching, it consumes 1 RPS.

## Technique 2: Intelligent caching

Not every request to your AI gateway contains unique content. System prompts, RAG context from knowledge bases, and templated instructions are often repeated across requests. Caching guardrail results for these repeated inputs eliminates redundant API calls.

### Architecture

```
User Request
    │
    ├── System Prompt ──────► Cache HIT → Skip guardrail
    ├── RAG Context ────────► Cache HIT → Skip guardrail
    └── User Input (unique) ► Cache MISS → Call ApplyGuardrail
```

### Implementation

```python
import hashlib
import json
import time

class GuardrailCache:
    """Cache guardrail results to avoid redundant API calls."""

    def __init__(self, default_ttl=3600):
        self.cache = {}
        self.default_ttl = default_ttl

    def _hash_content(self, text):
        """Generate a deterministic hash for the input text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text):
        """Return cached result if valid, None otherwise."""
        key = self._hash_content(text)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() < entry["expires_at"]:
                return entry["result"]
            else:
                del self.cache[key]
        return None

    def put(self, text, result, ttl=None):
        """Cache a guardrail result."""
        key = self._hash_content(text)
        self.cache[key] = {
            "result": result,
            "expires_at": time.time() + (ttl or self.default_ttl)
        }

# Initialize cache with different TTLs by content type
cache = GuardrailCache()

SYSTEM_PROMPT_TTL = 86400   # 24 hours — system prompts rarely change
RAG_CONTEXT_TTL = 3600      # 1 hour — matches knowledge base refresh
USER_INPUT_TTL = 0          # Never cache — unique per request

def apply_guardrail_with_cache(
    text, content_type, guardrail_id, guardrail_version
):
    """Apply guardrail with caching based on content type."""

    # User input is always evaluated fresh
    if content_type == "user_input":
        return call_apply_guardrail(text, guardrail_id, guardrail_version)

    # Check cache for system prompts and RAG context
    cached = cache.get(text)
    if cached is not None:
        return cached

    # Cache miss — call the API
    result = call_apply_guardrail(text, guardrail_id, guardrail_version)

    # Cache the result with appropriate TTL
    ttl = SYSTEM_PROMPT_TTL if content_type == "system_prompt" else RAG_CONTEXT_TTL
    cache.put(text, result, ttl=ttl)

    return result

def call_apply_guardrail(text, guardrail_id, guardrail_version):
    """Make the actual ApplyGuardrail API call."""
    response = bedrock_runtime.apply_guardrail(
        guardrailIdentifier=guardrail_id,
        guardrailVersion=guardrail_version,
        source="INPUT",
        content=[{"text": {"text": text}}]
    )
    return {
        "action": response["action"],
        "assessments": response.get("assessments", [])
    }
```

### Impact

For a typical enterprise AI gateway where 60–70% of text volume comes from system prompts and repeated RAG context, caching can reduce both TUPs and RPS by more than half. Only the unique user input portion of each request requires a fresh guardrail evaluation.

## Technique 3: Selective application

Not all content in a request carries the same risk. Applying guardrails selectively based on content type and risk tier reduces unnecessary API calls while maintaining compliance coverage.

### Risk-tiered approach

Classify your use cases by risk level and apply guardrails accordingly:

| Risk tier | Example use cases | Guardrail strategy |
|-----------|------------------|-------------------|
| **High** | Customer-facing chatbots, legal, compliance | Full guardrails on input AND output |
| **Medium** | Internal tools, analytics dashboards | Guardrails on user input only |
| **Low** | Batch processing, known templates | Skip guardrails or cache results |

### Content-level selective application

Within a single request, not all content blocks need evaluation:

```python
def apply_guardrail_selective(request):
    """Apply guardrails only to content that needs evaluation."""

    content_to_evaluate = []

    # Always evaluate user input
    content_to_evaluate.append(
        {"text": {"text": request["user_input"]}}
    )

    # Evaluate RAG context only if not previously cached
    for context in request.get("rag_contexts", []):
        if cache.get(context) is None:
            content_to_evaluate.append(
                {"text": {"text": context}}
            )

    # Skip system prompts — evaluated once at deployment time
    # Skip tool outputs — generated by trusted internal systems

    if not content_to_evaluate:
        return {"action": "NONE"}

    # Batch all content needing evaluation into a single call
    response = bedrock_runtime.apply_guardrail(
        guardrailIdentifier=guardrail_id,
        guardrailVersion=guardrail_version,
        source="INPUT",
        content=content_to_evaluate
    )

    return response
```

### When to skip guardrails safely

The following content types are generally safe to exclude from per-request guardrail evaluation:

- **System prompts** — Authored by your development team, evaluated once during deployment. Cache the result.
- **Tool outputs** — Generated by trusted internal systems (databases, APIs), not user-influenced.
- **Static templates** — Fixed response formats, greetings, disclaimers.

**Important**: Always apply guardrails to user-generated input and model-generated output that will be shown to end users.

## Combining all three techniques

The following example demonstrates how to combine batching, caching, and selective application in an enterprise AI gateway:

```python
def process_request(request, guardrail_id, guardrail_version):
    """
    Enterprise AI gateway request processing with optimized guardrails.
    Combines batching, caching, and selective application.
    """

    # Step 1: Selective application — identify what needs evaluation
    blocks_to_evaluate = []

    # User input — always evaluate
    blocks_to_evaluate.append({
        "text": {"text": request["user_input"]},
        "type": "user_input"
    })

    # RAG context — evaluate only if not cached
    for ctx in request.get("rag_contexts", []):
        cached_result = cache.get(ctx)
        if cached_result is None:
            blocks_to_evaluate.append({
                "text": {"text": ctx},
                "type": "rag_context"
            })

    # System prompt — skip (cached at deployment time)
    # Tool outputs — skip (trusted internal systems)

    if not blocks_to_evaluate:
        return {"action": "NONE", "source": "all_cached"}

    # Step 2: Batch all blocks into a single API call
    content_blocks = [
        {"text": {"text": block["text"]["text"]}}
        for block in blocks_to_evaluate
    ]

    response = bedrock_runtime.apply_guardrail(
        guardrailIdentifier=guardrail_id,
        guardrailVersion=guardrail_version,
        source="INPUT",
        content=content_blocks
    )

    # Step 3: Cache results for non-user content
    for block in blocks_to_evaluate:
        if block["type"] == "rag_context":
            cache.put(
                block["text"]["text"],
                response,
                ttl=RAG_CONTEXT_TTL
            )

    return response
```

### Expected impact

The following table shows the expected reduction in API calls and text units for a typical enterprise workload:

| Technique | RPS reduction | TUPs reduction |
|-----------|:------------:|:--------------:|
| Content block batching | 60–80% | None (same text volume) |
| Intelligent caching | 50–70% | 50–70% |
| Selective application | 30–50% | 30–50% |
| **Combined** | **80–90%** | **60–80%** |

Results vary based on the ratio of repeated vs. unique content in your workload.

## Monitoring your optimization

After implementing these techniques, monitor the following CloudWatch metrics to verify the impact:

- **`InvocationThrottles`** (Sum) — Should decrease toward zero
- **`TextUnitCount`** (Sum) — Should decrease proportionally to caching and selective application
- **`Invocations`** (Sum) — Should decrease proportionally to batching and caching

Additionally, implement client-side metrics to track:

- Cache hit rate (target: >60% for enterprise workloads)
- Requests skipped by selective application
- Average content blocks per batched API call

## Best practices and considerations

When implementing these optimization techniques, keep the following in mind:

- **Start with caching** — It provides the highest impact with the lowest implementation complexity.
- **Validate cache invalidation** — Ensure cached guardrail results are invalidated when guardrail configurations change. Use guardrail version numbers as part of the cache key.
- **Don't cache user input** — User-generated content is unique and potentially adversarial. Always evaluate it fresh.
- **Test with production traffic patterns** — Optimization impact depends heavily on your specific content mix. Use [guardrails detect mode](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-harmful-content-handling-options.html) to test without blocking.
- **Monitor for false negatives** — Selective application reduces coverage by design. Regularly audit which content types are excluded and verify the risk classification remains accurate.
- **Consider multi-region distribution** — Quotas are per-region. Distributing traffic across regions provides linear scaling of both RPS and TUPs limits.

## Conclusion

Enterprise-scale generative AI applications don't have to choose between comprehensive safety guardrails and high throughput. By combining content block batching, intelligent caching, and selective application, you can reduce `ApplyGuardrail` API calls by 80–90% while maintaining full compliance coverage for user-generated content.

These techniques are particularly effective for centralized AI gateway architectures serving thousands of users across hundreds of use cases, where repeated content (system prompts, RAG context, templates) represents the majority of text volume.

To get started, implement caching for your system prompts and RAG context — this single change often delivers the largest immediate impact. Then layer in content block batching and selective application as your throughput requirements grow.

For more information about Amazon Bedrock Guardrails, see the following resources:

- [Amazon Bedrock Guardrails documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html)
- [ApplyGuardrail API reference](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ApplyGuardrail.html)
- [Build safe generative AI applications like a Pro: Best Practices with Amazon Bedrock Guardrails](https://aws.amazon.com/blogs/machine-learning/build-safe-generative-ai-applications-like-a-pro-best-practices-with-amazon-bedrock-guardrails/)
- [Use the ApplyGuardrail API with long-context inputs and streaming outputs](https://aws.amazon.com/blogs/machine-learning/use-the-applyguardrail-api-with-long-context-inputs-and-streaming-outputs-in-amazon-bedrock/)
- [Amazon Bedrock Guardrails cross-account safeguards](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-enforcements.html)

---

### About the Author

**Guruprasad Seeryada** is a Senior Technical Account Manager at AWS Enterprise Support based in Atlanta, GA. He works with enterprise customers in the financial services industry to optimize their AWS architectures for resilience, security, and performance. He is passionate about helping customers adopt generative AI responsibly at scale.
