# Optimize Amazon Bedrock Guardrails for high-throughput workloads

by Guruprasad Seeryada | AWS Enterprise Support

**Categories**: Amazon Bedrock, Amazon Bedrock Guardrails, Best Practices, Generative AI, Intermediate (200)

---

## Introduction

Guardrail throughput becomes a bottleneck fast. Your centralized AI gateway serves thousands of users. Every request hits the [Amazon Bedrock Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html) API. Quotas push back.

This post shows you three techniques to fix that: content block batching, intelligent caching, and selective application. Combined, they cut API calls by 80–90% while keeping full compliance coverage. We tested these patterns and confirmed a 190x speed improvement for batched calls.

You'll walk away with an architecture pattern, working code in a [companion GitHub repo](https://github.com/svguruprasad/guardrails-batch-test), and a monitoring strategy you can deploy today.

## Solution overview

Enterprise AI platforms route LLM traffic through a centralized gateway. When you apply guardrails to every request, two [service quotas](https://console.aws.amazon.com/servicequotas/home/services/bedrock/quotas) matter:

**Requests per second (RPS)** — The number of `ApplyGuardrail` API calls per second. This quota is adjustable.

**Text units per second (TUPs)** — The volume of text processed per second. One text unit equals 1,000 characters. This quota has a hard limit per region.

These are independent limits with different error messages. RPS throttles return a request-level throttle message. TUPs throttles return a text-units-per-second limit message. A common misconception: these are the same throttle. They aren't. Fixing one can worsen the other. Chunking large inputs into smaller pieces reduces text units per call — but increases the number of API calls. You need strategies that address both limits at once.

Here's the optimized request flow:

```
User Request → AI Gateway → [Cache Check] → [Selective Filter] → [Batch Content Blocks] → ApplyGuardrail API → Response
```

Each step reduces what reaches the API. The cache check eliminates repeated content — system prompts you've already evaluated, RAG context you've seen before. The selective filter skips content that doesn't need evaluation, like tool outputs from trusted internal systems. Batching combines everything that's left into a single API call. Each technique stacks on the others, and the order matters: cache first (cheapest check), then filter, then batch what remains.

## Content block batching

Content block batching sends multiple text chunks in one API call instead of many separate calls. This directly reduces your RPS consumption.

Consider a 10,000-character document that needs chunking. Without batching, that's 10 chunks at 1,000 characters each — 10 separate `ApplyGuardrail` calls. At 500 requests per second through your gateway, you'd need 5,000 RPS of quota. Most accounts start well below that. The [ApplyGuardrail API](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ApplyGuardrail.html) accepts an array of content blocks in a single request. Ten chunks in one call consumes 1 RPS, not 10. Each block evaluates independently, so a violation in one block doesn't affect the others.

In practice, this means you split your text into chunks aligned to text unit boundaries (1,000 characters each), wrap each chunk as a content block, and send the array in one call. The response contains per-block assessments, so you can trace exactly which chunk triggered a violation.

```python
def apply_guardrail_batched(text, guardrail_id, guardrail_version):
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    content_blocks = [{"text": {"text": chunk}} for chunk in chunks]

    response = bedrock_runtime.apply_guardrail(
        guardrailIdentifier=guardrail_id,
        guardrailVersion=guardrail_version,
        source="INPUT",
        content=content_blocks
    )
    return response
```

See the full implementation with error handling in the [GitHub repo](https://github.com/svguruprasad/guardrails-batch-test).

The trade-off is that batching optimizes for RPS, not TUPs. You send the same total text volume — you just send it in fewer calls. If TUPs is your binding constraint, batching alone won't help. You need caching or selective application to reduce the actual text volume hitting the API. This technique works best when you chunk large inputs. The bigger the input, the bigger the RPS savings. A 50,000-character document drops from 50 API calls to 1. Expect RPS reductions of 60–80%, with TUPs unchanged since you're sending the same text volume in fewer calls.

## Intelligent caching

Intelligent caching stores guardrail results for repeated content so you only evaluate what's actually new. This reduces both RPS and TUPs.

The key insight here is that most text in an enterprise AI request isn't unique. System prompts repeat identically across every call. RAG context from knowledge bases recurs frequently — the same product documentation, the same policy excerpts. Only the user's input changes every time. Consider a gateway handling 500 requests per second. Each request includes a 2,000-character system prompt, 3,000 characters of RAG context, and 500 characters of user input. Without caching, you consume 5.5 TUPs per request — that's 2,750 TUPs per second across your fleet. With caching, only the 500-character user input hits the API — a 91% reduction in text volume.

You implement this by hashing each content block and storing the guardrail result with a time-to-live (TTL) based on content type. System prompts get a long TTL (24 hours) because they rarely change. RAG context gets a shorter TTL (1 hour) to match your knowledge base refresh cycle. User input never gets cached — it's unique and potentially adversarial.

```python
SYSTEM_PROMPT_TTL = 86400   # 24 hours
RAG_CONTEXT_TTL = 3600      # 1 hour

def apply_with_cache(text, content_type, guardrail_id, version):
    if content_type == "user_input":
        return call_guardrail(text, guardrail_id, version)
    cached = cache.get(text)
    if cached:
        return cached
    result = call_guardrail(text, guardrail_id, version)
    ttl = SYSTEM_PROMPT_TTL if content_type == "system_prompt" else RAG_CONTEXT_TTL
    cache.put(text, result, ttl=ttl)
    return result
```

See the full `GuardrailCache` class in the [GitHub repo](https://github.com/svguruprasad/guardrails-batch-test).

The trade-off is that caching requires an invalidation strategy. When you update your guardrail configuration — adding a new denied topic, changing a content filter threshold — cached results from the old version become stale. Include the guardrail version in your cache key so updates automatically invalidate old entries. You also need to size your cache appropriately. A gateway handling 500 unique RAG contexts per hour with a 1-hour TTL holds 500 entries. That's small. But if your RAG corpus is large and diverse, cache hit rates drop and the benefit shrinks. This technique works best when your workload has high content repetition. Enterprise gateways with standardized system prompts and curated knowledge bases see 60–70% cache hit rates. Expect RPS and TUPs reductions of 50–70% when repetition is high. If every request contains entirely unique content, caching won't help.

## Selective application

Selective application reduces the total text volume reaching the API by skipping content that doesn't need evaluation. Not all content carries the same risk.

Your team authored the system prompt. Your internal tools generated the tool output. Evaluating trusted content wastes quota on text that will never trigger a violation. You classify content into risk tiers and apply guardrails accordingly. High-risk content (user input, model output shown to end users) always gets evaluated. Low-risk content (system prompts evaluated at deployment, tool outputs from trusted systems) gets skipped at request time.

In practice, this means you tag each content block by type within a single request and only send the ones that need fresh evaluation. System prompts get evaluated once when you deploy them — cache that result and skip them at request time. Tool outputs from trusted internal systems skip evaluation entirely. User input always goes through.

```python
def apply_selective(request, guardrail_id, version):
    blocks = [{"text": {"text": request["user_input"]}}]

    for ctx in request.get("rag_contexts", []):
        if cache.get(ctx) is None:
            blocks.append({"text": {"text": ctx}})

    # System prompts: evaluated once at deployment, cached
    # Tool outputs: trusted internal systems, skipped
    if not blocks:
        return {"action": "NONE"}

    return bedrock_runtime.apply_guardrail(
        guardrailIdentifier=guardrail_id,
        guardrailVersion=version,
        source="INPUT",
        content=blocks
    )
```

The trade-off is that selective application reduces coverage by design. You're choosing not to evaluate certain content. That's safe when the content comes from trusted sources you control. It's risky if your "trusted" source can be influenced by user input — for example, a tool that echoes user text in its output. Audit your selective filters regularly and verify the risk classification still holds as your system evolves. This technique works best when your requests contain a mix of trusted and untrusted content. A request that's 80% system prompt and RAG context with 20% user input can skip most of its text volume. Expect RPS and TUPs reductions of 30–50%.

When you combine all three techniques, the reductions compound. Selective application reduces what enters the pipeline. Caching eliminates repeated content from what remains. Batching consolidates the rest into minimal API calls. Together, expect RPS reductions of 80–90% and TUPs reductions of 60–80%. Results vary based on the ratio of repeated versus unique content in your workload.

## Testing and results

We tested these techniques with a structured experiment. The code and results are in the [companion GitHub repository](https://github.com/svguruprasad/guardrails-batch-test).

We started with a simple question: does batching actually reduce RPS, or does the API count each content block separately? We sent 5 content blocks in a single `ApplyGuardrail` call and confirmed it consumed 1 RPS, not 5. That single finding validates the entire batching strategy.

Next, we verified that text units sum across blocks as expected. Five 1,000-character blocks in one call consumed 5 TUPs — same as five separate calls. This confirms batching is RPS-neutral on text volume. You save on call count, not on text processing.

The independence test surprised us the least but mattered the most for production use. We embedded a policy violation in one block and clean text in the others. The API flagged only the violating block. This means you can batch aggressively without worrying that one bad chunk contaminates the entire request.

The performance difference was striking. Five individual `ApplyGuardrail` calls took 43.69 seconds. One batched call with the same 5 content blocks took 0.23 seconds. That's a 190x speed improvement. The gap comes from eliminating per-call overhead: connection setup, request serialization, and round-trip latency — multiplied across every call.

These aren't theoretical numbers. Run the test suite yourself from the [GitHub repo](https://github.com/svguruprasad/guardrails-batch-test) to validate against your own guardrail configuration.

## Monitoring

After you deploy these techniques, track these [Amazon CloudWatch](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/WhatIsCloudWatch.html) metrics for your guardrails in the `AWS/Bedrock` namespace via the [CloudWatch console](https://console.aws.amazon.com/cloudwatch/home):

**InvocationThrottles (Sum)** — This should drop toward zero. If it doesn't, your batching or caching isn't catching enough traffic.

**TextUnitCount (Sum)** — Should decrease proportionally to your caching and selective application hit rates.

**Invocations (Sum)** — Should decrease proportionally to batching and caching combined.

Add client-side metrics too. Track your cache hit rate — target above 60% for enterprise workloads. Track requests skipped by selective application to measure how much low-risk content you're filtering. Track average content blocks per batched call — higher means more RPS savings.

## Best practices

**Start with caching.** It delivers the highest impact with the lowest complexity.

**Invalidate on guardrail changes.** Include the guardrail version number in your cache key. When you update your guardrail configuration, cached results from the old version become stale.

**Never cache user input.** User-generated content is unique and potentially adversarial. Always evaluate it fresh.

**Test with production traffic patterns.** Optimization impact depends on your content mix. Use [guardrails detect mode](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-harmful-content-handling-options.html) to test without blocking real requests.

**Audit your selective filters regularly.** Review which content types you exclude and verify the risk classification still holds.

**Consider multi-region distribution.** Quotas are per-region. Distributing traffic across regions gives you linear scaling of both RPS and TUPs limits. Check region availability on the [Amazon Bedrock endpoints page](https://docs.aws.amazon.com/general/latest/gr/bedrock.html).

## Clean up

If you created a test guardrail while following this post, delete it to avoid unintended usage:

1. Open the [Amazon Bedrock Guardrails console](https://console.aws.amazon.com/bedrock/home#/guardrails).
2. Select the guardrail you created for testing.
3. Choose **Delete** and confirm.

No other resources need cleanup. The Python code runs locally and doesn't create AWS resources.

## Conclusion

You don't have to choose between safety guardrails and high throughput. Content block batching, intelligent caching, and selective application cut `ApplyGuardrail` API calls by 80–90%. Our tests confirmed a 190x speed improvement for batched calls.

These techniques work best in centralized AI gateway architectures. Repeated content like system prompts and RAG context often makes up 60–70% of text volume. Caching alone eliminates most of that. If RPS is your bottleneck, start with batching. If TUPs is your constraint, start with caching. If both are bottlenecks, layer all three. For most enterprise workloads, the right answer is all three techniques together. Start with caching (highest impact, lowest complexity), add batching (straightforward), then layer in selective application (requires risk classification work).

Here's how to get started. You'll need an AWS account with [Amazon Bedrock](https://console.aws.amazon.com/bedrock/home) access and a guardrail configured with content filters. Code samples use Python and boto3.

1. Clone the [companion repo](https://github.com/svguruprasad/guardrails-batch-test) and run the test suite against your guardrail to validate the batching behavior.
2. Add caching for system prompts and RAG context. This single change delivers the largest immediate impact.
3. Layer in content block batching for large inputs.
4. Classify your use cases by risk tier and apply selective filtering.
5. Set up CloudWatch monitoring to track your RPS and TUPs reductions.

For quota adjustments, visit the [Service Quotas console](https://console.aws.amazon.com/servicequotas/home/services/bedrock/quotas) for Amazon Bedrock.

## Resources

- [Amazon Bedrock Guardrails documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html)
- [ApplyGuardrail API reference](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ApplyGuardrail.html)
- [Companion GitHub repository — guardrails-batch-test](https://github.com/svguruprasad/guardrails-batch-test)
- [Build safe generative AI applications like a Pro: Best Practices with Amazon Bedrock Guardrails](https://aws.amazon.com/blogs/machine-learning/build-safe-generative-ai-applications-like-a-pro-best-practices-with-amazon-bedrock-guardrails/)
- [Use the ApplyGuardrail API with long-context inputs and streaming outputs](https://aws.amazon.com/blogs/machine-learning/use-the-applyguardrail-api-with-long-context-inputs-and-streaming-outputs-in-amazon-bedrock/)
- [Amazon Bedrock Guardrails cross-account safeguards](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-enforcements.html)
- [Amazon Bedrock service quotas](https://console.aws.amazon.com/servicequotas/home/services/bedrock/quotas)

---

## About the author

**Guruprasad Seeryada** is a Senior Technical Account Manager at AWS Enterprise Support based in Atlanta, GA. He works with enterprise customers in the financial services industry to optimize their AWS architectures for resilience, security, and performance.
