# Optimize Amazon Bedrock Guardrails for high-throughput workloads

This repository holds the code for the post below. [`guardrail_optimizer.py`](guardrail_optimizer.py) has helpers for three ways to use less of your Amazon Bedrock Guardrails quota: content block batching, result caching, and selective application. [`test_batch.py`](test_batch.py) checks the batching behavior against a live guardrail, so you can see the numbers for yourself.

## Quick start

You need Python 3.9 or later, an AWS account with Amazon Bedrock access, and credentials in your shell that allow `bedrock:CreateGuardrail`, `bedrock:DeleteGuardrail`, and `bedrock:ApplyGuardrail`.

```bash
git clone https://github.com/aws-samples/sample-bedrock-guardrails-throughput-optimization
cd sample-bedrock-guardrails-throughput-optimization
python3 -m pip install -r requirements.txt
export AWS_PROFILE=your-profile   # replace with a profile that has Bedrock access

python3 setup_guardrail.py                        # creates a test guardrail and prints its ID
python3 test_batch.py --guardrail-id <id>         # runs four checks and writes a results JSON file
python3 delete_guardrail.py --guardrail-id <id>   # deletes the test guardrail
```

On Windows, set the profile with `set AWS_PROFILE=your-profile` in Command Prompt or `$env:AWS_PROFILE="your-profile"` in PowerShell. Replace `<id>` with the ID that `setup_guardrail.py` prints.

All three scripts default to us-east-1. To use another Region, pass the same `--region` value to all three.

When the test run finishes, it prints a summary like this and saves the full results to `batch-test-results-<timestamp>.json`:

```
    H1: Batched call = 1 request for N blocks         ✅ CONFIRMED
    H2: Batched text units ≤ single (per-call rounding) ✅ CONFIRMED
    H3: Violation in a batch is detected (aggregated)  ✅ CONFIRMED
    H4: Rate-limited batching ran without throttle     ✅ CONFIRMED

    Total ApplyGuardrail invocations this run: 11
```

## What's in the repo

| File | What it does |
|------|--------------|
| [`guardrail_optimizer.py`](guardrail_optimizer.py) | The reusable helpers: `apply_guardrail_batched`, the `GuardrailCache` class with `apply_with_cache`, `apply_selective`, and `total_text_units` for reading per-policy usage. |
| [`test_batch.py`](test_batch.py) | Runs four checks against a live guardrail: call reduction, text unit accounting, detection inside a batch, and paced batching. |
| [`setup_guardrail.py`](setup_guardrail.py) | Creates a test guardrail with the content filters the checks expect. |
| [`delete_guardrail.py`](delete_guardrail.py) | Deletes the test guardrail. |
| [`batch-test-results-20260812-182047.json`](batch-test-results-20260812-182047.json) | The run that the numbers in the post come from. |
| [`requirements.txt`](requirements.txt) | Pinned `boto3` and `botocore` versions. |

## Cost

A test run makes 11 `ApplyGuardrail` calls and uses about 85 text units. Guardrails are billed per text unit, so a run costs a few cents at most. See [Amazon Bedrock pricing](https://aws.amazon.com/bedrock/pricing/) for current rates. The test guardrail costs nothing while idle, but it counts against your account's guardrail quota, so delete it when you're done.

In production, the cost of Guardrails scales with the text you evaluate. Every text unit you remove with caching or selective application lowers your bill as well as your quota use. Batching mostly saves calls, not text units, so it does little for cost.

## Known limitations

- `GuardrailCache` keeps entries in process memory. A gateway with more than one instance needs a shared store such as Amazon ElastiCache.
- The numbers in the post come from one run in us-east-1 against a guardrail that uses content filters only. Guardrails with more policy types use more text units per call.
- Test 4 checks the pacing logic at low volume. It doesn't reproduce throttling at production scale.

---

# Optimize Amazon Bedrock Guardrails for high-throughput workloads

by Guruprasad Seeryada | AWS Enterprise Support

**Categories**: Amazon Bedrock, Amazon Bedrock Guardrails, Best Practices, Generative AI, Intermediate (200)

---

## Introduction

Say you run a central AI gateway for your company. Every prompt and every model response passes through it, and you've decided that all of it should go through [Amazon Bedrock Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html). That works fine in a proof of concept. Then traffic grows to thousands of users, and the `ApplyGuardrail` API starts throttling your gateway.

This post covers three techniques that reduce how much Guardrails quota each request uses: content block batching, result caching, and selective application. Each one attacks a different part of the problem, and they work well together. How much you save depends on how much of your traffic repeats, so rather than promise a percentage, the post gives you a test suite to measure your own numbers.

By the end you'll have a request flow for your gateway, working Python code in [`guardrail_optimizer.py`](guardrail_optimizer.py), and a short list of Amazon CloudWatch metrics to watch.

## Two quotas, not one

Two [service quotas](https://console.aws.amazon.com/servicequotas/home/services/bedrock/quotas) govern `ApplyGuardrail`. The first is requests per second (RPS), which counts API calls. The second is text units per second (TUPs), which counts how much text you evaluate. One text unit is 1,000 characters. Both quotas are set per Region, and you can request an increase for either one in the Service Quotas console.

Text units add up across policy types. A 1,000-character call against a guardrail with content filters, denied topics, and sensitive information filters uses three text units, one for each policy type. Categories inside the content filter don't add up the same way. Turning on all six content filter categories still counts as one text unit per 1,000 characters. The post [Best practices for applying Amazon Bedrock Guardrails to code generation workflows](https://aws.amazon.com/blogs/machine-learning/best-practices-for-applying-amazon-bedrock-guardrails-to-code-generation-workflows/) walks through this math in more detail.

The two quotas are separate, and each one has its own throttling message. That makes it easy to fix one and make the other worse. If you split a large document into small pieces so that no single call carries too much text, you make more calls, and now RPS is the problem. The techniques in this post were picked so you can work on both at once.

Here's the request flow we'll build:

```
User request → AI gateway → [cache check] → [selective filter] → [batch content blocks] → ApplyGuardrail → response
```

Each stage removes work before the next one runs. The cache check drops content you've already evaluated, such as a system prompt that's identical on every request. The selective filter drops content you've decided doesn't need evaluation, such as output from your own internal tools. Batching packs whatever is left into one API call. Run them in that order, because the cache check is the cheapest.

## Content block batching

The [ApplyGuardrail API](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ApplyGuardrail.html) accepts a list of content blocks in one request. If you're chunking a long document anyway, you can send all the chunks in one call instead of one call per chunk. That's all batching is, and it goes straight at your RPS quota.

Take a 10,000-character document split into ten 1,000-character chunks. Sent one at a time, that's ten calls. If your gateway handles 500 of these documents per second, you need 5,000 RPS. Sent as one batch, each document is one call, and the same traffic needs 500 RPS.

```python
def apply_guardrail_batched(client, text, guardrail_id, guardrail_version="DRAFT",
                            chunk_chars=1000):
    chunks = [text[i:i + chunk_chars] for i in range(0, len(text), chunk_chars)]
    content_blocks = [{"text": {"text": chunk}} for chunk in chunks]

    return client.apply_guardrail(
        guardrailIdentifier=guardrail_id,
        guardrailVersion=guardrail_version,
        source="INPUT",
        content=content_blocks,
    )
```

The full version in [`guardrail_optimizer.py`](guardrail_optimizer.py) adds error handling and refuses input that's too large for one request.

Before you rely on batching, know what comes back. The response to a batched call is aggregated. You get one top-level `action` and, in our tests, one assessment for the whole call rather than one per block. If any block violates a policy, the whole call comes back as `GUARDRAIL_INTERVENED`. That's the safe direction, because a bad block can't hide inside a batch. The catch is that the response won't tell you which block caused it. If your application needs to know, check the blocks from a flagged call again in smaller calls.

Batching changes your text unit count less than you might expect. You still send the same characters, so most of your TUPs use stays the same. Rounding makes a small difference, which the results section covers. If TUPs is the quota that's hurting you, batching won't fix it, but caching and selective application will.

Each request also has a size limit. A request over the limit is rejected with a `ValidationException` rather than throttled, so split very large inputs across several batched calls. Check the current limit for your Region in Service Quotas.

One security note. Cutting text every 1,000 characters can split a phrase across two blocks. A banned term or a prompt injection that straddles the boundary might not be detected, even though the same text inside one block would be. For untrusted input, use overlapping chunks or split on sentence or paragraph boundaries.

## Result caching

Most of the text in a gateway request isn't new. The system prompt is the same on every call. RAG context comes from a knowledge base, so the same product documents and policy excerpts show up again and again. Usually only the user's message is new. If you cache guardrail results for the repeated parts, you evaluate them once instead of on every request, which cuts both RPS and TUPs.

Here's how much text that can be. Say each request carries a 2,000-character system prompt, 3,000 characters of RAG context, and a 500-character user message. That's 5.5 text units per request, or 2,750 text units per second at 500 requests per second. If the system prompt and RAG context come from the cache, only the 500-character message goes to the API, about 9% of the original text. That figure is arithmetic for this example, not a measurement. Your real savings depend on how often your cache hits.

The cache key combines a hash of the content with the guardrail version. The version matters. When you change your guardrail, for example by adding a denied topic, results from the old version stop matching and get evaluated again. Each content type gets its own time to live (TTL). System prompts rarely change, so they get 24 hours. RAG context gets one hour, to match a typical knowledge base refresh. User input is never cached, because it's different every time and it's the content you trust least.

```python
def apply_with_cache(client, text, content_type, guardrail_id, version, cache):
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
```

The `GuardrailCache` class in [`guardrail_optimizer.py`](guardrail_optimizer.py) handles the key, the TTLs, and hit rate tracking. It keeps entries in memory. If your gateway runs on several instances, back the cache with a shared store such as Amazon ElastiCache so every instance sees the same entries.

Think about cache size too. If you see 500 distinct RAG passages an hour and use a one hour TTL, the cache holds about 500 entries, which is small. If your knowledge base is large and each request pulls different passages, the hit rate drops and so do the savings. If every request is unique, caching won't help.

## Selective application

Some of the content in a request came from you. Your team wrote the system prompt, and your own internal services produced the tool output. Running guardrails on that text on every request spends quota on content you already trust.

Selective application sorts content by risk. User input, and any model output shown to users, is always evaluated. The system prompt is evaluated once when you deploy it, and that result is cached. Output from trusted internal tools is skipped. RAG context is sent only when it isn't already in the cache.

```python
def apply_selective(client, request, guardrail_id, version, cache=None):
    blocks = [{"text": {"text": request["user_input"]}}]
    for ctx in request.get("rag_contexts", []):
        if cache is None or cache.get(ctx, version) is None:
            blocks.append({"text": {"text": ctx}})

    # System prompts were evaluated at deploy time and trusted tool output is
    # skipped, so neither is sent here.
    return client.apply_guardrail(
        guardrailIdentifier=guardrail_id, guardrailVersion=version,
        source="INPUT", content=blocks,
    )
```

This function doesn't write RAG passages back to the cache. It can't, because the batched response doesn't say which passage caused an intervention. Fill the cache for RAG passages some other way, for example by running each passage through `apply_with_cache` when you add it to your knowledge base.

If your application calls models through `InvokeModel` or `Converse` with a guardrail attached, Bedrock can do this filtering for you. You mark the parts of the prompt the guardrail should evaluate, and it skips the rest. See [Apply tags to user input to filter content](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-tagging.html).

The trade-off is that you cover less on purpose. That's safe only if the content you trust is truly out of the user's reach. A tool that echoes user text back, or a document a user can upload into your knowledge base, turns trusted content into user content. Review what you skip each time you add a tool or a data source.

## Putting them together

The three techniques stack. Selective application decides what enters the pipeline, caching removes repeats from what's left, and batching sends the rest in as few calls as possible. Your overall savings depend on your mix of repeated and new content, so measure them with the test suite and CloudWatch before you plan capacity around them.

## Testing and results

We ran [`test_batch.py`](test_batch.py) once against a test guardrail in us-east-1 on August 12, 2026. The guardrail used content filters only. The raw output is in [`batch-test-results-20260812-182047.json`](batch-test-results-20260812-182047.json). It's a single run, so read the timings as a direction rather than a benchmark.

First, we checked that batching cuts calls. The test wraps the client and counts real calls instead of assuming them. Five single-block calls counted as five, and one batched call with the same five blocks counted as one. You can confirm this on the service side with the `Invocations` metric.

Next came text units. The `ApplyGuardrail` response has no single text unit field. It reports units per policy type, in fields such as `contentPolicyUnits` and `topicPolicyUnits`, so the test adds them up. We sent five 1,500-character blocks. As five separate calls they used 10 text units, two per call, because each call rounds up. As one batched call they used 8, which matches rounding once over 7,500 characters. So batching can save a little on text units as well as on calls.

One result surprised us. In the pacing test, we sent 30 chunks of exactly 1,000 characters in three batched calls of 10,000 characters each. We expected 30 text units and got 33, one extra per call. The large batch test showed the same pattern: 20 blocks totaling 30,000 characters used 31 units. Every call in the run used its character count divided by 1,000, rounded down, plus one. For most lengths that's the same as rounding up. When a call lands exactly on a multiple of 1,000 characters, it costs one extra unit. We haven't found this documented, so treat it as something we observed. When you plan capacity, allow for one extra unit per call.

Then we tested detection. We sent a harmless question and a prompt injection together in one call. The guardrail intervened and reported `PROMPT_ATTACK` as `BLOCKED`. It returned one assessment for both blocks, so the response didn't say which block was the problem, as described in the batching section.

The last check was speed. The five sequential calls took 1.48 seconds, and the batched call took 0.28 seconds. Most of that gap is per-call overhead, meaning request setup plus a network round trip for every call. If you want a speedup figure for your own environment, warm up the client first and average several runs. The first calls in a process include SDK and TLS setup, which makes the sequential path look slower than it is.

The pacing test also sent its three batched calls under a TUPs budget without being throttled. That shows the pacing logic works. It doesn't show what happens at production volume, because the test stays far below the quota on purpose.

## Monitoring

Guardrail metrics live in the `AWS/Bedrock/Guardrails` namespace in [Amazon CloudWatch](https://console.aws.amazon.com/cloudwatch/home). The [metrics reference](https://docs.aws.amazon.com/bedrock/latest/userguide/monitoring-guardrails-cw-metrics.html) lists all of them. Three matter most here.

`InvocationThrottles` counts throttled calls, and it should fall toward zero. If it doesn't, read the throttling message to see which quota you're hitting before you change anything.

`TextUnitCount` counts text units used. You can break it down by the `GuardrailPolicyType` dimension to see which policy type uses the most. It should fall as caching and selective application take effect.

`Invocations` counts `ApplyGuardrail` calls, and it should fall as batching and caching take effect. Throttled calls don't show up in this metric, so read it together with `InvocationThrottles`.

Your own code should report a few numbers as well: the cache hit rate, how many blocks selective application skipped, and the average number of blocks per batched call. Together they tell you which technique is doing the work.

## Best practices

Start with caching. It's the simplest to add, and it cuts both quotas.

Put the guardrail version in your cache key, so a change to your guardrail makes old results stop matching.

Never cache user input.

Test with real traffic patterns. [Detect mode](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-harmful-content-handling-options.html) shows what your guardrail would do without blocking users, which lets you measure while you tune.

Review your selective filters each time you add a tool or a data source.

Request a quota increase when you need one. RPS and TUPs are both adjustable per Region in the [Service Quotas console](https://console.aws.amazon.com/servicequotas/home/services/bedrock/quotas). The techniques in this post lower how much quota you need, but you still have to size your quota for your peak traffic.

If one Region isn't enough, spread traffic across Regions. Quotas are per Region, so each Region you add brings its own RPS and TUPs. Check your data residency requirements first, and check where Bedrock is available on the [Amazon Bedrock endpoints page](https://docs.aws.amazon.com/general/latest/gr/bedrock.html).

## Clean up

If you created the test guardrail, delete it:

```bash
python3 delete_guardrail.py --guardrail-id <id>
```

You can also delete it in the console:

1. Open the [Amazon Bedrock Guardrails console](https://console.aws.amazon.com/bedrock/home#/guardrails).
2. Select the guardrail you created for testing.
3. Choose **Delete** and confirm.

`setup_guardrail.py` is the only script that creates a resource. The test script only calls the runtime API.

## Conclusion

Running guardrails on every request doesn't have to mean constant throttling. Batching cuts the number of calls. Caching and selective application cut the amount of text you send. Which one to start with depends on the quota you're hitting: batching for RPS, caching for TUPs, and all three if you're short on both. For most gateways, a good order is caching first, then batching, then selective application. Selective application comes last because it takes the most work up front, since you have to classify your content by risk.

To try it, follow the quick start at the top of this page, and then:

1. Add caching for system prompts and RAG context.
2. Batch the chunks of large inputs.
3. Classify content by risk, and skip what you trust.
4. Watch the CloudWatch metrics above as you roll out each step.

## Resources

- [Amazon Bedrock Guardrails documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html)
- [ApplyGuardrail API reference](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ApplyGuardrail.html)
- [Monitor Amazon Bedrock Guardrails using CloudWatch metrics](https://docs.aws.amazon.com/bedrock/latest/userguide/monitoring-guardrails-cw-metrics.html)
- [Companion repository: sample-bedrock-guardrails-throughput-optimization](https://github.com/aws-samples/sample-bedrock-guardrails-throughput-optimization)
- [Build safe generative AI applications like a Pro: Best Practices with Amazon Bedrock Guardrails](https://aws.amazon.com/blogs/machine-learning/build-safe-generative-ai-applications-like-a-pro-best-practices-with-amazon-bedrock-guardrails/)
- [Use the ApplyGuardrail API with long-context inputs and streaming outputs](https://aws.amazon.com/blogs/machine-learning/use-the-applyguardrail-api-with-long-context-inputs-and-streaming-outputs-in-amazon-bedrock/)
- [Amazon Bedrock Guardrails cross-account safeguards](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-enforcements.html)
- [Amazon Bedrock service quotas](https://console.aws.amazon.com/servicequotas/home/services/bedrock/quotas)

---

## About the author

**Guruprasad Seeryada** is a Senior Technical Account Manager at AWS Enterprise Support based in Atlanta, GA. He works with enterprise customers in the financial services industry to optimize their AWS architectures for resilience, security, and performance.
