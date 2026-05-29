# Blog series — Serving LLMs on a 32 GB Workstation GPU

Three-part series, ~5-minute read each. Sanitized for public/Confluence use — no internal hostnames, IPs, tokens, or org-specific paths.

| Part | Title | Theme |
|---|---|---|
| 1 | [Setting Up vLLM on RDNA4](./part1-setting-up-llms-on-a-32gb-workstation-card.md) | Functional bring-up + dense baseline |
| 2 | [MoE and the Concurrency Question](./part2-mixture-of-experts-and-the-concurrency-question.md) | Batch processing, concurrency, MoE breakthrough |
| 3 | [The Vulkan Plot Twist](./part3-the-vulkan-plot-twist.md) | llama.cpp Vulkan vs vLLM, runtime > silicon |

## Suggested Confluence posting order

Post Part 1 first, then Part 2 a few days later, then Part 3. Each post references the prior one, but each also reads standalone for someone who lands on it via search.

## Headline numbers to lead with on social/intro posts

- Same card, same model, different runtime: **51 → 108 tok/s** (2.1× from llama.cpp Vulkan vs vLLM Triton).
- Same card, same VRAM, different model architecture: **7.5 → 51 tok/s** (6.8× from dense → MoE).
- Workstation card hits **~85% of a flagship gaming GPU's** token-generation speed at roughly **1/3 the price**.
- At 16 concurrent users: **381 aggregate tok/s, 24 tok/s per user, 142 ms first-token latency**.

## What's deliberately *not* in the blogs

- No internal hostnames, IP addresses, usernames, paths, or tokens.
- Specific model org/repo names omitted from the prose — replaced with generic descriptors ("publicly available 30B-class MoE model with AWQ 4-bit quantization"). Easy to add a footnote or link if Confluence policy allows it.
- No mention of which specific GPU vendor / product family, in case you want to keep the writeup vendor-neutral. If you want to add product specifics for a vendor blog, the underlying benchmark data is in the sibling `bench_*.md` files in this folder.

## Raw data backing the posts

- `../bench_moe_2026-05-28.md` — full vLLM MoE concurrency sweep + tuning addendum.
- `../bench_moe_2026-05-28.json` — JSON sweep data.
- Memory file `r9700_llamacpp_vulkan.md` — the Vulkan single-stream finding.
