# Part 2 — When 30 Billion Parameters Got Faster Than 14 Billion: MoE and the Concurrency Question

*By Pradeep Nallimelli — AMD Field AI Enthusiast*

*Reading time: ~5 minutes*

In [Part 1](./part1-setting-up-llms-on-a-32gb-workstation-card.md) we set up a single 32 GB workstation GPU with vLLM and a 14B dense model, and discovered it was bandwidth-bound at single-stream — about 7.5 tokens per second. Useful, but not exciting.

This post is about two changes that broke that ceiling: switching the model architecture, and turning on the thing this card was actually built for — concurrency.

## The model swap: from dense to Mixture-of-Experts

We replaced the 14B dense model with a **Mixture-of-Experts (MoE) model: 30 billion total parameters, but only ~3 billion active per token.** Same 4-bit quantization. Same card. Same vLLM container.

A quick mental model of MoE for anyone who hasn't met it:

- A dense transformer reads *every* weight in the model for *every* output token.
- An MoE transformer is divided into many small expert sub-networks (often 128 of them per layer). For each token, a tiny routing network picks the best 8 experts. Only those experts run.

So even though the model has 30B parameters sitting in VRAM, each token only touches about 3B of them. The other 27B are idle for that particular token — but they're available the moment a different token wants them.

If single-stream throughput is bound by "how many bytes do I read per output token," and MoE cuts that number by ~5×, you can see where this is going.

## The number

Single-stream, same benchmark as before, same card:

> **From 7.5 tokens/sec to ~51 tokens/sec.** A 6.8× speedup. On a larger, smarter model.

First-token latency dropped from ~410 ms to ~40 ms in the same swap. Power draw at idle-ish single-user was actually *lower* (216 W vs 276 W), because compute units are no longer bottlenecked waiting on memory.

This is the headline of the whole series. **Picking the right model architecture mattered more than any kernel tuning we tried later.** It's a free 6× — no new hardware, no exotic settings, just understanding what the silicon is actually limited by and choosing a model shape that doesn't trigger that limit.

## Now the concurrency question

Single-user speed is one thing. The reason you'd put a card like this in a small team's lab or a hobbyist's machine is to serve *several* concurrent things — a couple of agents looping over tools, a code assistant for a pair of developers, an experimental fleet.

We swept concurrency from 1 to 16 simultaneous streaming requests, distinct prompts, 512 output tokens each:

| Concurrent users | Aggregate tokens/sec | Per-user tokens/sec | First-token latency |
|---:|---:|---:|---:|
| 1  | 51  | 51 | 40 ms |
| 2  | 80  | 40 | 57 ms |
| 4  | 136 | 34 | 85 ms |
| 8  | 226 | 28 | 111 ms |
| 16 | 381 | 24 | 142 ms |

A few things to notice:

**Aggregate throughput scales nicely** — 16 users get ~7.5× the work-per-second of a single user. The card stays useful as you add load.

**Per-user speed degrades, but gracefully.** Even at 16 concurrent users, each one is still getting 24 tokens/sec — faster than most people read. First-token latency stays well under 200 ms, which is comfortably in "feels instant" territory for chat and tool-loop work.

**The scaling is sub-linear for MoE specifically**, and that's expected. With a dense model, every concurrent user fetches the same weights, so 16 users barely cost more bandwidth than 1. With MoE, different users hit different experts, so aggregate bandwidth demand goes up. The good news is we *started* from a single-user number that was 6× higher — so even after the sub-linear penalty, we end up far ahead.

## What about KV cache?

Memory matters as much as bandwidth. With this model and a 32 GB card, after weights are loaded we have about 7.7 GB left for KV cache — enough for ~84,000 tokens of context across all active requests. In practical terms that's roughly 20 simultaneous users at 4K context each. So the *workload* limit on this card is more likely to be KV cache (with long contexts) than compute or bandwidth.

A useful rule of thumb: at 4K context, KV starts to backpressure around 20–24 concurrent users on a 32 GB card running this size model. At 8K context, around 10–12.

## A tuning detour worth mentioning

vLLM ships with hand-tuned kernel configs for many GPUs, but not for every (model, GPU) pair. For our combination, the runtime printed a polite warning:

> *Using default MoE config. Performance might be sub-optimal!*

We spent some time trying to close that warning — running the upstream auto-tuner, borrowing a config from a related data-center GPU, patching kernel parameters. **All three approaches failed.** The auto-tuner doesn't support 4-bit MoE quantization in the current release. Borrowing a config from a different GPU architecture caused a **4.6× regression** at high concurrency — the tile sizes were tuned for a different wavefront layout, different on-chip memory, and different matrix instructions, and using them on the wrong architecture made everything slower.

The lesson there is one of the most useful negative results in the project: **the "sub-optimal" warning is cosmetic.** The kernel that runs is the generic, untuned one — but it's already turning in 51 tokens/sec single-stream and 381 aggregate. There's no quick win hiding in a config file. The real unlock would be the GPU vendor's blessed inference kernel library extending to this consumer/workstation architecture (today it ships for data-center cards only). Not something we can patch our way to from the outside.

## What this means for agentic workloads

For the typical "small fleet" use case — 3 to 8 simultaneous agents looping over tools, doing reads and short generations:

- **You're in the comfortable zone.** Per-user speeds of 28–34 tokens/sec, first-token latencies around 100 ms, total power draw under 310 W.
- **You can push to 16 concurrent users** before per-user speed dips below the "feels fast" threshold.
- **You're not going to bottleneck on KV cache** unless you start handling long-context summarization or code-base-scale tool returns.

So the 32 GB workstation card, with a thoughtfully chosen MoE model, is a legitimate platform for serving a small agent team — not a toy.

## How to replicate

- Use the same vLLM container from Part 1.
- Swap in any publicly available **30B-class MoE model with AWQ 4-bit quantization** from Hugging Face. (There are several actively maintained.)
- Start vLLM with `--enable-prefix-caching --enable-auto-tool-choice --tool-call-parser hermes` for agent use cases.
- For concurrency benchmarks, any async Python script firing N parallel `chat/completions` requests works. Use distinct prompts to avoid the prefix cache flattering your numbers.

---

*Next up: **Part 3 — The Vulkan Plot Twist.** Same card. Same model. Different runtime. The single-stream number jumps from 51 to 108 tokens/sec — and we end up about 85% of the way to a flagship gaming GPU that costs roughly three times as much. The kernel ecosystem turns out to matter more than the silicon.*
