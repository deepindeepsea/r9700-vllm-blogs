# Part 3 — The Vulkan Plot Twist: When the Runtime Mattered More Than the Silicon

*Reading time: ~5 minutes*

In [Part 1](./part1-setting-up-llms-on-a-32gb-workstation-card.md) we set up vLLM on a 32 GB workstation GPU and got 7.5 tokens per second from a dense 14B model.

In [Part 2](./part2-mixture-of-experts-and-the-concurrency-question.md) we switched to a Mixture-of-Experts model and jumped to 51 tokens per second single-stream, with 381 aggregate tokens per second at 16 concurrent users — a comfortable home for a small fleet of agents.

We thought that was the ceiling for this card. Then we tried one more thing.

## A community datapoint we couldn't ignore

There's an active issue on the llama.cpp GitHub tracker where members of the open-source community have been benchmarking the same MoE model on various GPUs using llama.cpp's **Vulkan backend** — a portable graphics API that runs on basically any modern GPU regardless of vendor.

Their result for a flagship gaming GPU (the kind that costs roughly three times as much as our workstation card) was about **127 tokens per second single-stream** on this MoE model at 4-bit quantization.

We had 51 tokens per second on vLLM. The flagship card was at 127. We were on a card with similar raw bandwidth and meaningfully more VRAM.

So we asked the obvious question: **what if the runtime is the thing leaving performance on the floor, not the silicon?**

The experiment was tiny. Same card. Same model — actually the same architecture and quantization scheme, just packaged in the GGUF format that llama.cpp uses instead of the AWQ format vLLM consumes. Same host. Different inference engine.

## The number

We ran the standard `llama-bench` micro-benchmark via the official llama.cpp Vulkan Docker container, with all layers offloaded to GPU and flash attention enabled.

The benchmark reports two numbers: **pp512** (prompt processing throughput on a 512-token prompt) and **tg128** (token-generation throughput for 128 new tokens). These are the canonical metrics the community uses for cross-GPU comparisons.

| Test | Workstation 32 GB card (this run) | Flagship gaming GPU (community result) |
|---|---:|---:|
| Prompt processing (pp512) | **2,822 tokens/sec** | — |
| Token generation (tg128) | **108 tokens/sec** | ~127 tokens/sec |

Two takeaways jump out:

**One: the workstation card hits ~85% of a flagship gaming GPU's token-generation speed.** For a card at roughly one-third the price, with more VRAM, that's a remarkable result.

**Two: the same hardware just went from 51 tokens/sec on vLLM to 108 tokens/sec on llama.cpp Vulkan. A 2.1× single-stream speedup, with no hardware change.**

This is the plot twist of the series. The bottleneck wasn't the silicon. It was the maturity of the inference kernels available for this particular GPU architecture inside vLLM's chosen kernel framework.

## Why the gap exists

vLLM gets its MoE kernels from a framework that auto-generates GPU code from a high-level Python-like language. That framework has excellent, hand-tuned, vendor-supported kernels for data-center GPUs. For *consumer/workstation*-class GPUs from the same vendor — even though they share a lot of design DNA — it currently falls back to a generic, untuned code path.

llama.cpp, by contrast, has a Vulkan backend written by community contributors who have specifically squeezed performance out of consumer GPUs from every vendor. Vulkan is "just" a graphics API, but in 2025 it's also one of the most pragmatic ways to run LLMs portably on commodity hardware. The kernels are mature, hand-shaped to the cooperative-matrix instructions modern consumer GPUs expose, and the community has been iterating on them for over a year.

So the same hardware, exposed through two different software stacks, gives wildly different numbers. The silicon was never the limit.

## So which one should you use?

The honest answer is: **both, for different jobs.**

**Use vLLM when you need:**

- An OpenAI-compatible HTTP API.
- Concurrent users (it handles batching, scheduling, and KV cache across many sessions far better than llama.cpp).
- Tool calling, structured outputs, prefix caching, automatic prompt formatting.
- Fleet-scale serving where aggregate throughput at high concurrency matters more than single-user peak speed.

This is the right tool for hosting an internal AI service that several developers or agents share.

**Use llama.cpp Vulkan when you need:**

- Maximum single-user speed on consumer/workstation hardware.
- A single binary, no Python, no containers required.
- Portability across GPU vendors (the same Vulkan path runs on graphics cards from every major manufacturer).
- A simpler stack for personal / hobbyist use, demos, or laptop-class inference.

If you're one developer running a private coding assistant on your own GPU, llama.cpp Vulkan is probably the better answer today. If you're standing up a small shared service for a team, vLLM is.

## The bigger lesson

Three numbers from this series tell the whole story:

- **7.5 tokens/sec** — same card, dense 14B model on vLLM.
- **51 tokens/sec** — same card, MoE 30B/3B-active model on vLLM. *(Changed the model.)*
- **108 tokens/sec** — same card, same MoE model on llama.cpp Vulkan. *(Changed the runtime.)*

A 14× difference end-to-end. Zero hardware changes.

The lesson isn't that any one of these tools is better than another. It's that **the choice of model architecture and the maturity of the kernels available for your specific hardware can each matter as much as buying a more expensive GPU.** Before you spec up, try the alternatives on what you already have. The free performance is often startling.

For workstation-class GPUs with ~32 GB of VRAM, the practical 2026 picture is something like this: you have a perfectly capable platform for serving a small agent fleet (vLLM, ~24 tokens/sec per user at 16 concurrent users), and you have a single-user experience that's competitive with much more expensive flagship hardware (llama.cpp Vulkan, ~108 tokens/sec). Both, on the same box, swappable in an afternoon.

## How to replicate

The llama.cpp Vulkan side is almost embarrassingly simple:

```bash
docker run --rm --device /dev/dri --group-add video \
  -v /path/to/your/gguf/models:/models \
  --entrypoint /app/llama-bench \
  ghcr.io/ggml-org/llama.cpp:full-vulkan \
  -m /models/<your-Q4_K_XL-model>.gguf \
  -ngl 99 -fa 1 -p 512 -n 128 -r 3
```

`-ngl 99` puts all layers on GPU. `-fa 1` enables flash attention. `-p 512 -n 128` runs the standard pp512/tg128 benchmarks. `-r 3` averages over three runs.

Pick up the GGUF quant of the same MoE model from any of the public Hugging Face repos (search for the model name plus "GGUF"). The "UD-Q4_K_XL" variant is the one the community uses for these cross-GPU comparisons.

## Wrapping up

Three blog posts ago we were asking whether a 32 GB workstation GPU could serve LLMs at all. The answer turned out to be yes, comfortably, in two very different modes — and the more interesting story along the way was that **what runs on the chip matters more than the chip's spec sheet**, almost every time you look closely.

If you have access to a workstation-class GPU and you've been waiting for a reason to try it for LLM serving, this is your sign. The tooling is here, the models are here, and the numbers are better than the hardware's price tag suggests.
