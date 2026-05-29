# Part 1 — Can a 32 GB Workstation GPU Really Serve LLMs? Setting Up vLLM on RDNA4

*Reading time: ~5 minutes*

## The question

Most people picture LLM inference happening on data-center cards — H100s, MI300X, eight-GPU servers humming in a rack. But a lot of real-world AI work doesn't need that scale. A developer wants a private code assistant. A small team wants a couple of agents looping over tools. A researcher wants to prototype without renting GPU time.

For that world, the interesting question is simpler:

> Can a single 32 GB workstation GPU actually serve a modern LLM well enough to feel real?

This three-part series walks through what we found out. We took a current-generation 32 GB workstation card, pointed open-source inference stacks at it, and measured.

This first part is the setup story — the "does it even work" chapter. Parts two and three get to the fun results.

## The hardware and the stack

The card under test is a modern 32 GB workstation GPU based on a new consumer/workstation graphics architecture (think "the workstation cousin of the latest gaming GPUs" — not a data-center accelerator). Two important things about that:

- **32 GB of VRAM** is the magic number. It's enough to fit a quantized 30B-class model *and* meaningful KV cache for a handful of concurrent users.
- It's a **graphics-style architecture, not a compute-style one**. That distinction will matter a lot in Part 3.

The host is a standard Linux server with plenty of CPU and RAM — nothing exotic. The runtime is the open-source vLLM project, packaged in a vendor-supplied Docker image that pairs vLLM with the matching GPU compute stack. Everything in this series uses publicly available containers and publicly available models from Hugging Face. No proprietary builds.

## The first model: a dense 14B at 4-bit

To set a baseline, we started with a well-understood model: a **14B-parameter dense transformer**, quantized to 4 bits using AWQ. This is the kind of model that's a workhorse for code completion, chat, and general-purpose assistant work. At 4-bit weights it occupies roughly 9–10 GB of VRAM, leaving plenty of headroom on a 32 GB card for KV cache and overhead.

Bringing it up was a one-command affair once the container was pulled:

```bash
docker run -d --name vllm-server \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host --shm-size=16g \
  --network=host \
  <vendor>/<vllm-image>:<tag> \
  vllm serve <org>/<14B-AWQ-model> \
    --dtype float16 --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching
```

(Generic shape — substitute the image tag and model name from your provider of choice. No environment-specific values are needed here.)

A few minutes later, the server was answering OpenAI-compatible chat completion requests on port 8000. **It worked on the first try** — which, for anyone who's done GPU plumbing before, is itself worth noting. The container ecosystem around vLLM has matured a lot.

## The first benchmark — and the first surprise

We ran a simple single-stream benchmark: one user, one prompt, 512 output tokens, temperature 0. The result:

> **~7.5 tokens per second.**

Functional, useful for coding-assistant-style workloads, but slower than expected for a card with this much memory bandwidth and compute. Why?

The clue came from a follow-up experiment. We tried a smaller model — half the parameters — at the same quantization. Throughput **roughly doubled.**

That's a textbook signature of being **memory-bandwidth-bound at batch size 1.** At single-stream decode, the GPU spends most of its time waiting on weights to arrive from VRAM. Compute units sit idle. So the throughput is essentially `bandwidth / model-size-in-bytes`, which is why halving the model doubles the speed.

This is true on basically every GPU at batch=1, including the most expensive data-center cards. The difference is just how much bandwidth you have.

## What this means

Two takeaways set up the rest of the series:

1. **The workstation card works.** Functional inference of a 14B-class model on a single GPU, with a real OpenAI-compatible API, prefix caching, tool calling — all of it. No exotic plumbing.

2. **Single-stream speed is a memory game, not a compute game.** If we want to get more performance out of this card, we have one of two paths: serve *more* requests in parallel (so the bandwidth gets amortized across users), or change the *shape* of the model so each output token reads fewer weights.

Parts two and three pull on each of those threads. Part two is the concurrency story — how multiple users actually scale on this card, and how a different model architecture changed everything. Part three is the plot twist: keeping the model the same and just swapping the inference runtime, and finding out the silicon had more to give than we thought.

## How to replicate

If you want to try this yourself:

- Use any modern 24–32 GB GPU.
- Pull your vendor's published vLLM container image (most major GPU vendors maintain one).
- Pick a publicly available AWQ-quantized 7B–14B model from Hugging Face.
- The `docker run` shape above is generic — the only platform-specific bits are the `--device` flags and the image tag.
- For the benchmark, any Python script that hits the `/v1/chat/completions` endpoint with `stream=true` and times tokens is enough. vLLM also ships its own benchmarking scripts.

Total time from "container not pulled yet" to "first token streamed back" was about 20 minutes the first time, including the model download.

---

*Next up: **Part 2 — When 30 billion parameters got faster than 14 billion.** We swap in a Mixture-of-Experts model and watch the single-stream number go from 7.5 to 51 tokens per second — same card, same VRAM budget, bigger model.*
