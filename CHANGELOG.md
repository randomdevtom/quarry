# Changelog

## Unreleased

- New architecture: LFM2 MoE (`lfm2moe`, e.g. LFM2.5-8B-A1B): 13.7-14.1 tok/s generation and
  38 tok/s prompts on the test laptop (llama.cpp: 12.4 / 35.7), logits match llama.cpp.
  Its chat template, thinking, and tool calls in LFM's pythonic format
- Q4_K/Q5_K kernels: scales unpacked in registers (a store-forwarding stall on every block)
- Q6_K kernel rewritten at 512 bits
- `Q4_KR`: Q4_K reordered in place at load for a cheaper dot, and exact batched integer
  matmuls for Q4_KR/Q6_K
- expert cache: optional per-part hook after a read (used for the in-place repack)
- non-streamed replies include `reasoning_content`
- `tests/test_kquants.c`: K-quant kernels on random blocks (runs in `make test`, no model
  needed); `tests/test_lfm.c`: logits against a llama.cpp reference

## v0.3.0 (2026-11-23)

- Chat page: replies keep generating on the server when the phone sleeps or the tab is
  closed, and the page picks them up again on reload (`/v1/job`, `/v1/job?partial=1`,
  `/v1/job?last=1`)
- Qwen3.5/3.6: faster prompt processing (~32 tok/s at 4K tokens), blocked attention,
  vectorized softmax, batched routing; all 8 experts by default
- Adaptive MTP drafting: drafts only while acceptance stays above ~65%
- Images for Qwen3.5/3.6 (llama.cpp's libmtmd); prompt state reused across turns
- `developer` role accepted as `system`
- Inkling 2: fixed-order expert sum, `QUARRY_TOPK`
- New logo

## v0.2.0 (2026-11-22)

- New architectures: Inkling 2 (`inkling2_loop`), Qwen3.5/3.6 MoE (`qwen35moe`) with MTP
  speculative decoding, Mellum 2.1 (`mellum`)
- Q4_0/Q4_1 kernels, packed expert file
- Tool calling, fp16 KV cache
- Built-in chat page, installer, `/v1/stats`, `/v1/cancel`
- Reasoning streamed as `reasoning_content`

## v0.1.0 (2026-11-22)

- First release: Qwen3-MoE on the CPU with an SSD-backed expert cache, next-layer expert
  prefetch, OpenAI-compatible API
