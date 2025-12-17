#!/usr/bin/env python3
"""Benchmark T5 decoding with optional torch.compile.

Measures:
- TTFT: time from start of encode to first decoded token.
- TPS: tokens/sec for steady-state decoding (after first token) and overall.

Example:
  python3 bench_t5_compile.py --prompt-len 128 --batch-size 4 --output-tokens 64 --compile
"""

from __future__ import annotations

import argparse
import dataclasses
import statistics
import time
from typing import Any

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration


@dataclasses.dataclass(frozen=True)
class RunMetrics:
    ttft_s: float
    total_s: float
    tps_decode: float
    tps_overall: float


def _device_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _parse_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"fp32", "float32"}:
        return torch.float32
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def _make_synthetic_prompt_tokens(
    *,
    batch_size: int,
    prompt_len: int,
    vocab_size: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Synthetic token IDs keep token-length exact and avoid tokenizer overhead.
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    # randint() generator must be on CPU; then move.
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, prompt_len),
        dtype=torch.long,
        generator=g,
    ).to(device)
    attention_mask = torch.ones((batch_size, prompt_len), dtype=torch.long, device=device)
    return input_ids, attention_mask


@torch.inference_mode()
def _run_one(
    *,
    encoder: Any,
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    decoder_start_token_id: int,
    output_tokens: int,
    device: torch.device,
) -> RunMetrics:
    if output_tokens < 1:
        raise ValueError("output_tokens must be >= 1")

    _device_sync(device)
    t0 = time.perf_counter()

    encoder_outputs = encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )

    decoder_input_ids = torch.full(
        (input_ids.shape[0], 1),
        decoder_start_token_id,
        dtype=torch.long,
        device=device,
    )

    # First token step (TTFT includes encoder + this step).
    out = model(
        decoder_input_ids=decoder_input_ids,
        encoder_outputs=encoder_outputs,
        attention_mask=attention_mask,
        use_cache=True,
        past_key_values=None,
        return_dict=True,
    )
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    past = out.past_key_values

    _device_sync(device)
    t1 = time.perf_counter()

    # Remaining tokens, incremental (KV cache).
    remaining = output_tokens - 1
    for _ in range(remaining):
        out = model(
            decoder_input_ids=next_token,
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=past,
            return_dict=True,
        )
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past = out.past_key_values

    _device_sync(device)
    t2 = time.perf_counter()

    ttft_s = t1 - t0
    total_s = t2 - t0

    decode_s = max(t2 - t1, 1e-12)
    tps_decode = remaining / decode_s if remaining > 0 else float("inf")

    overall_s = max(total_s, 1e-12)
    tps_overall = output_tokens / overall_s

    return RunMetrics(ttft_s=ttft_s, total_s=total_s, tps_decode=tps_decode, tps_overall=tps_overall)


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark t5-base with torch.compile (TTFT + TPS).")
    p.add_argument("--model", default="t5-base", help="HF model id (default: t5-base)")
    p.add_argument("--prompt-len", type=int, required=True, help="Input prompt length in TOKENS")
    p.add_argument("--batch-size", type=int, required=True, help="Batch size")
    p.add_argument("--output-tokens", type=int, required=True, help="Number of output tokens to decode")

    compile_group = p.add_mutually_exclusive_group()
    compile_group.add_argument("--compile", action="store_true", help="Enable torch.compile")
    compile_group.add_argument("--no-compile", action="store_true", help="Disable torch.compile")

    p.add_argument("--compile-backend", default="inductor", help="torch.compile backend (default: inductor)")
    p.add_argument(
        "--compile-mode",
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode",
    )
    p.add_argument("--fullgraph", action="store_true", help="Pass fullgraph=True to torch.compile")
    p.add_argument("--dynamic", action="store_true", help="Pass dynamic=True to torch.compile")

    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device (default: auto)",
    )
    p.add_argument(
        "--dtype",
        default="bf16",
        choices=["fp32", "bf16", "fp16"],
        help="Model dtype (default: bf16; fp16 only recommended on cuda)",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")

    p.add_argument("--warmup", type=int, default=1, help="Warmup runs (default: 1)")
    p.add_argument("--runs", type=int, default=5, help="Measured runs (default: 5)")
    p.add_argument(
        "--include-compile-time",
        action="store_true",
        help="If set, do NOT discard the first compile run from metrics.",
    )

    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device=cuda requested, but CUDA is not available")

    dtype = _parse_dtype(args.dtype)

    # Make matmul precision explicit (helps CPU + reduces surprises on some platforms).
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # transformers>=4.57 deprecates torch_dtype= in favor of dtype=
    model = T5ForConditionalGeneration.from_pretrained(args.model, dtype=dtype)
    model.eval()
    model.to(device)

    encoder = model.get_encoder()

    decoder_start_token_id = model.config.decoder_start_token_id
    if decoder_start_token_id is None:
        if tokenizer.pad_token_id is None:
            raise SystemExit("decoder_start_token_id and tokenizer.pad_token_id are both None")
        decoder_start_token_id = int(tokenizer.pad_token_id)

    # torch.compile controls
    want_compile = args.compile or (not args.no_compile)
    if want_compile:
        if not hasattr(torch, "compile"):
            raise SystemExit("torch.compile not available in this torch version")
        encoder = torch.compile(
            encoder,
            backend=args.compile_backend,
            mode=args.compile_mode,
            fullgraph=args.fullgraph,
            dynamic=args.dynamic,
        )
        model = torch.compile(
            model,
            backend=args.compile_backend,
            mode=args.compile_mode,
            fullgraph=args.fullgraph,
            dynamic=args.dynamic,
        )

    input_ids, attention_mask = _make_synthetic_prompt_tokens(
        batch_size=args.batch_size,
        prompt_len=args.prompt_len,
        vocab_size=int(model.config.vocab_size),
        device=device,
        seed=args.seed,
    )

    # Warmup (also triggers compilation).
    warmup_runs = max(args.warmup, 0)
    measured_runs = max(args.runs, 1)

    warmup_metrics: list[RunMetrics] = []
    for _ in range(warmup_runs):
        warmup_metrics.append(
            _run_one(
                encoder=encoder,
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_start_token_id=int(decoder_start_token_id),
                output_tokens=args.output_tokens,
                device=device,
            )
        )

    results: list[RunMetrics] = []
    for _ in range(measured_runs):
        results.append(
            _run_one(
                encoder=encoder,
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_start_token_id=int(decoder_start_token_id),
                output_tokens=args.output_tokens,
                device=device,
            )
        )

    # Optionally include the first warmup as part of results (useful for end-to-end).
    if args.include_compile_time and warmup_metrics:
        results = warmup_metrics[:1] + results

    def _mean_std(vals: list[float]) -> tuple[float, float]:
        if len(vals) == 1:
            return vals[0], 0.0
        return statistics.mean(vals), statistics.stdev(vals)

    ttft = [m.ttft_s for m in results]
    total = [m.total_s for m in results]
    tps_d = [m.tps_decode for m in results]
    tps_o = [m.tps_overall for m in results]

    ttft_mean, ttft_std = _mean_std(ttft)
    total_mean, total_std = _mean_std(total)
    tpsd_mean, tpsd_std = _mean_std(tps_d)
    tpso_mean, tpso_std = _mean_std(tps_o)

    print("=== T5 benchmark ===")
    print(f"model: {args.model}")
    print(f"device: {device}")
    print(f"dtype: {dtype}")
    print(f"compile: {want_compile} (backend={args.compile_backend}, mode={args.compile_mode}, fullgraph={args.fullgraph}, dynamic={args.dynamic})")
    print(f"batch_size: {args.batch_size}")
    print(f"prompt_len(tokens): {args.prompt_len}")
    print(f"output_tokens: {args.output_tokens}")
    print(f"runs: {len(results)} (warmup={warmup_runs}, measured={measured_runs}, include_compile_time={args.include_compile_time})")
    print()
    print(f"TTFT (s): mean={ttft_mean:.6f} std={ttft_std:.6f}")
    print(f"Total (s): mean={total_mean:.6f} std={total_std:.6f}")
    print(f"Decode TPS (tokens/s, after first token): mean={tpsd_mean:.2f} std={tpsd_std:.2f}")
    print(f"Overall TPS (tokens/s, incl TTFT): mean={tpso_mean:.2f} std={tpso_std:.2f}")


if __name__ == "__main__":
    main()
