#!/usr/bin/env python3
"""Benchmark script for Fish-TTS with detailed profiling."""

import argparse
import time

import torch


def benchmark(device: str = "cuda", precision: str = "bf16", profile: bool = False):
    """Run benchmark with optional profiling."""
    from fish_tts import FishTTS

    def get_vram():
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0

    print("=" * 60)
    print("Fish-TTS Benchmark")
    print("=" * 60)
    print()

    # Initialize
    print(f"Device: {device}, Precision: {precision}")
    print("torch.compile: Always enabled")
    print()
    print("Initializing...")
    t0 = time.perf_counter()
    synth = FishTTS(device=device, precision=precision)
    init_time = time.perf_counter() - t0
    print(f"Init time: {init_time:.1f}s")
    print(f"VRAM: {get_vram():.2f} GB")
    print()

    # Warmup
    print("Warming up...")
    _ = synth.synthesize("Test")
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    print()

    # Benchmark
    tests = [
        ("Short", "Hello world!"),
        ("Medium", "The quick brown fox jumps over the lazy dog."),
        (
            "Long",
            "In a world where technology advances rapidly, artificial intelligence "
            "has emerged as a transformative force reshaping how we live and work.",
        ),
    ]

    print("Synthesis Benchmark:")
    print("-" * 60)

    total_audio = 0
    total_time = 0

    for name, text in tests:
        t0 = time.perf_counter()
        audio = synth.synthesize(text)
        elapsed = time.perf_counter() - t0

        # Calculate audio duration (44100 Hz, 16-bit mono, subtract WAV header)
        audio_duration = (len(audio) - 44) / (44100 * 2)
        rtf = elapsed / audio_duration if audio_duration > 0 else 0

        total_audio += audio_duration
        total_time += elapsed

        print(f"{name:8s}: {len(text):3d} chars -> {audio_duration:5.1f}s audio in {elapsed:5.2f}s (RTF={rtf:.3f})")

    avg_rtf = total_time / total_audio if total_audio > 0 else 0
    print("-" * 60)
    print(f"Average RTF: {avg_rtf:.3f}")
    print()

    # Detailed profiling if requested
    if profile:
        print("Detailed Profiling (Long text):")
        print("-" * 60)
        profile_synthesis(synth, tests[2][1])
        print()

    # Streaming benchmark
    print("Streaming Benchmark:")
    print("-" * 60)

    text = tests[2][1]  # Use long text
    chunks = []
    t0 = time.perf_counter()
    first_chunk_time = None

    for chunk in synth.synthesize_stream(text):
        if first_chunk_time is None:
            first_chunk_time = time.perf_counter() - t0
        chunks.append(chunk)

    total_stream_time = time.perf_counter() - t0
    total_pcm = sum(len(c) for c in chunks)
    audio_duration = total_pcm / (44100 * 2)  # 16-bit mono
    rtf = total_stream_time / audio_duration if audio_duration > 0 else 0

    print(f"First chunk: {first_chunk_time:.3f}s")
    print(f"Total: {audio_duration:.1f}s audio in {total_stream_time:.2f}s (RTF={rtf:.3f})")
    print(f"Chunks: {len(chunks)}")
    print()

    print("=" * 60)
    if torch.cuda.is_available():
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


def profile_synthesis(synth, text: str):
    """Profile synthesis to find bottlenecks."""
    from fish_tts.models.inference import generate_long

    # Profile token generation
    prompt_text, prompt_tokens = synth._get_prompt_data(None)

    token_times = []
    vocoder_time = 0
    codes_list = []

    t_gen_start = time.perf_counter()
    token_count = 0

    for response in generate_long(
        model=synth._model,
        device=synth.device,
        decode_one_token=synth._decode_one_token,
        text=text,
        max_new_tokens=2048,
        temperature=0.7,
        top_p=0.8,
        repetition_penalty=1.1,
        prompt_text=prompt_text,
        prompt_tokens=prompt_tokens,
    ):
        if response.action == "sample":
            t_now = time.perf_counter()
            if token_times:
                token_times.append(t_now - token_times[-1])
            else:
                token_times.append(t_now - t_gen_start)
            codes_list.append(response.codes)
            token_count += response.codes.shape[1]
        elif response.action == "next":
            break

    t_gen_end = time.perf_counter()
    gen_time = t_gen_end - t_gen_start

    # Profile vocoder
    if codes_list:
        codes = torch.cat(codes_list, dim=1)
        t_voc_start = time.perf_counter()
        _ = synth._decode_codes(codes)
        vocoder_time = time.perf_counter() - t_voc_start

    total_time = gen_time + vocoder_time
    tok_per_sec = token_count / gen_time if gen_time > 0 else 0

    print(f"Token generation: {gen_time:.2f}s ({token_count} tokens, {tok_per_sec:.1f} tok/s)")
    print(f"Vocoder decode:   {vocoder_time:.2f}s")
    print(f"Total:            {total_time:.2f}s")
    print(f"Gen/Total ratio:  {gen_time/total_time*100:.1f}%")

    if len(token_times) > 10:
        avg_token_time = sum(token_times[5:]) / len(token_times[5:])  # Skip first 5
        print(f"Avg token time:   {avg_token_time*1000:.1f}ms (after warmup)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Fish-TTS")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--profile", action="store_true", help="Enable detailed profiling")
    args = parser.parse_args()

    benchmark(device=args.device, precision=args.precision, profile=args.profile)


if __name__ == "__main__":
    main()
