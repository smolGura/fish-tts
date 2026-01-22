#!/usr/bin/env python3
"""Detailed profiling to find optimization opportunities."""

import time
import torch

print("Loading model...")
from fish_tts import get_instance
synth = get_instance()

print("\n" + "=" * 60)
print("Detailed Profiling")
print("=" * 60)

# Profile individual components
from fish_speech.models.text2semantic.inference import generate_long

text = "Hello, this is a test of the text to speech system."

# 1. Token generation timing
print("\n[1] Token generation...")
prompt_text, prompt_tokens = synth._get_prompt_data(None)

torch.cuda.synchronize()
t0 = time.perf_counter()

codes_list = []
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
        codes_list.append(response.codes)
    elif response.action == "next":
        break

torch.cuda.synchronize()
t_gen = time.perf_counter() - t0
total_tokens = sum(c.shape[1] for c in codes_list)
print(f"    Time: {t_gen*1000:.1f}ms for {total_tokens} tokens")
print(f"    Speed: {total_tokens/t_gen:.1f} tok/s")

# 2. Vocoder timing
print("\n[2] Vocoder decode...")
codes = torch.cat(codes_list, dim=1)

# Multiple runs to get average
times = []
for _ in range(5):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    audio = synth._decode_codes(codes)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)

t_voc = sum(times) / len(times)
audio_duration = len(audio) / 44100
print(f"    Time: {t_voc*1000:.1f}ms (avg of 5)")
print(f"    Audio: {audio_duration:.2f}s")
print(f"    Vocoder RTF: {t_voc/audio_duration:.3f}")

# 3. Check vocoder dtype
print("\n[3] Vocoder info...")
print(f"    Vocoder dtype: {next(synth._vocoder.parameters()).dtype}")
print(f"    Vocoder device: {next(synth._vocoder.parameters()).device}")

# 4. Check if vocoder is compiled
print(f"    Vocoder compiled: {hasattr(synth._vocoder, '_orig_mod')}")

# Summary
print("\n" + "=" * 60)
print("BREAKDOWN")
print("=" * 60)
total = t_gen + t_voc
print(f"Token generation: {t_gen*1000:.1f}ms ({t_gen/total*100:.1f}%)")
print(f"Vocoder decode:   {t_voc*1000:.1f}ms ({t_voc/total*100:.1f}%)")
print(f"Total:            {total*1000:.1f}ms")
print(f"Audio duration:   {audio_duration:.2f}s")
print(f"Overall RTF:      {total/audio_duration:.3f}")
