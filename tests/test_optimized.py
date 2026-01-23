#!/usr/bin/env python3
"""Test optimized FishTTS performance."""

import time
import wave
import io

# Test initialization with compile and warmup
print("=" * 60)
print("Testing optimized FishTTS (compile=True, warmup=True)")
print("=" * 60)

t0 = time.perf_counter()
from fish_tts import FishTTS
synth = FishTTS(device="cuda", compile=True, warmup=True, precision="bf16")
t_init = time.perf_counter() - t0
print(f"\nInit time (with warmup): {t_init:.1f}s")

# Test batch synthesis
print("\n" + "=" * 60)
print("Testing batch synthesis")
print("=" * 60)

test_text = "Hello, this is a test of the optimized fish TTS system."

t0 = time.perf_counter()
audio = synth.synthesize(test_text)
t_synth = time.perf_counter() - t0

# Get audio duration
with wave.open(io.BytesIO(audio), "rb") as wf:
    duration = wf.getnframes() / wf.getframerate()

rtf = t_synth / duration
print(f"Text: {test_text}")
print(f"Synthesis time: {t_synth:.2f}s")
print(f"Audio duration: {duration:.2f}s")
print(f"RTF: {rtf:.2f}")

# Test streaming
print("\n" + "=" * 60)
print("Testing streaming synthesis")
print("=" * 60)

stream_text = "This is a longer text to test streaming synthesis. We want to measure the latency of the first audio chunk."

t0 = time.perf_counter()
first_chunk_time = None
chunk_count = 0
total_pcm_bytes = 0

for chunk in synth.synthesize_stream(stream_text):
    if first_chunk_time is None:
        first_chunk_time = time.perf_counter() - t0
        print(f"First chunk latency: {first_chunk_time:.3f}s")
    chunk_count += 1
    total_pcm_bytes += len(chunk)

t_total = time.perf_counter() - t0

# Calculate audio duration (16-bit mono 44100Hz)
stream_duration = total_pcm_bytes / (2 * 44100)
stream_rtf = t_total / stream_duration

print(f"Total chunks: {chunk_count}")
print(f"Total time: {t_total:.2f}s")
print(f"Audio duration: {stream_duration:.2f}s")
print(f"Stream RTF: {stream_rtf:.2f}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Init time:           {t_init:.1f}s")
print(f"Batch RTF:           {rtf:.2f}")
print(f"Stream RTF:          {stream_rtf:.2f}")
print(f"First chunk latency: {first_chunk_time:.3f}s")

# Save test audio
with open("/tmp/optimized_output.wav", "wb") as f:
    f.write(audio)
print(f"\nAudio saved to /tmp/optimized_output.wav")
