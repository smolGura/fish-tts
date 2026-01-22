#!/usr/bin/env python3
"""Test new features: prefill, pipeline, dynamic references."""

import time

print("Loading model...")
from fish_tts import get_instance, VoiceProfile
synth = get_instance()

print("\n" + "=" * 60)
print("Testing new features")
print("=" * 60)

# Test 1: Basic synthesis (baseline)
print("\n[1] Baseline synthesis (no references)...")
t0 = time.perf_counter()
audio = synth.synthesize("Hello, this is a test.")
t1 = time.perf_counter()
print(f"    Time: {t1-t0:.3f}s")

# Test 2: Dynamic reference management
print("\n[2] Testing dynamic reference management...")

# Create a dummy profile (normally from encode_reference)
import numpy as np
dummy_codes = np.random.randint(0, 4096, (10, 50), dtype=np.int64)
profile1 = VoiceProfile(codes=dummy_codes, text="Reference one", name="voice1")
profile2 = VoiceProfile(codes=dummy_codes, text="Reference two", name="voice2")

# Add references
synth.add_reference(profile1)
print(f"    After add_reference(profile1): {synth.num_references} reference(s)")

synth.add_reference(profile2)
print(f"    After add_reference(profile2): {synth.num_references} reference(s)")

# Get references
refs = synth.get_references()
print(f"    get_references(): {[r.name for r in refs]}")

# Clear references
synth.clear_references()
print(f"    After clear_references(): {synth.num_references} reference(s)")

# Set references
synth.set_references([profile1, profile2])
print(f"    After set_references([p1, p2]): {synth.num_references} reference(s)")

# Test 3: Synthesis with prefilled references
print("\n[3] Synthesis with prefilled references...")
t0 = time.perf_counter()
audio = synth.synthesize("Hello with prefilled references.")
t1 = time.perf_counter()
print(f"    Time: {t1-t0:.3f}s")

# Test 4: Pipeline vs simple streaming
print("\n[4] Comparing pipeline vs simple streaming...")

test_text = "This is a longer sentence to test the streaming synthesis with pipeline optimization."

# Simple streaming (no pipeline)
print("    [4a] Simple streaming (pipeline=False)...")
t0 = time.perf_counter()
chunks_simple = []
first_chunk_time_simple = None
for chunk in synth.synthesize_stream(test_text, pipeline=False):
    if first_chunk_time_simple is None:
        first_chunk_time_simple = time.perf_counter() - t0
    chunks_simple.append(chunk)
t_simple = time.perf_counter() - t0
print(f"         First chunk: {first_chunk_time_simple:.3f}s, Total: {t_simple:.3f}s, Chunks: {len(chunks_simple)}")

# Pipeline streaming
print("    [4b] Pipeline streaming (pipeline=True)...")
t0 = time.perf_counter()
chunks_pipeline = []
first_chunk_time_pipeline = None
for chunk in synth.synthesize_stream(test_text, pipeline=True):
    if first_chunk_time_pipeline is None:
        first_chunk_time_pipeline = time.perf_counter() - t0
    chunks_pipeline.append(chunk)
t_pipeline = time.perf_counter() - t0
print(f"         First chunk: {first_chunk_time_pipeline:.3f}s, Total: {t_pipeline:.3f}s, Chunks: {len(chunks_pipeline)}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Simple streaming:   first={first_chunk_time_simple:.3f}s, total={t_simple:.3f}s")
print(f"Pipeline streaming: first={first_chunk_time_pipeline:.3f}s, total={t_pipeline:.3f}s")
if t_simple > 0:
    speedup = (t_simple - t_pipeline) / t_simple * 100
    print(f"Pipeline speedup:   {speedup:.1f}%")

# Clear for next run
synth.clear_references()
