#!/usr/bin/env python3
"""Test singleton pattern - model should only load once."""

import time

print("=" * 60)
print("Testing singleton pattern")
print("=" * 60)

# First call - should load and compile
print("\n[1] First get_instance() call...")
t0 = time.perf_counter()

from fish_tts import get_instance

synth1 = get_instance()
t1 = time.perf_counter()
print(f"First call: {t1 - t0:.1f}s")

# Second call - should be instant
print("\n[2] Second get_instance() call...")
t0 = time.perf_counter()
synth2 = get_instance()
t2 = time.perf_counter()
print(f"Second call: {t2 - t0:.4f}s")

# Verify same instance
print(f"\nSame instance: {synth1 is synth2}")

# Test synthesis
print("\n[3] Testing synthesis...")
t0 = time.perf_counter()
audio = synth1.synthesize("Hello, this is a test.")
t3 = time.perf_counter()
print(f"Synthesis: {t3 - t0:.2f}s")

# Second synthesis - should be fast (no recompile)
print("\n[4] Second synthesis...")
t0 = time.perf_counter()
audio2 = synth1.synthesize("Another test sentence.")
t4 = time.perf_counter()
print(f"Second synthesis: {t4 - t0:.2f}s")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"First get_instance():  {t1 - t0:.1f}s (load + compile + warmup)")
print(f"Second get_instance(): {t2 - t0:.4f}s (cached)")
print(f"Same instance:         {synth1 is synth2}")
