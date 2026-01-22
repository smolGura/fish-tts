#!/usr/bin/env python3
"""Test synthesis with reference audio."""

import io
import subprocess
import tempfile
import time
import wave
from pathlib import Path


def convert_mp3_to_wav(mp3_path: str) -> bytes:
    """Convert MP3 to WAV bytes using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3_path, "-ar", "44100", "-ac", "1", wav_path],
        capture_output=True,
        check=True,
    )

    with open(wav_path, "rb") as f:
        wav_bytes = f.read()

    Path(wav_path).unlink()
    return wav_bytes


def main():
    from fish_tts import FishTTS, VoiceProfile

    # Paths
    mp3_path = "/home/progcat/Desktop/smolGura/wake-up.mp3"
    txt_path = "/home/progcat/Desktop/smolGura/ref-transcribe.txt"
    output_path = "/home/progcat/Desktop/smolGura/packages/fish-speech-onnx/test_output.wav"
    profile_path = "/home/progcat/Desktop/smolGura/packages/fish-speech-onnx/gura_voice.npy"

    # Read reference text
    with open(txt_path, "r") as f:
        ref_text = f.read().strip()

    print(f"Reference text: {ref_text[:100]}...")
    print()

    # === Convert MP3 to WAV ===
    print("=== Convert MP3 to WAV ===")
    t0 = time.perf_counter()
    wav_bytes = convert_mp3_to_wav(mp3_path)
    print(f"WAV size: {len(wav_bytes) / 1024:.1f} KB ({time.perf_counter() - t0:.2f}s)")
    print()

    # === Initialize ===
    print("=== Initialize ===")
    t0 = time.perf_counter()
    synth = FishTTS(device="cuda", compile=False, precision="bf16")
    print(f"Init time: {time.perf_counter() - t0:.1f}s")
    print(f"Compiled: {synth.is_compiled}, Precision: {synth.precision}")
    print()

    # === Encode reference ===
    print("=== Encode reference ===")
    t0 = time.perf_counter()
    profile = synth.encode_reference(wav_bytes, ref_text)
    print(f"Profile shape: {profile.codes.shape} ({time.perf_counter() - t0:.2f}s)")

    # Save for reuse
    profile.save(profile_path)
    print(f"Saved to: {profile_path}")
    print()

    # === Synthesize with voice cloning ===
    print("=== Synthesize ===")
    test_text = "Hello! Nice to meet you. How are you doing today?"
    print(f"Text: {test_text}")

    t0 = time.perf_counter()
    audio = synth.synthesize(test_text, references=[profile])
    elapsed = time.perf_counter() - t0

    # Get duration
    with wave.open(io.BytesIO(audio), "rb") as wf:
        duration = wf.getnframes() / wf.getframerate()

    print(f"Audio: {duration:.1f}s ({len(audio)/1024:.1f} KB) in {elapsed:.1f}s")
    print()

    # Save
    with open(output_path, "wb") as f:
        f.write(audio)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
