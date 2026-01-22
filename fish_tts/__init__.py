"""Fish-TTS: Standalone TTS using DualARTransformer and DAC vocoder.

Features:
- Singleton pattern: model loaded once per process
- torch.compile (mandatory): ~120 tokens/sec, RTF ~0.26
- Prefilled references: set voice profiles once, reuse across calls
- Pipeline streaming: parallel generation and decoding (~18% faster)
- Dynamic references: add/remove voice profiles at runtime
- No external fish_speech dependency

Usage:
    from fish_tts import get_instance, VoiceProfile

    # Get singleton instance (first call loads and compiles model)
    synth = get_instance()

    # Basic synthesis
    audio = synth.synthesize("Hello world")

    # Voice cloning with prefilled references
    profile = VoiceProfile.load("voice.npy", text="reference transcript")
    synth.set_references([profile])  # Set once
    audio = synth.synthesize("Text to speak")  # Uses prefilled references

    # Dynamic reference management
    synth.add_reference(another_profile)
    synth.clear_references()

    # Streaming (with pipeline parallelization)
    for chunk in synth.synthesize_stream("Long text..."):
        play_audio(chunk)
"""

from fish_tts.synthesizer import FishTTS, VoiceProfile, get_instance, reset_instance

__version__ = "0.6.0"
__all__ = ["FishTTS", "VoiceProfile", "get_instance", "reset_instance"]
