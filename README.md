# fish-tts

Lightweight TTS using DualARTransformer and DAC vocoder, based on [Fish-Speech](https://github.com/fishaudio/fish-speech).

## Features

- **Standalone package**: No external fish_speech dependency
- **Singleton pattern**: Model loaded once per process
- **torch.compile**: ~120 tokens/sec, RTF ~0.26
- **Prefilled references**: Set voice profiles once, reuse across calls
- **Pipeline streaming**: Parallel generation and decoding (~18% faster)
- **Dynamic references**: Add/remove voice profiles at runtime

## Installation

```bash
uv add git+https://github.com/smolGura/fish-tts
```

## Usage

### Basic Synthesis

```python
from fish_tts import get_instance

# Get singleton instance (first call loads and compiles model)
synth = get_instance()

# Basic synthesis
audio = synth.synthesize("Hello world")
```

### Voice Cloning

```python
from fish_tts import get_instance, VoiceProfile

synth = get_instance()

# Load voice profile
profile = VoiceProfile.load("voice.npy", text="reference transcript")

# Set reference (prefilled for efficiency)
synth.set_references([profile])

# Synthesize with cloned voice
audio = synth.synthesize("Text to speak")
```

### Streaming

```python
# Pipeline streaming for low latency
for chunk in synth.synthesize_stream("Long text to speak..."):
    play_audio(chunk)
```

### Dynamic Reference Management

```python
# Add/remove voice profiles at runtime
synth.add_reference(another_profile)
synth.clear_references()
```

## Performance

| Mode | Speed | RTF |
|------|-------|-----|
| torch.compile | ~120 tokens/sec | ~0.26 |
| Pipeline streaming | +18% faster | - |

## License

CC-BY-NC-SA-4.0

## References

- [Fish-Speech](https://github.com/fishaudio/fish-speech)
- [Fish-Speech Paper](https://arxiv.org/html/2411.01156v1)
