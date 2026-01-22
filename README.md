# Fish-Speech ONNX

Portable and fast TTS inference using Fish-Speech with ONNX acceleration.

## Features

- **Standalone package**: Can be installed and used independently
- **Portability first**: Minimal dependencies, cross-platform support
- **Fast inference**: ONNX acceleration with GPU/CPU support
- **Automatic model download**: Models downloaded from HuggingFace Hub on first use

## Installation

```bash
# Basic installation (CPU only)
pip install fish-speech-onnx

# With CUDA support
pip install fish-speech-onnx[cuda]

# For model conversion
pip install fish-speech-onnx[convert]
```

Or using uv:

```bash
uv add fish-speech-onnx
```

## Usage

### Basic Synthesis

```python
from fish_speech_onnx import FishSpeechONNX, SynthesisConfig

# Initialize (models auto-downloaded to ~/.cache/fish-speech-onnx/)
synth = FishSpeechONNX(device="cuda")  # or "cpu"

# Basic synthesis
audio = synth.synthesize("Hello, world!")
with open("output.wav", "wb") as f:
    f.write(audio)

# Custom configuration
config = SynthesisConfig(temperature=0.5, top_p=0.9)
audio = synth.synthesize("Custom config", config=config)
```

### Streaming Synthesis

```python
# Low-latency streaming
for chunk in synth.synthesize_stream("Streaming output"):
    play_audio(chunk)  # Play in real-time
```

### Voice Cloning

```python
# Use reference audio for voice cloning
with open("reference.wav", "rb") as f:
    ref_audio = f.read()
audio = synth.synthesize("Clone voice", reference_audio=ref_audio)
```

### Custom Model Directory

```python
# Specify model directory
synth = FishSpeechONNX(
    model_dir="./my_models",  # Downloads here if not exists
    device="cpu",
)
```

## Architecture

Phase 1 (Current): Hybrid mode
- Vocoder: ONNX (fast inference)
- Transformer: PyTorch (compatibility)

Phase 2 (Future): Full ONNX
- Complete ONNX conversion for maximum portability

## License

CC-BY-NC-SA-4.0

This project uses models from [Fish-Speech](https://github.com/fishaudio/fish-speech).

## References

- [Fish-Speech](https://github.com/fishaudio/fish-speech)
- [Fish-Speech Paper](https://arxiv.org/html/2411.01156v1)
