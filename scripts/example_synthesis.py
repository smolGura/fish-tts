#!/usr/bin/env python3
"""Example script demonstrating Fish-Speech ONNX usage.

Usage:
    uv run python scripts/example_synthesis.py
    uv run python scripts/example_synthesis.py --text "Hello world"
    uv run python scripts/example_synthesis.py --device cuda
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fish-Speech ONNX synthesis example")
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, this is a test of Fish Speech text to speech synthesis.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output WAV file path",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to model directory (auto-downloads if not specified)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Top-p (nucleus) sampling parameter",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Reference audio file for voice cloning (WAV format)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming mode",
    )

    args = parser.parse_args()

    # Import here to show initialization time
    logger.info("Importing Fish-Speech ONNX...")
    from fish_speech_onnx import FishSpeechONNX, SynthesisConfig

    # Initialize synthesizer
    logger.info("Initializing synthesizer on %s...", args.device)
    synth = FishSpeechONNX(
        model_dir=args.model_dir,
        device=args.device,
    )
    logger.info("Synthesizer ready. Model directory: %s", synth.model_dir)

    # Configure synthesis
    config = SynthesisConfig(
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Load reference audio if provided
    reference_audio = None
    if args.reference:
        logger.info("Loading reference audio: %s", args.reference)
        with open(args.reference, "rb") as f:
            reference_audio = f.read()

    # Synthesize
    logger.info("Synthesizing: %s", args.text[:50] + "..." if len(args.text) > 50 else args.text)

    if args.stream:
        # Streaming mode
        logger.info("Using streaming mode")
        all_chunks = []
        for chunk in synth.synthesize_stream(
            args.text,
            reference_audio=reference_audio,
            config=config,
        ):
            all_chunks.append(chunk)
            logger.info("Received chunk: %d bytes", len(chunk))

        # Combine chunks and save
        from fish_speech_onnx.utils.audio import AudioProcessor

        processor = AudioProcessor(sample_rate=synth.sample_rate)
        combined_pcm = b"".join(all_chunks)
        audio = processor.pcm_to_wav_bytes(combined_pcm, sample_rate=synth.sample_rate)
    else:
        # Normal mode
        audio = synth.synthesize(
            args.text,
            reference_audio=reference_audio,
            config=config,
        )

    # Save output
    output_path = Path(args.output)
    with open(output_path, "wb") as f:
        f.write(audio)

    logger.info("Audio saved to: %s", output_path)
    logger.info("Audio size: %.2f KB", len(audio) / 1024)


if __name__ == "__main__":
    main()
