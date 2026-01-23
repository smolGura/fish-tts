#!/usr/bin/env python3
"""Example script demonstrating Fish-TTS usage.

Usage:
    # Basic synthesis (no voice cloning)
    uv run python scripts/example_synthesis.py --text "Hello world"

    # Voice cloning with audio + transcript
    uv run python scripts/example_synthesis.py --text "Hello" --reference audio.wav --transcript "Reference text"

    # Voice cloning with pre-encoded profile
    uv run python scripts/example_synthesis.py --text "Hello" --profile voice.npy --transcript "Reference text"

    # Streaming mode
    uv run python scripts/example_synthesis.py --text "Long text..." --stream
"""

import argparse
import io
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def convert_to_wav(audio_path: Path) -> bytes:
    """Convert audio file to WAV format using ffmpeg."""
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(audio_path),
                "-f",
                "wav",
                "-ar",
                "44100",
                "-ac",
                "1",
                "-acodec",
                "pcm_s16le",
                "-",
            ],
            capture_output=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error("Error converting audio: %s", e.stderr.decode())
        sys.exit(1)
    except FileNotFoundError:
        logger.error("ffmpeg not found. Install ffmpeg to convert non-WAV audio.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Fish-TTS synthesis example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic synthesis (default voice)
  %(prog)s --text "Hello world"

  # Voice cloning: encode reference audio on-the-fly
  %(prog)s --text "Nice to meet you" --reference voice.wav --transcript "Hello, this is my voice."

  # Voice cloning: use pre-encoded profile (.npy)
  %(prog)s --text "Nice to meet you" --profile voice.npy --transcript "Hello, this is my voice."

  # Streaming output
  %(prog)s --text "Long text here..." --stream --profile voice.npy --transcript "..."
""",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, this is a test of Fish TTS text to speech synthesis.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output.wav",
        help="Output WAV file path (default: output.wav)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Model precision (default: bf16)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Top-p sampling parameter (default: 0.8)",
    )

    # Voice cloning options
    ref_group = parser.add_argument_group("Voice Cloning")
    ref_group.add_argument(
        "--reference",
        "-r",
        type=Path,
        help="Reference audio file for voice cloning (WAV, MP3, etc.)",
    )
    ref_group.add_argument(
        "--profile",
        "-p",
        type=Path,
        help="Pre-encoded voice profile (.npy) - use instead of --reference",
    )
    ref_group.add_argument(
        "--transcript",
        "-t",
        type=str,
        help="Transcript of the reference audio (REQUIRED for voice cloning)",
    )

    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming mode",
    )

    args = parser.parse_args()

    # Validate voice cloning args
    if args.reference and args.profile:
        parser.error("Cannot use both --reference and --profile. Choose one.")

    if (args.reference or args.profile) and not args.transcript:
        parser.error("--transcript is required when using voice cloning")

    # Import here to show initialization time
    logger.info("Loading Fish-TTS...")
    from fish_tts import FishTTS, VoiceProfile

    # Initialize synthesizer
    logger.info("Initializing on %s (%s)...", args.device, args.precision)
    synth = FishTTS(device=args.device, precision=args.precision)
    logger.info("Ready.")
    logger.info("")

    # Prepare voice profile if provided
    references = None
    if args.profile:
        logger.info("Loading voice profile: %s", args.profile)
        profile = VoiceProfile.load(args.profile, text=args.transcript)
        references = [profile]
        logger.info("  Codes shape: %s", profile.codes.shape)
    elif args.reference:
        logger.info("Encoding reference audio: %s", args.reference)
        # Load and convert audio
        if args.reference.suffix.lower() == ".wav":
            audio_bytes = args.reference.read_bytes()
        else:
            logger.info("  Converting %s to WAV...", args.reference.suffix)
            audio_bytes = convert_to_wav(args.reference)

        profile = synth.encode_reference(audio_bytes, args.transcript)
        references = [profile]
        logger.info("  Codes shape: %s", profile.codes.shape)

    logger.info("")
    logger.info("Text: %s", args.text[:80] + "..." if len(args.text) > 80 else args.text)
    logger.info("Output: %s", args.output)
    logger.info("")

    # Synthesize
    if args.stream:
        logger.info("Streaming synthesis...")
        chunks = []
        for i, chunk in enumerate(
            synth.synthesize_stream(
                args.text,
                references=references,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        ):
            chunks.append(chunk)
            chunk_duration = len(chunk) / (44100 * 2)  # 16-bit mono
            logger.info("  Chunk %d: %.2fs audio", i + 1, chunk_duration)

        # Combine chunks into WAV
        total_pcm = b"".join(chunks)
        audio = pcm_to_wav(total_pcm, sample_rate=44100)
        logger.info("Total chunks: %d", len(chunks))
    else:
        logger.info("Synthesizing...")
        audio = synth.synthesize(
            args.text,
            references=references,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    # Save output
    output_path = Path(args.output)
    output_path.write_bytes(audio)

    audio_duration = (len(audio) - 44) / (44100 * 2)  # Subtract WAV header
    logger.info("")
    logger.info("Saved: %s (%.1fs audio, %.1f KB)", output_path, audio_duration, len(audio) / 1024)


def pcm_to_wav(pcm_data: bytes, sample_rate: int = 44100) -> bytes:
    """Convert raw PCM bytes to WAV format."""
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buffer.getvalue()


if __name__ == "__main__":
    main()
