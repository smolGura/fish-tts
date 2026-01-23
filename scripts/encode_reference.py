#!/usr/bin/env python3
"""Encode reference audio into a voice profile (.npy) for voice cloning."""

import argparse
import io
import subprocess
import sys
from pathlib import Path


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
        print(f"Error converting audio: {e.stderr.decode()}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(
            "Error: ffmpeg not found. Please install ffmpeg to convert non-WAV audio.",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Encode reference audio into a voice profile for voice cloning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Encode with transcript text
  %(prog)s reference.wav "Hello, this is the reference transcript." -o voice.npy

  # Encode with transcript from file
  %(prog)s reference.wav -t transcript.txt -o voice.npy

  # Auto-generate output name from input
  %(prog)s reference.wav "Hello world"
  # -> saves to reference.npy
""",
    )
    parser.add_argument(
        "audio", type=Path, help="Input audio file (WAV, MP3, FLAC, etc.)"
    )
    parser.add_argument("transcript", nargs="?", help="Reference transcript text")
    parser.add_argument(
        "-t",
        "--transcript-file",
        type=Path,
        help="Read transcript from file instead of argument",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output .npy file path (default: same name as audio with .npy extension)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--precision",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Model precision (default: bf16)",
    )
    args = parser.parse_args()

    # Validate inputs
    if not args.audio.exists():
        print(f"Error: Audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    # Get transcript
    if args.transcript_file:
        if not args.transcript_file.exists():
            print(
                f"Error: Transcript file not found: {args.transcript_file}",
                file=sys.stderr,
            )
            sys.exit(1)
        transcript = args.transcript_file.read_text(encoding="utf-8").strip()
    elif args.transcript:
        transcript = args.transcript
    else:
        print(
            "Error: Must provide transcript either as argument or via --transcript-file",
            file=sys.stderr,
        )
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = args.audio.with_suffix(".npy")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Audio:      {args.audio}")
    print(f"Transcript: {transcript[:50]}{'...' if len(transcript) > 50 else ''}")
    print(f"Output:     {output_path}")
    print()

    # Load model and encode
    print("Loading Fish-TTS model...")
    from fish_tts import FishTTS

    synth = FishTTS(device=args.device, precision=args.precision)

    print("Encoding reference audio...")
    # Convert to WAV if needed
    if args.audio.suffix.lower() == ".wav":
        audio_bytes = args.audio.read_bytes()
    else:
        print(f"  Converting {args.audio.suffix} to WAV...")
        audio_bytes = convert_to_wav(args.audio)
    profile = synth.encode_reference(audio_bytes, transcript)

    # Save profile
    profile.save(output_path)
    print(f"Saved voice profile to: {output_path}")
    print(f"  Codes shape: {profile.codes.shape}")
    print()
    print("Usage:")
    print(f'  profile = VoiceProfile.load("{output_path}", text="{transcript[:30]}...")')
    print("  audio = synth.synthesize(text, references=[profile])")


if __name__ == "__main__":
    main()
