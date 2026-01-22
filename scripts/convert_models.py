#!/usr/bin/env python3
"""Model conversion script for Fish-Speech ONNX.

This script converts Fish-Speech models to ONNX format.

Usage:
    uv run python scripts/convert_models.py --input-dir ./models --output-dir ./models/onnx
    uv run python scripts/convert_models.py --vocoder-only
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_vocoder(
    input_path: Path,
    output_path: Path,
    opset_version: int = 17,
) -> None:
    """Convert vocoder to ONNX.

    Args:
        input_path: Path to codec.pth file.
        output_path: Path for output ONNX file.
        opset_version: ONNX opset version.
    """
    from fish_speech_onnx.convert import export_vocoder

    logger.info("Converting vocoder: %s -> %s", input_path, output_path)

    export_vocoder(
        pytorch_model_path=input_path,
        output_path=output_path,
        opset_version=opset_version,
    )

    logger.info("Vocoder conversion complete")


def optimize_model(
    input_path: Path,
    output_path: Path | None = None,
    quantize: bool = False,
    fp16: bool = False,
) -> None:
    """Optimize ONNX model.

    Args:
        input_path: Path to ONNX model.
        output_path: Path for optimized model.
        quantize: Whether to apply INT8 quantization.
        fp16: Whether to convert to FP16.
    """
    from fish_speech_onnx.convert import optimize_onnx_model, quantize_model
    from fish_speech_onnx.convert.optimize import convert_to_fp16, simplify_model

    logger.info("Optimizing model: %s", input_path)

    # Simplify first
    simplified = simplify_model(input_path, output_path)

    # Apply FP16 if requested
    if fp16:
        fp16_path = simplified.with_suffix(".fp16.onnx")
        convert_to_fp16(simplified, fp16_path)
        simplified = fp16_path

    # Quantize if requested
    if quantize:
        quantized_path = simplified.with_suffix(".int8.onnx")
        quantize_model(simplified, quantized_path)

    logger.info("Optimization complete")


def main():
    parser = argparse.ArgumentParser(description="Convert Fish-Speech models to ONNX")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./models",
        help="Input directory containing PyTorch models",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for ONNX models (default: same as input)",
    )
    parser.add_argument(
        "--vocoder-only",
        action="store_true",
        help="Only convert vocoder (recommended for Phase 1)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply INT8 quantization",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Convert to FP16 precision",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find codec.pth
    codec_path = input_dir / "codec.pth"
    if not codec_path.exists():
        # Try to find in subdirectories
        codec_files = list(input_dir.rglob("codec.pth"))
        if codec_files:
            codec_path = codec_files[0]
        else:
            logger.error("codec.pth not found in %s", input_dir)
            return

    # Convert vocoder
    vocoder_output = output_dir / "vocoder.onnx"
    convert_vocoder(
        input_path=codec_path,
        output_path=vocoder_output,
        opset_version=args.opset,
    )

    # Optimize
    if args.quantize or args.fp16:
        optimize_model(
            input_path=vocoder_output,
            quantize=args.quantize,
            fp16=args.fp16,
        )

    if not args.vocoder_only:
        logger.warning(
            "Transformer ONNX conversion is not yet supported due to known issues. "
            "See https://github.com/fishaudio/fish-speech/issues/903"
        )

    logger.info("Conversion complete. Output directory: %s", output_dir)


if __name__ == "__main__":
    main()
