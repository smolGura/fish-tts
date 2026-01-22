"""Export Fish-Speech codec (DAC) to ONNX format with dynamic shape support.

Uses torch.export with dynamic shapes for proper variable-length audio handling.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch


def load_codec_model(checkpoint_path: str, device: str = "cpu"):
    """Load the codec model from checkpoint using hydra config."""
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    # Register eval resolver for hydra
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    print(f"Loading checkpoint from {checkpoint_path}")
    t0 = time.perf_counter()

    # Find fish_speech config directory
    import fish_speech.models.dac.modded_dac as dac_module

    fish_speech_path = Path(dac_module.__file__).parent.parent.parent
    config_dir = fish_speech_path / "configs"
    print(f"Using config dir: {config_dir}")

    # Clear any previous hydra state
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    # Initialize with absolute config dir path
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="modded_dac_vq")

    model = instantiate(cfg)

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Strip "generator." prefix if present
    if any("generator" in k for k in state_dict):
        state_dict = {
            k.replace("generator.", ""): v
            for k, v in state_dict.items()
            if "generator." in k
        }

    # Load weights
    result = model.load_state_dict(state_dict, strict=False, assign=True)
    print(f"Load result: {result}")

    model.eval()
    model.to(device)

    t1 = time.perf_counter()
    print(f"Model loaded in {t1 - t0:.2f}s")

    # Get n_codebooks from model
    n_codebooks = model.quantizer.quantizer.n_codebooks + 1  # +1 for semantic
    config = {"n_codebooks": n_codebooks}
    return model, config


class EncoderWrapper(torch.nn.Module):
    """Wrapper for exporting encoder to ONNX with dynamic shapes."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to indices.

        Args:
            audio: (batch, 1, samples) audio tensor

        Returns:
            indices: (batch, n_codebooks, seq_len) codebook indices
        """
        # Use tensor operation for audio length to preserve dynamic shape
        batch_size = audio.shape[0]
        audio_lengths = torch.full(
            (batch_size,), audio.shape[-1], dtype=torch.long, device=audio.device
        )

        # Encode
        indices, _ = self.model.encode(audio, audio_lengths)

        return indices


class DecoderWrapper(torch.nn.Module):
    """Wrapper for exporting decoder to ONNX with dynamic shapes."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode indices to audio.

        Args:
            indices: (batch, n_codebooks, seq_len) codebook indices

        Returns:
            audio: (batch, 1, samples) audio tensor
        """
        # Use tensor operation for feature length to preserve dynamic shape
        batch_size = indices.shape[0]
        feature_lengths = torch.full(
            (batch_size,), indices.shape[-1], dtype=torch.long, device=indices.device
        )

        # Decode
        audio, _ = self.model.decode(indices, feature_lengths)

        return audio


def export_encoder_dynamo(model, output_path: str, device: str = "cpu"):
    """Export encoder using dynamo-based export."""
    print("\n=== Exporting Encoder (dynamo) ===")

    wrapper = EncoderWrapper(model)
    wrapper.eval()

    # Create dummy input
    dummy_audio = torch.randn(1, 1, 44100, device=device)
    print(f"Input shape: {dummy_audio.shape}")

    # Test forward pass
    with torch.no_grad():
        output = wrapper(dummy_audio)
    print(f"Output shape: {output.shape}")

    # Define dynamic shapes
    batch = torch.export.Dim("batch", min=1, max=4)
    samples = torch.export.Dim("samples", min=1024, max=44100 * 60)  # Up to 60 seconds

    dynamic_shapes = {
        "audio": {0: batch, 2: samples},
    }

    # Export with dynamo
    print(f"Exporting to {output_path}")
    t0 = time.perf_counter()

    try:
        torch.onnx.export(
            wrapper,
            (dummy_audio,),
            output_path,
            input_names=["audio"],
            output_names=["indices"],
            dynamic_shapes=dynamic_shapes,
            opset_version=17,
            dynamo=True,
        )
        t1 = time.perf_counter()
        print(f"Export completed in {t1 - t0:.2f}s")
        return True
    except Exception as e:
        print(f"Dynamo export failed: {e}")
        return False


def export_encoder_legacy(model, output_path: str, device: str = "cpu"):
    """Export encoder using legacy TorchScript export."""
    print("\n=== Exporting Encoder (legacy) ===")

    wrapper = EncoderWrapper(model)
    wrapper.eval()

    # Create dummy input
    dummy_audio = torch.randn(1, 1, 44100, device=device)
    print(f"Input shape: {dummy_audio.shape}")

    # Test forward pass
    with torch.no_grad():
        output = wrapper(dummy_audio)
    print(f"Output shape: {output.shape}")

    # Export with legacy exporter
    print(f"Exporting to {output_path}")
    t0 = time.perf_counter()

    torch.onnx.export(
        wrapper,
        dummy_audio,
        output_path,
        input_names=["audio"],
        output_names=["indices"],
        dynamic_axes={
            "audio": {0: "batch", 2: "samples"},
            "indices": {0: "batch", 2: "seq_len"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    t1 = time.perf_counter()
    print(f"Export completed in {t1 - t0:.2f}s")

    # Verify with multiple input sizes
    verify_onnx_dynamic(output_path, wrapper, device)


def export_decoder_dynamo(model, output_path: str, device: str = "cpu"):
    """Export decoder using dynamo-based export."""
    print("\n=== Exporting Decoder (dynamo) ===")

    wrapper = DecoderWrapper(model)
    wrapper.eval()

    n_codebooks = model.quantizer.quantizer.n_codebooks + 1
    dummy_indices = torch.randint(0, 4096, (1, n_codebooks, 22), device=device)
    print(f"Input shape: {dummy_indices.shape}")

    # Test forward pass
    with torch.no_grad():
        output = wrapper(dummy_indices)
    print(f"Output shape: {output.shape}")

    # Define dynamic shapes
    batch = torch.export.Dim("batch", min=1, max=4)
    seq_len = torch.export.Dim("seq_len", min=1, max=2048)

    dynamic_shapes = {
        "indices": {0: batch, 2: seq_len},
    }

    # Export with dynamo
    print(f"Exporting to {output_path}")
    t0 = time.perf_counter()

    try:
        torch.onnx.export(
            wrapper,
            (dummy_indices,),
            output_path,
            input_names=["indices"],
            output_names=["audio"],
            dynamic_shapes=dynamic_shapes,
            opset_version=17,
            dynamo=True,
        )
        t1 = time.perf_counter()
        print(f"Export completed in {t1 - t0:.2f}s")
        return True
    except Exception as e:
        print(f"Dynamo export failed: {e}")
        return False


def export_decoder_legacy(model, output_path: str, device: str = "cpu"):
    """Export decoder using legacy TorchScript export."""
    print("\n=== Exporting Decoder (legacy) ===")

    wrapper = DecoderWrapper(model)
    wrapper.eval()

    n_codebooks = model.quantizer.quantizer.n_codebooks + 1
    dummy_indices = torch.randint(0, 4096, (1, n_codebooks, 22), device=device)
    print(f"Input shape: {dummy_indices.shape}")

    # Test forward pass
    with torch.no_grad():
        output = wrapper(dummy_indices)
    print(f"Output shape: {output.shape}")

    # Export with legacy exporter
    print(f"Exporting to {output_path}")
    t0 = time.perf_counter()

    torch.onnx.export(
        wrapper,
        dummy_indices,
        output_path,
        input_names=["indices"],
        output_names=["audio"],
        dynamic_axes={
            "indices": {0: "batch", 2: "seq_len"},
            "audio": {0: "batch", 2: "samples"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    t1 = time.perf_counter()
    print(f"Export completed in {t1 - t0:.2f}s")

    # Verify with multiple input sizes
    verify_onnx_dynamic(output_path, wrapper, device, is_decoder=True)


def verify_onnx_dynamic(
    onnx_path: str, wrapper: torch.nn.Module, device: str, is_decoder: bool = False
):
    """Verify ONNX model with multiple input sizes."""
    import onnxruntime as ort

    print("Verifying ONNX with multiple input sizes...")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    if is_decoder:
        # Test with different sequence lengths
        test_sizes = [10, 22, 50, 100]
        n_codebooks = 10

        for seq_len in test_sizes:
            dummy = torch.randint(0, 4096, (1, n_codebooks, seq_len), device=device)
            with torch.no_grad():
                expected = wrapper(dummy).cpu().numpy()

            onnx_output = session.run(None, {input_name: dummy.cpu().numpy()})[0]

            if onnx_output.shape[-1] == expected.shape[-1]:
                max_diff = np.max(np.abs(onnx_output - expected))
                print(f"  seq_len={seq_len}: shape OK, max_diff={max_diff:.6f}")
            else:
                print(
                    f"  seq_len={seq_len}: SHAPE MISMATCH - "
                    f"expected {expected.shape}, got {onnx_output.shape}"
                )
    else:
        # Test with different audio lengths
        test_sizes = [22050, 44100, 88200, 132300]  # 0.5s, 1s, 2s, 3s

        for samples in test_sizes:
            dummy = torch.randn(1, 1, samples, device=device)
            with torch.no_grad():
                expected = wrapper(dummy).cpu().numpy()

            onnx_output = session.run(None, {input_name: dummy.cpu().numpy()})[0]

            if onnx_output.shape[-1] == expected.shape[-1]:
                # For encoder, compare indices (integers)
                match = np.array_equal(onnx_output, expected)
                print(f"  samples={samples}: shape OK, exact_match={match}")
            else:
                print(
                    f"  samples={samples}: SHAPE MISMATCH - "
                    f"expected {expected.shape}, got {onnx_output.shape}"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Export Fish-Speech codec to ONNX with dynamic shapes"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/progcat/.cache/fish-speech-onnx/models/fishaudio--openaudio-s1-mini/codec.pth",
        help="Path to codec.pth checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/progcat/.cache/fish-speech-onnx/models/fishaudio--openaudio-s1-mini",
        help="Output directory for ONNX files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--use-dynamo",
        action="store_true",
        help="Use dynamo-based export (experimental)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, config = load_codec_model(args.checkpoint, args.device)
    print(f"Model config: n_codebooks={config.get('n_codebooks', 8)}")

    encoder_path = output_dir / "encoder.onnx"
    decoder_path = output_dir / "decoder.onnx"

    if args.use_dynamo:
        # Try dynamo export
        encoder_ok = export_encoder_dynamo(model, str(encoder_path), args.device)
        decoder_ok = export_decoder_dynamo(model, str(decoder_path), args.device)

        if not encoder_ok or not decoder_ok:
            print("\nDynamo export failed, falling back to legacy...")
            if not encoder_ok:
                export_encoder_legacy(model, str(encoder_path), args.device)
            if not decoder_ok:
                export_decoder_legacy(model, str(decoder_path), args.device)
    else:
        # Use legacy export
        export_encoder_legacy(model, str(encoder_path), args.device)
        export_decoder_legacy(model, str(decoder_path), args.device)

    print("\n=== Export Summary ===")
    if encoder_path.exists():
        print(
            f"Encoder: {encoder_path} ({encoder_path.stat().st_size / 1024 / 1024:.1f} MB)"
        )
    if decoder_path.exists():
        print(
            f"Decoder: {decoder_path} ({decoder_path.stat().st_size / 1024 / 1024:.1f} MB)"
        )


if __name__ == "__main__":
    main()
