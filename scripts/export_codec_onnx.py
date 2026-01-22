"""Export Fish-Speech codec (DAC) to ONNX format."""

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
    """Wrapper for exporting encoder to ONNX."""

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
        # Get audio length
        audio_lengths = torch.tensor([audio.shape[-1]], device=audio.device)

        # Encode
        indices, _ = self.model.encode(audio, audio_lengths)

        return indices


class DecoderWrapper(torch.nn.Module):
    """Wrapper for exporting decoder to ONNX."""

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
        # Get feature length
        feature_lengths = torch.tensor([indices.shape[-1]], device=indices.device)

        # Decode
        audio, _ = self.model.decode(indices, feature_lengths)

        return audio


def export_encoder(model, output_path: str, device: str = "cpu"):
    """Export encoder to ONNX."""
    print("\n=== Exporting Encoder ===")

    wrapper = EncoderWrapper(model)
    wrapper.eval()

    # Create dummy input: 1 second of audio at 44100 Hz
    dummy_audio = torch.randn(1, 1, 44100, device=device)

    print(f"Input shape: {dummy_audio.shape}")

    # Test forward pass
    with torch.no_grad():
        output = wrapper(dummy_audio)
    print(f"Output shape: {output.shape}")

    # Export
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
        dynamo=False,  # Use legacy exporter for stability
    )

    t1 = time.perf_counter()
    print(f"Export completed in {t1 - t0:.2f}s")

    # Verify
    verify_onnx(output_path, dummy_audio.cpu().numpy(), output.cpu().numpy())


def export_decoder(model, output_path: str, device: str = "cpu"):
    """Export decoder to ONNX."""
    print("\n=== Exporting Decoder ===")

    wrapper = DecoderWrapper(model)
    wrapper.eval()

    # Create dummy input: indices for ~1 second of audio
    # With encoder_rates [2,4,8,8] = 512x compression, then 4x downsample in quantizer
    # 44100 / (512 * 4) = ~22 frames
    # n_codebooks = 1 (semantic) + 9 (residual) = 10
    n_codebooks = model.quantizer.quantizer.n_codebooks + 1  # +1 for semantic
    dummy_indices = torch.randint(0, 4096, (1, n_codebooks, 22), device=device)

    print(f"Input shape: {dummy_indices.shape}")

    # Test forward pass
    with torch.no_grad():
        output = wrapper(dummy_indices)
    print(f"Output shape: {output.shape}")

    # Export
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
        dynamo=False,  # Use legacy exporter for stability
    )

    t1 = time.perf_counter()
    print(f"Export completed in {t1 - t0:.2f}s")

    # Verify
    verify_onnx(output_path, dummy_indices.cpu().numpy(), output.cpu().numpy())


def verify_onnx(onnx_path: str, input_data: np.ndarray, expected_output: np.ndarray):
    """Verify ONNX model produces correct output."""
    import onnxruntime as ort

    print("Verifying ONNX output...")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: input_data})[0]

    # Compare
    if np.allclose(onnx_output, expected_output, rtol=1e-3, atol=1e-5):
        print("Verification PASSED")
    else:
        max_diff = np.max(np.abs(onnx_output - expected_output))
        print(f"Verification FAILED - max diff: {max_diff}")


def main():
    parser = argparse.ArgumentParser(description="Export Fish-Speech codec to ONNX")
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
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, config = load_codec_model(args.checkpoint, args.device)
    print(f"Model config: n_codebooks={config.get('n_codebooks', 8)}")

    # Export encoder
    encoder_path = output_dir / "encoder.onnx"
    export_encoder(model, str(encoder_path), args.device)

    # Export decoder
    decoder_path = output_dir / "decoder.onnx"
    export_decoder(model, str(decoder_path), args.device)

    print("\n=== Export Summary ===")
    print(f"Encoder: {encoder_path} ({encoder_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"Decoder: {decoder_path} ({decoder_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
