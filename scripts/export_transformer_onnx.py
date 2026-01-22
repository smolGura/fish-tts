"""Export Fish-Speech DualARTransformer to ONNX format.

Strategy: Export forward pass only, keep KV cache management in Python.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def load_transformer_model(model_dir: str, device: str = "cpu"):
    """Load the DualARTransformer model."""
    from fish_speech.models.text2semantic.llama import DualARTransformer

    print(f"Loading transformer from {model_dir}")
    t0 = time.perf_counter()

    model = DualARTransformer.from_pretrained(
        path=model_dir,
        load_weights=True,
        lora_config=None,
    )
    model.to(device)
    model.eval()

    t1 = time.perf_counter()
    print(f"Model loaded in {t1 - t0:.2f}s")
    print(f"Config: dim={model.config.dim}, n_layer={model.config.n_layer}")

    return model


class SlowModelPrefillWrapper(nn.Module):
    """Wrapper for slow model prefill (processing full prompt without KV cache)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for prefill.

        Args:
            input_ids: (batch, seq_len) input token IDs

        Returns:
            logits: (batch, seq_len, vocab_size) output logits
            hidden_states: (batch, seq_len, dim) hidden states for fast model
        """
        # Get embeddings
        x = self.model.embed(input_ids)

        # Process through slow layers (without KV cache for prefill)
        for layer in self.model.layers:
            x = layer(x, input_pos=None)

        # Final norm and output
        x = self.model.norm(x)
        logits = self.model.output(x)

        return logits, x


class SlowModelDecodeWrapper(nn.Module):
    """Wrapper for slow model single-token decode with external KV cache."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.n_layer = model.config.n_layer
        self.n_local_heads = model.config.n_local_heads
        self.head_dim = model.config.head_dim

    def forward(
        self,
        input_ids: torch.Tensor,
        input_pos: torch.Tensor,
        k_caches: torch.Tensor,
        v_caches: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for single token decode.

        Args:
            input_ids: (batch, 1) single token
            input_pos: (1,) position index
            k_caches: (n_layer, batch, n_local_heads, max_seq, head_dim)
            v_caches: (n_layer, batch, n_local_heads, max_seq, head_dim)

        Returns:
            logits: (batch, 1, vocab_size)
            hidden_states: (batch, 1, dim)
            new_k_caches: updated K caches
            new_v_caches: updated V caches
        """
        # Get embeddings
        x = self.model.embed(input_ids)

        # Process through slow layers with KV cache
        new_k_list = []
        new_v_list = []

        for i, layer in enumerate(self.model.layers):
            # Get layer's K/V cache
            k_cache = k_caches[i]
            v_cache = v_caches[i]

            # Forward with attention and cache update
            x, new_k, new_v = self._forward_layer_with_cache(
                layer, x, input_pos, k_cache, v_cache
            )

            new_k_list.append(new_k)
            new_v_list.append(new_v)

        # Final norm and output
        x = self.model.norm(x)
        logits = self.model.output(x)

        new_k_caches = torch.stack(new_k_list, dim=0)
        new_v_caches = torch.stack(new_v_list, dim=0)

        return logits, x, new_k_caches, new_v_caches

    def _forward_layer_with_cache(
        self,
        layer,
        x: torch.Tensor,
        input_pos: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward single layer with external KV cache."""
        # This is simplified - actual implementation needs to match layer internals
        # For now, this is a placeholder
        raise NotImplementedError("Layer-level cache management needs custom implementation")


class FastModelWrapper(nn.Module):
    """Wrapper for fast model (codebook prediction)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        hidden_states: torch.Tensor,
        codebook_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for fast model.

        Args:
            hidden_states: (batch, dim) from slow model or previous codebook
            codebook_idx: (1,) codebook position index

        Returns:
            logits: (batch, 1, codebook_size) output logits
        """
        # Process through fast layers
        x = hidden_states.unsqueeze(1) if hidden_states.dim() == 2 else hidden_states

        for layer in self.model.fast_layers:
            x = layer(x, input_pos=codebook_idx)

        x = self.model.fast_norm(x)
        logits = self.model.fast_output(x)

        return logits


def analyze_model_for_export(model):
    """Analyze model to understand export requirements."""
    print("\n=== Model Analysis for ONNX Export ===")

    config = model.config
    print(f"\nSlow Model:")
    print(f"  - Layers: {config.n_layer}")
    print(f"  - Dim: {config.dim}")
    print(f"  - Heads: {config.n_head} (Q), {config.n_local_heads} (K/V)")
    print(f"  - Head dim: {config.head_dim}")
    print(f"  - Vocab size: {config.vocab_size}")

    print(f"\nFast Model:")
    print(f"  - Layers: {config.n_fast_layer}")
    print(f"  - Dim: {config.fast_dim}")
    print(f"  - Codebook size: {config.codebook_size}")
    print(f"  - Num codebooks: {config.num_codebooks}")

    # Calculate KV cache size per layer
    # Shape: (batch, n_local_heads, max_seq, head_dim)
    kv_cache_per_layer = config.n_local_heads * config.max_seq_len * config.head_dim * 2  # K + V
    total_kv_cache = kv_cache_per_layer * config.n_layer * 4  # float32 bytes

    print(f"\nKV Cache Requirements:")
    print(f"  - Per layer: {kv_cache_per_layer * 4 / 1024 / 1024:.1f} MB (float32)")
    print(f"  - Total ({config.n_layer} layers): {total_kv_cache / 1024 / 1024:.1f} MB")
    print(f"  - Max sequence: {config.max_seq_len}")

    # Estimate model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Size:")
    print(f"  - Parameters: {total_params / 1e6:.1f}M")
    print(f"  - Size (float32): {total_params * 4 / 1024 / 1024:.1f} MB")
    print(f"  - Size (float16): {total_params * 2 / 1024 / 1024:.1f} MB")


def test_simple_export(model, output_dir: Path, device: str):
    """Test simple ONNX export of embedding + single layer."""
    print("\n=== Testing Simple Export ===")

    class SimpleWrapper(nn.Module):
        """Minimal wrapper for testing export."""

        def __init__(self, model):
            super().__init__()
            self.embeddings = model.embeddings
            self.codebook_embeddings = model.codebook_embeddings
            self.norm = model.norm
            self.output = model.output

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            """Simple forward without attention layers."""
            # Text embeddings
            x = self.embeddings(input_ids)
            x = self.norm(x)
            logits = self.output(x)
            return logits

    wrapper = SimpleWrapper(model)
    wrapper.eval()

    # Test input
    dummy_input = torch.randint(0, 1000, (1, 10), device=device)
    print(f"Input shape: {dummy_input.shape}")

    with torch.no_grad():
        output = wrapper(dummy_input)
    print(f"Output shape: {output.shape}")

    # Export
    output_path = output_dir / "transformer_embed_test.onnx"
    print(f"Exporting to {output_path}")

    try:
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(output_path),
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_len"},
                "logits": {0: "batch", 1: "seq_len"},
            },
            opset_version=17,
            dynamo=False,
        )
        print("Export successful!")

        # Verify
        import onnxruntime as ort

        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        onnx_output = session.run(None, {"input_ids": dummy_input.cpu().numpy()})[0]

        diff = np.abs(onnx_output - output.cpu().numpy()).max()
        print(f"Max diff: {diff}")

        return True
    except Exception as e:
        print(f"Export failed: {e}")
        return False


def export_fast_model(model, output_dir: Path, device: str):
    """Export fast model for codebook prediction."""
    print("\n=== Exporting Fast Model ===")

    class FastModelExport(nn.Module):
        """Fast model wrapper for ONNX export."""

        def __init__(self, model):
            super().__init__()
            self.fast_embeddings = model.fast_embeddings
            self.fast_layers = model.fast_layers
            self.fast_norm = model.fast_norm
            self.fast_output = model.fast_output

        def forward(
            self,
            codebook_input: torch.Tensor,  # (batch, 1) codebook index
            input_pos: torch.Tensor,  # (1,) position
        ) -> torch.Tensor:
            """Forward pass."""
            x = self.fast_embeddings(codebook_input)

            for layer in self.fast_layers:
                x = layer(x, input_pos=input_pos)

            x = self.fast_norm(x)
            logits = self.fast_output(x)
            return logits

    wrapper = FastModelExport(model)
    wrapper.eval()

    # Setup caches for fast model
    model.setup_caches(max_batch_size=1, max_seq_len=32, dtype=torch.float32)

    # Test input
    dummy_codebook = torch.randint(0, 1024, (1, 1), device=device)
    dummy_pos = torch.tensor([0], device=device)

    print(f"Input shape: codebook={dummy_codebook.shape}, pos={dummy_pos.shape}")

    with torch.no_grad():
        output = wrapper(dummy_codebook, dummy_pos)
    print(f"Output shape: {output.shape}")

    # Export
    output_path = output_dir / "transformer_fast.onnx"
    print(f"Exporting to {output_path}")

    try:
        torch.onnx.export(
            wrapper,
            (dummy_codebook, dummy_pos),
            str(output_path),
            input_names=["codebook_input", "input_pos"],
            output_names=["logits"],
            dynamic_axes={
                "codebook_input": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )
        print("Export successful!")

        # Verify
        import onnxruntime as ort

        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        onnx_output = session.run(
            None,
            {
                "codebook_input": dummy_codebook.cpu().numpy(),
                "input_pos": dummy_pos.cpu().numpy(),
            },
        )[0]

        diff = np.abs(onnx_output - output.cpu().numpy()).max()
        print(f"Max diff: {diff}")

        return True
    except Exception as e:
        print(f"Export failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Export DualARTransformer to ONNX")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/home/progcat/.cache/fish-speech-onnx/models/fishaudio--openaudio-s1-mini",
        help="Path to model directory",
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
        "--analyze-only",
        action="store_true",
        help="Only analyze model, don't export",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_transformer_model(args.model_dir, args.device)

    # Analyze
    analyze_model_for_export(model)

    if args.analyze_only:
        return

    # Test simple export
    test_simple_export(model, output_dir, args.device)

    # Export fast model
    export_fast_model(model, output_dir, args.device)

    print("\n=== Export Summary ===")
    for f in output_dir.glob("transformer*.onnx"):
        print(f"{f.name}: {f.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
