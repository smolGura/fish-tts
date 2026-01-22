"""Tests for configuration module."""

import pytest

import tempfile
from pathlib import Path

import numpy as np

from fish_speech_onnx.config import (
    ModelConfig,
    SpecialTokens,
    SynthesisConfig,
    TransformerConfig,
    VocoderConfig,
    VoiceProfile,
)


class TestTransformerConfig:
    """Tests for TransformerConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TransformerConfig()

        assert config.dim == 1024
        assert config.n_layer == 28
        assert config.n_head == 16
        assert config.num_codebooks == 10
        assert config.codebook_size == 4096
        assert config.vocab_size == 155776

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TransformerConfig(
            dim=512,
            n_layer=12,
            num_codebooks=8,
        )

        assert config.dim == 512
        assert config.n_layer == 12
        assert config.num_codebooks == 8


class TestSynthesisConfig:
    """Tests for SynthesisConfig."""

    def test_default_values(self):
        """Test default synthesis config."""
        config = SynthesisConfig()

        assert config.temperature == 0.7
        assert config.top_p == 0.8
        assert config.max_tokens == 1024
        assert config.sample_rate == 44100

    def test_custom_values(self):
        """Test custom synthesis config."""
        config = SynthesisConfig(
            temperature=0.5,
            top_p=0.9,
            max_tokens=2048,
        )

        assert config.temperature == 0.5
        assert config.top_p == 0.9
        assert config.max_tokens == 2048


class TestSpecialTokens:
    """Tests for SpecialTokens."""

    def test_token_ids(self):
        """Test special token IDs."""
        assert SpecialTokens.BOS == 151643
        assert SpecialTokens.EOS == 151644
        assert SpecialTokens.PAD == 151645
        assert SpecialTokens.TEXT == 151652
        assert SpecialTokens.VOICE == 151653
        assert SpecialTokens.AUDIO_START == 151655
        assert SpecialTokens.AUDIO_END == 151656

    def test_semantic_id(self):
        """Test semantic token ID conversion."""
        assert SpecialTokens.semantic_id(0) == 151658
        assert SpecialTokens.semantic_id(100) == 151758
        assert SpecialTokens.semantic_id(4095) == 155753

    def test_semantic_id_invalid(self):
        """Test semantic ID with invalid index."""
        with pytest.raises(ValueError):
            SpecialTokens.semantic_id(-1)

        with pytest.raises(ValueError):
            SpecialTokens.semantic_id(4096)

    def test_is_semantic_token(self):
        """Test semantic token detection."""
        assert SpecialTokens.is_semantic_token(151658) is True
        assert SpecialTokens.is_semantic_token(155753) is True
        assert SpecialTokens.is_semantic_token(151657) is False
        assert SpecialTokens.is_semantic_token(100) is False

    def test_semantic_index(self):
        """Test semantic index extraction."""
        assert SpecialTokens.semantic_index(151658) == 0
        assert SpecialTokens.semantic_index(151758) == 100
        assert SpecialTokens.semantic_index(155753) == 4095

    def test_semantic_index_invalid(self):
        """Test semantic index with non-semantic token."""
        with pytest.raises(ValueError):
            SpecialTokens.semantic_index(100)


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_cache_dir(self):
        """Test default cache directory."""
        config = ModelConfig()
        cache_dir = config.get_cache_dir()

        assert cache_dir.name == "fish-speech-onnx"
        assert ".cache" in str(cache_dir)

    def test_custom_cache_dir(self):
        """Test custom cache directory."""
        config = ModelConfig(cache_dir="/tmp/test-cache")
        cache_dir = config.get_cache_dir()

        assert str(cache_dir) == "/tmp/test-cache"

    def test_model_dir_priority(self):
        """Test model_dir takes priority over cache."""
        config = ModelConfig(
            model_dir="/custom/models",
            cache_dir="/tmp/cache",
        )
        model_dir = config.get_model_dir()

        assert str(model_dir) == "/custom/models"


class TestVoiceProfile:
    """Tests for VoiceProfile (official Fish-Speech .npy format)."""

    def test_creation(self):
        """Test VoiceProfile creation with numpy array."""
        codes = np.array([[0, 1, 2], [100, 200, 300]], dtype=np.int64)
        profile = VoiceProfile(codes=codes)

        assert profile.codes.shape == (2, 3)
        assert profile.codes.dtype == np.int64
        np.testing.assert_array_equal(profile.codes, codes)

    def test_creation_1d(self):
        """Test VoiceProfile with 1D array."""
        codes = np.array([0, 1, 2, 3], dtype=np.int64)
        profile = VoiceProfile(codes=codes)

        assert profile.codes.shape == (4,)

    def test_save_and_load_npy(self):
        """Test VoiceProfile save and load with .npy format."""
        codes = np.array(
            [[0, 100, 200, 300], [1, 101, 201, 301], [2, 102, 202, 302]],
            dtype=np.int64,
        )
        profile = VoiceProfile(codes=codes)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "voice.npy"
            profile.save(filepath)

            # Verify file exists
            assert filepath.exists()

            # Load and verify
            loaded = VoiceProfile.load(filepath)
            np.testing.assert_array_equal(loaded.codes, profile.codes)
            assert loaded.codes.dtype == np.int64

    def test_load_official_format(self):
        """Test loading .npy file in official Fish-Speech format."""
        # Simulate official format: (num_codebooks, seq_len)
        codes = np.random.randint(0, 4096, size=(8, 100), dtype=np.int64)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "official_format.npy"
            np.save(filepath, codes)

            loaded = VoiceProfile.load(filepath)

            np.testing.assert_array_equal(loaded.codes, codes)
            assert loaded.codes.shape == (8, 100)

    def test_codebook_values_in_range(self):
        """Test that codebook indices are in valid range (0-4095)."""
        codes = np.array([[0, 4095, 2048], [1, 4094, 2047]], dtype=np.int64)
        profile = VoiceProfile(codes=codes)

        assert profile.codes.min() >= 0
        assert profile.codes.max() <= 4095
