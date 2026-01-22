"""Tests for audio processing module."""

import io

import numpy as np
import pytest

from fish_speech_onnx.utils.audio import AudioProcessor, StreamingAudioBuffer


class TestAudioProcessor:
    """Tests for AudioProcessor."""

    def test_init(self):
        """Test initialization."""
        processor = AudioProcessor(sample_rate=44100)
        assert processor.sample_rate == 44100

    def test_numpy_to_wav_bytes(self):
        """Test numpy to WAV conversion."""
        processor = AudioProcessor(sample_rate=44100)

        # Create test audio (1 second of sine wave)
        t = np.linspace(0, 1, 44100)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        wav_bytes = processor.numpy_to_wav_bytes(audio)

        # Check WAV header
        assert wav_bytes[:4] == b"RIFF"
        assert wav_bytes[8:12] == b"WAVE"

    def test_numpy_to_pcm_bytes(self):
        """Test numpy to PCM conversion."""
        processor = AudioProcessor()

        audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        pcm_bytes = processor.numpy_to_pcm_bytes(audio, dtype="int16")

        # Check size (5 samples * 2 bytes per sample)
        assert len(pcm_bytes) == 10

    def test_pcm_to_wav_bytes(self):
        """Test PCM to WAV conversion."""
        processor = AudioProcessor(sample_rate=44100)

        # Create PCM data
        pcm_data = np.zeros(1000, dtype=np.int16).tobytes()

        wav_bytes = processor.pcm_to_wav_bytes(pcm_data)

        # Check WAV header
        assert wav_bytes[:4] == b"RIFF"
        assert wav_bytes[8:12] == b"WAVE"

    def test_read_wav(self):
        """Test WAV reading."""
        processor = AudioProcessor(sample_rate=44100)

        # Create test WAV
        t = np.linspace(0, 0.1, 4410)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        wav_bytes = processor.numpy_to_wav_bytes(audio, sample_rate=44100)

        # Read it back
        audio_read, sr = processor.read_wav(wav_bytes)

        assert sr == 44100
        assert len(audio_read) == len(audio)

    def test_resample(self):
        """Test audio resampling."""
        processor = AudioProcessor()

        # Create audio at 44100 Hz
        audio = np.zeros(44100, dtype=np.float32)  # 1 second

        # Resample to 22050 Hz
        resampled = processor.resample(audio, 44100, 22050)

        assert len(resampled) == 22050

    def test_resample_same_rate(self):
        """Test resampling with same rate returns original."""
        processor = AudioProcessor()

        audio = np.random.randn(1000).astype(np.float32)
        resampled = processor.resample(audio, 44100, 44100)

        np.testing.assert_array_equal(audio, resampled)


class TestStreamingAudioBuffer:
    """Tests for StreamingAudioBuffer."""

    def test_init(self):
        """Test buffer initialization."""
        buffer = StreamingAudioBuffer(sample_rate=44100, chunk_samples=1024)

        assert buffer.sample_rate == 44100
        assert buffer.chunk_samples == 1024
        assert buffer.buffered_samples == 0

    def test_add(self):
        """Test adding audio to buffer."""
        buffer = StreamingAudioBuffer(chunk_samples=1024)

        audio = np.zeros(500, dtype=np.float32)
        buffer.add(audio)

        assert buffer.buffered_samples == 500

    def test_get_chunks(self):
        """Test getting chunks from buffer."""
        buffer = StreamingAudioBuffer(chunk_samples=100)

        # Add 250 samples
        audio = np.zeros(250, dtype=np.float32)
        buffer.add(audio)

        # Should get 2 chunks of 100 samples, with 50 remaining
        chunks = list(buffer.get_chunks())

        assert len(chunks) == 2
        assert buffer.buffered_samples == 50

    def test_flush(self):
        """Test flushing buffer."""
        buffer = StreamingAudioBuffer(chunk_samples=100)

        audio = np.zeros(50, dtype=np.float32)
        buffer.add(audio)

        remaining = buffer.flush()

        assert remaining is not None
        assert buffer.buffered_samples == 0

    def test_flush_empty(self):
        """Test flushing empty buffer."""
        buffer = StreamingAudioBuffer()

        remaining = buffer.flush()

        assert remaining is None
