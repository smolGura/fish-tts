"""Fish-TTS synthesizer - standalone TTS without fish_speech dependency.

This module provides a clean TTS wrapper with optimized inference.

Features:
- Singleton pattern: model loaded once, reused across calls
- torch.compile with Inductor backend for fast inference
- Persistent compile cache for faster subsequent starts
- Prefilled references: pre-encode voice profiles for faster synthesis
- Pipeline: parallel token generation and vocoder decoding
- Dynamic references: add/remove voice profiles at runtime
- Streaming output for real-time playback
"""

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

# MUST set cache dir BEFORE importing torch
_COMPILE_CACHE_DIR = Path.home() / ".cache" / "fish-tts" / "torch_compile"
_COMPILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(_COMPILE_CACHE_DIR)

import numpy as np
import torch

# Enable Inductor optimizations
import torch._inductor.config

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
if hasattr(torch._inductor.config, "fx_graph_cache"):
    torch._inductor.config.fx_graph_cache = True

logger = logging.getLogger(__name__)

# Singleton instance
_instance: "FishTTS | None" = None
_instance_lock = threading.Lock()


@dataclass
class VoiceProfile:
    """Voice profile containing encoded reference audio codes."""

    codes: np.ndarray  # Shape: (num_codebooks, seq_len)
    text: str = ""  # Reference transcript for voice cloning
    name: str = ""  # Optional name for identification

    def save(self, path: str | Path) -> None:
        """Save profile to .npy file."""
        np.save(path, self.codes)

    @classmethod
    def load(cls, path: str | Path, text: str = "", name: str = "") -> "VoiceProfile":
        """Load profile from .npy file."""
        codes = np.load(path)
        if not name:
            name = Path(path).stem
        return cls(codes=codes, text=text, name=name)


@dataclass
class _PrefillCache:
    """Cache for prefilled reference data."""

    prompt_text: list[str] = field(default_factory=list)
    prompt_tokens: list[torch.Tensor] = field(default_factory=list)
    profiles: list[VoiceProfile] = field(default_factory=list)


class FishTTS:
    """TTS synthesizer using DualARTransformer and DAC vocoder.

    Example:
        synth = FishTTS(device="cuda")

        # Basic synthesis
        audio = synth.synthesize("Hello world!")

        # With voice cloning
        profile = synth.encode_reference(wav_bytes, "reference text")
        audio = synth.synthesize("Nice to meet you", references=[profile])
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        device: Literal["cpu", "cuda"] = "cuda",
        precision: Literal["bf16", "fp16", "fp32"] = "bf16",
        warmup: bool = True,
    ):
        """Initialize the synthesizer.

        Args:
            model_dir: Path to model directory. If None, downloads from HuggingFace.
            device: Device to use ("cpu" or "cuda").
            precision: Model precision ("bf16", "fp16", "fp32").
            warmup: Whether to run warmup after loading (triggers compilation).

        Note:
            torch.compile is always enabled for acceptable inference speed.
            Without compilation, inference is ~10x slower (unusable).
        """
        self.device = device
        self._precision = precision
        self._warmup = warmup
        self._model = None
        self._decode_one_token = None
        self._vocoder = None
        self._is_warmed_up = False

        # Prefill cache for references
        self._prefill_cache = _PrefillCache()
        self._prefill_lock = threading.Lock()

        # Determine precision dtype
        if precision == "bf16":
            self._dtype = torch.bfloat16
        elif precision == "fp16":
            self._dtype = torch.float16
        else:
            self._dtype = torch.float32

        # Ensure model is available
        self._model_dir = self._ensure_model(model_dir)

        # Load models
        self._load_models()

        # Run warmup to trigger compilation
        if warmup:
            self._run_warmup()

    def _ensure_model(self, model_dir: str | Path | None) -> Path:
        """Ensure model files are available."""
        if model_dir is not None:
            return Path(model_dir)

        # Download from HuggingFace
        from huggingface_hub import snapshot_download

        cache_dir = Path.home() / ".cache" / "fish-tts" / "models"
        repo_id = "fishaudio/openaudio-s1-mini"

        logger.info("Downloading model from %s", repo_id)
        model_path = snapshot_download(
            repo_id=repo_id,
            local_dir=cache_dir / repo_id.replace("/", "--"),
            local_dir_use_symlinks=False,
        )
        return Path(model_path)

    def _load_models(self) -> None:
        """Load transformer and vocoder models."""
        from fish_tts.models.inference import init_model

        logger.info("Loading models from %s", self._model_dir)

        # Load transformer
        t0 = time.perf_counter()
        self._model, self._decode_one_token = init_model(
            checkpoint_path=str(self._model_dir),
            device=self.device,
            precision=self._dtype,
            compile=True,  # Always compile for acceptable inference speed
        )
        t1 = time.perf_counter()
        logger.info("Transformer loaded in %.1fs", t1 - t0)

        # Load vocoder
        codec_path = self._model_dir / "codec.pth"
        if codec_path.exists():
            self._load_vocoder(codec_path)
        else:
            logger.warning("codec.pth not found, vocoder not loaded")

        # Log VRAM usage
        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / 1024**3
            logger.info("VRAM usage: %.2f GB", vram)

    def _load_vocoder(self, codec_path: Path) -> None:
        """Load the vocoder model."""
        from fish_tts.models.vocoder import (
            DAC,
            DownsampleResidualVectorQuantize,
            VocoderModelArgs,
            WindowLimitedTransformer,
        )

        logger.info("Loading vocoder from %s", codec_path)

        # Build vocoder config
        transformer_config_fn = lambda **kw: VocoderModelArgs(
            block_size=4096,
            n_layer=kw.get("n_layer", 8),
            n_head=kw.get("n_head", 16),
            dim=kw.get("dim", 1024),
            intermediate_size=kw.get("intermediate_size", 3072),
            n_local_heads=-1,
            head_dim=64,
            rope_base=10000,
            norm_eps=1e-5,
            dropout_rate=0.1,
            attn_dropout_rate=0.1,
            channels_first=True,
        )

        post_module = WindowLimitedTransformer(
            causal=True,
            window_size=128,
            input_dim=1024,
            config=transformer_config_fn(
                n_layer=8, n_head=16, dim=1024, intermediate_size=3072
            ),
        )
        pre_module = WindowLimitedTransformer(
            causal=True,
            window_size=128,
            input_dim=1024,
            config=transformer_config_fn(
                n_layer=8, n_head=16, dim=1024, intermediate_size=3072
            ),
        )

        quantizer = DownsampleResidualVectorQuantize(
            input_dim=1024,
            n_codebooks=9,
            codebook_size=1024,
            codebook_dim=8,
            quantizer_dropout=0.5,
            downsample_factor=(2, 2),
            post_module=post_module,
            pre_module=pre_module,
            semantic_codebook_size=4096,
        )

        transformer_general_config_fn = lambda **kw: VocoderModelArgs(
            block_size=kw.get("block_size", 16384),
            n_layer=kw.get("n_layer", 8),
            n_head=kw.get("n_head", 8),
            dim=kw.get("dim", 512),
            intermediate_size=kw.get("intermediate_size", 1536),
            n_local_heads=-1,
            head_dim=64,
            rope_base=10000,
            norm_eps=1e-5,
            dropout_rate=0.1,
            attn_dropout_rate=0.1,
            channels_first=True,
        )

        self._vocoder = DAC(
            sample_rate=44100,
            encoder_dim=64,
            encoder_rates=[2, 4, 8, 8],
            decoder_dim=1536,
            decoder_rates=[8, 8, 4, 2],
            encoder_transformer_layers=[0, 0, 0, 4],
            decoder_transformer_layers=[4, 0, 0, 0],
            quantizer=quantizer,
            transformer_general_config=transformer_general_config_fn,
        )

        # Load weights
        state_dict = torch.load(codec_path, map_location=self.device, weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Strip "generator." prefix
        if any("generator" in k for k in state_dict):
            state_dict = {
                k.replace("generator.", ""): v
                for k, v in state_dict.items()
                if "generator." in k
            }

        self._vocoder.load_state_dict(state_dict, strict=False, assign=True)
        self._vocoder.eval()
        self._vocoder.to(self.device)

        # Optimize vocoder: convert to bf16 for faster decode
        if self.device == "cuda":
            self._vocoder = self._vocoder.to(dtype=torch.bfloat16)
            logger.info("Vocoder loaded (bf16)")
        else:
            logger.info("Vocoder loaded")

    def _run_warmup(self) -> None:
        """Run warmup to trigger torch.compile compilation."""
        logger.info("Running warmup (first run triggers compilation)...")
        t0 = time.perf_counter()

        try:
            from fish_tts.models.inference import generate_long

            for response in generate_long(
                model=self._model,
                device=self.device,
                decode_one_token=self._decode_one_token,
                text="Hello.",
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.8,
                repetition_penalty=1.1,
                prompt_text=[],
                prompt_tokens=[],
            ):
                if response.action == "next":
                    break

            self._is_warmed_up = True
            t1 = time.perf_counter()
            logger.info("Warmup complete in %.1fs (compilation cached)", t1 - t0)

        except Exception as e:
            logger.warning("Warmup failed: %s", e)

    def encode_reference(
        self,
        audio_bytes: bytes,
        text: str,
    ) -> VoiceProfile:
        """Encode reference audio into a voice profile.

        Args:
            audio_bytes: WAV audio bytes.
            text: Transcript of the reference audio.

        Returns:
            VoiceProfile for voice cloning.
        """
        if self._vocoder is None:
            raise RuntimeError("Vocoder not loaded")

        # Read and process audio
        audio = self._read_wav(audio_bytes)

        # Encode with vocoder
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)
        # Match vocoder dtype (bf16 on CUDA for faster inference)
        audio_tensor = audio_tensor.to(self.device, dtype=self._dtype)

        audio_lengths = torch.tensor([audio_tensor.shape[-1]], device=self.device)

        with torch.no_grad():
            indices, _ = self._vocoder.encode(audio_tensor, audio_lengths)

        codes = indices.squeeze(0).cpu().numpy().astype(np.int64)

        return VoiceProfile(codes=codes, text=text)

    # =========================================================================
    # Reference Management (Prefill)
    # =========================================================================

    def set_references(self, profiles: list[VoiceProfile]) -> None:
        """Set the voice profiles for synthesis.

        This pre-encodes the references for faster synthesis.

        Args:
            profiles: List of VoiceProfiles to use.
        """
        with self._prefill_lock:
            self._prefill_cache = _PrefillCache(
                prompt_text=[p.text for p in profiles],
                prompt_tokens=[torch.from_numpy(p.codes) for p in profiles],
                profiles=list(profiles),
            )
            logger.info("Set %d reference(s)", len(profiles))

    def add_reference(self, profile: VoiceProfile) -> None:
        """Add a voice profile to the current references.

        Args:
            profile: VoiceProfile to add.
        """
        with self._prefill_lock:
            self._prefill_cache.profiles.append(profile)
            self._prefill_cache.prompt_text.append(profile.text)
            self._prefill_cache.prompt_tokens.append(torch.from_numpy(profile.codes))
            logger.info(
                "Added reference '%s', total: %d",
                profile.name,
                len(self._prefill_cache.profiles),
            )

    def clear_references(self) -> None:
        """Clear all voice profiles."""
        with self._prefill_lock:
            self._prefill_cache = _PrefillCache()
            logger.info("Cleared all references")

    def get_references(self) -> list[VoiceProfile]:
        """Get the current voice profiles.

        Returns:
            List of current VoiceProfiles.
        """
        with self._prefill_lock:
            return list(self._prefill_cache.profiles)

    @property
    def num_references(self) -> int:
        """Number of current references."""
        return len(self._prefill_cache.profiles)

    def _get_prompt_data(
        self, references: list[VoiceProfile] | None
    ) -> tuple[list[str], list[torch.Tensor]]:
        """Get prompt data, using prefill cache if no explicit references given."""
        if references is not None:
            return (
                [p.text for p in references],
                [torch.from_numpy(p.codes) for p in references],
            )
        else:
            with self._prefill_lock:
                return (
                    list(self._prefill_cache.prompt_text),
                    list(self._prefill_cache.prompt_tokens),
                )

    def synthesize(
        self,
        text: str,
        references: list[VoiceProfile] | None = None,
        temperature: float = 0.7,
        top_p: float = 0.8,
        repetition_penalty: float = 1.1,
        max_tokens: int = 2048,
    ) -> bytes:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize.
            references: List of VoiceProfiles for voice cloning.
                       If None, uses prefilled references (set via set_references()).
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            repetition_penalty: Repetition penalty.
            max_tokens: Maximum tokens to generate.

        Returns:
            WAV audio bytes.
        """
        from fish_tts.models.inference import generate_long

        prompt_text, prompt_tokens = self._get_prompt_data(references)

        codes_list = []
        for response in generate_long(
            model=self._model,
            device=self.device,
            decode_one_token=self._decode_one_token,
            text=text,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            prompt_text=prompt_text,
            prompt_tokens=prompt_tokens,
        ):
            if response.action == "sample":
                codes_list.append(response.codes)
            elif response.action == "next":
                break

        if not codes_list:
            raise RuntimeError("No audio generated")

        codes = torch.cat(codes_list, dim=1)

        return self._decode_to_wav(codes)

    def synthesize_stream(
        self,
        text: str,
        references: list[VoiceProfile] | None = None,
        chunk_tokens: int = 20,
        min_first_chunk: int = 10,
        **kwargs,
    ) -> Iterator[bytes]:
        """Streaming synthesis with pipeline (parallel generation and decoding).

        Args:
            text: Text to synthesize.
            references: List of VoiceProfiles for voice cloning.
                       If None, uses prefilled references (set via set_references()).
            chunk_tokens: Tokens per audio chunk (default 20 for ~0.5s audio).
            min_first_chunk: Minimum tokens for first chunk (lower = faster start).
            **kwargs: Additional arguments (temperature, top_p, repetition_penalty, max_tokens).

        Yields:
            PCM audio chunks (16-bit, 44100 Hz).
        """
        from fish_tts.models.inference import generate_long

        prompt_text, prompt_tokens = self._get_prompt_data(references)

        codes_queue: queue.Queue[torch.Tensor | None] = queue.Queue(maxsize=3)
        audio_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=3)

        error_holder: list[Exception] = []

        def decoder_worker():
            """Decode audio in a separate thread."""
            try:
                while True:
                    codes = codes_queue.get()
                    if codes is None:
                        break
                    audio = self._decode_to_pcm(codes)
                    audio_queue.put(audio)
            except Exception as e:
                error_holder.append(e)
            finally:
                audio_queue.put(None)

        decoder_thread = threading.Thread(target=decoder_worker, daemon=True)
        decoder_thread.start()

        try:
            buffer = []
            is_first_chunk = True
            total_tokens = 0

            for response in generate_long(
                model=self._model,
                device=self.device,
                decode_one_token=self._decode_one_token,
                text=text,
                max_new_tokens=kwargs.get("max_tokens", 2048),
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.8),
                repetition_penalty=kwargs.get("repetition_penalty", 1.1),
                prompt_text=prompt_text,
                prompt_tokens=prompt_tokens,
                streaming=True,  # Enable true streaming
            ):
                if response.action == "sample":
                    buffer.append(response.codes)
                    total_tokens += response.codes.shape[1]

                    threshold = min_first_chunk if is_first_chunk else chunk_tokens

                    if total_tokens >= threshold:
                        codes = torch.cat(buffer, dim=1)
                        codes_queue.put(codes)
                        buffer = []
                        total_tokens = 0
                        is_first_chunk = False

                        # Check for decoded audio and yield immediately
                        while not audio_queue.empty():
                            audio = audio_queue.get_nowait()
                            if audio is not None:
                                yield audio

                elif response.action == "next":
                    if buffer:
                        codes = torch.cat(buffer, dim=1)
                        codes_queue.put(codes)
                    break

        finally:
            codes_queue.put(None)

        decoder_thread.join()

        while not audio_queue.empty():
            audio = audio_queue.get_nowait()
            if audio is not None:
                yield audio

        if error_holder:
            raise error_holder[0]

    def _decode_to_wav(self, codes: torch.Tensor) -> bytes:
        """Decode codes to WAV bytes."""
        audio = self._decode_codes(codes)
        return self._to_wav_bytes(audio)

    def _decode_to_pcm(self, codes: torch.Tensor) -> bytes:
        """Decode codes to raw PCM bytes."""
        audio = self._decode_codes(codes)
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()

    def _decode_codes(self, codes: torch.Tensor) -> np.ndarray:
        """Decode codes to audio numpy array."""
        if self._vocoder is None:
            raise RuntimeError("Vocoder not loaded")

        if codes.ndim == 2:
            codes = codes.unsqueeze(0)

        codes = codes.to(self.device)
        feature_lengths = torch.tensor([codes.shape[-1]], device=self.device)

        with torch.inference_mode():
            audio, _ = self._vocoder.decode(codes, feature_lengths)

        return audio.squeeze().float().cpu().numpy()

    def _read_wav(self, audio_bytes: bytes) -> np.ndarray:
        """Read WAV bytes to numpy array."""
        import io
        import wave

        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            audio_data = wf.readframes(n_frames)

            audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            audio = audio / 32768.0

            if sample_rate != 44100:
                from scipy import signal

                audio = signal.resample(audio, int(len(audio) * 44100 / sample_rate))

        return audio

    def _to_wav_bytes(self, audio: np.ndarray, sample_rate: int = 44100) -> bytes:
        """Convert numpy audio to WAV bytes."""
        import io
        import wave

        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())

        return buffer.getvalue()

    @property
    def sample_rate(self) -> int:
        """Output sample rate."""
        return 44100

    @property
    def precision(self) -> str:
        """Model precision."""
        return self._precision


def get_instance(
    model_dir: str | Path | None = None,
    device: Literal["cpu", "cuda"] = "cuda",
    precision: Literal["bf16", "fp16", "fp32"] = "bf16",
    warmup: bool = True,
) -> FishTTS:
    """Get or create the singleton FishTTS instance.

    This ensures the model is loaded only once per process. Subsequent calls
    return the same instance, ignoring any different parameters.

    Args:
        model_dir: Path to model directory. If None, downloads from HuggingFace.
        device: Device to use ("cpu" or "cuda").
        precision: Model precision ("bf16", "fp16", "fp32").
        warmup: Whether to run warmup after loading.

    Returns:
        The singleton FishTTS instance.

    Note:
        torch.compile is always enabled for acceptable inference speed.

    Example:
        # First call loads and compiles the model (~54s with cache, ~248s first time)
        synth = get_instance()

        # Subsequent calls return the same instance instantly
        synth = get_instance()  # instant

        # Use for synthesis
        audio = synth.synthesize("Hello world")
    """
    global _instance

    if _instance is not None:
        return _instance

    with _instance_lock:
        if _instance is not None:
            return _instance

        logger.info("Creating singleton FishTTS instance...")
        _instance = FishTTS(
            model_dir=model_dir,
            device=device,
            precision=precision,
            warmup=warmup,
        )
        return _instance


def reset_instance() -> None:
    """Reset the singleton instance (for testing or reconfiguration)."""
    global _instance
    with _instance_lock:
        if _instance is not None:
            logger.info("Resetting singleton FishTTS instance")
            _instance = None
