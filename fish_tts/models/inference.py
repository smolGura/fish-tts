"""Inference utilities for Fish-TTS.

This module provides token generation and content encoding functions
for text-to-speech synthesis.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Literal, Optional, Union

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm

from fish_tts.models.llama import DualARTransformer
from fish_tts.models.tokenizer import (
    IM_END_TOKEN,
    MODALITY_TOKENS,
    FishTokenizer,
)


def multinomial_sample_one_no_sync(probs_sort: torch.Tensor) -> torch.Tensor:
    """Multinomial sampling without CUDA synchronization."""
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
    previous_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convert logits to probabilities with sampling parameters."""
    # Apply repetition penalty
    if previous_tokens is not None:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=-1, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=-1, index=previous_tokens, src=score)

    # Apply top-p sampling
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(
        torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
    )
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[0] = False  # keep at least one option
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))
    logits = logits / torch.clip(temperature, min=1e-5)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
    previous_tokens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample from logits."""
    probs = logits_to_probs(
        logits=logits[0, -1],
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        previous_tokens=previous_tokens,
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def decode_one_token_ar(
    model: DualARTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    previous_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Decode one token using DualAR."""
    forward_result = model.forward_generate(
        x, input_pos,
        audio_masks=audio_masks,
        audio_parts=audio_parts,
    )
    logits = forward_result.logits
    hidden_states = forward_result.hidden_states

    codebooks = [
        sample(
            logits,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            previous_tokens=(
                previous_tokens[:, 0] if previous_tokens is not None else None
            ),
        )[0]
    ]

    # Clear cache for fast_layers
    for layer in model.fast_layers:
        if hasattr(layer, "attention") and hasattr(layer.attention, "kv_cache"):
            layer.attention.kv_cache.k_cache.fill_(0)
            layer.attention.kv_cache.v_cache.fill_(0)

    input_pos = torch.tensor([0], device=hidden_states.device, dtype=torch.long)
    model.forward_generate_fast(hidden_states, input_pos)
    a = codebooks[0] - model.tokenizer.semantic_begin_id
    a[a < 0] = 0
    hidden_states = model.fast_embeddings(a)
    codebooks.append(a)

    for codebook_idx in range(1, model.config.num_codebooks):
        input_pos = torch.tensor(
            [codebook_idx], device=hidden_states.device, dtype=torch.long
        )
        logits = model.forward_generate_fast(hidden_states, input_pos)

        short_logits = logits[:, :, :1024]

        a = sample(
            short_logits,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            previous_tokens=(
                previous_tokens[codebook_idx + 1]
                if previous_tokens is not None
                else None
            ),
        )[0]

        hidden_states = model.fast_embeddings(a)
        codebooks.append(a)

    codebooks = torch.stack(codebooks, dim=1)

    del logits, hidden_states, forward_result

    return codebooks.T


def decode_n_tokens(
    model: DualARTransformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    decode_one_token: Callable = decode_one_token_ar,
    show_progress: bool = True,
):
    """Decode multiple tokens."""
    device = cur_token.device
    codebook_dim = model.config.num_codebooks + 1

    previous_tokens = torch.zeros(
        (codebook_dim, model.config.max_seq_len),
        dtype=torch.int,
        device=device,
    )

    # Pre-compute end token ID
    end_token_id = model.tokenizer.get_token_id(IM_END_TOKEN)

    iterator = tqdm(range(num_new_tokens)) if show_progress else range(num_new_tokens)

    for i in iterator:
        win_size = 16
        if i < win_size:
            window = previous_tokens[:, :win_size]
        else:
            window = previous_tokens[:, i - win_size : i]

        with sdpa_kernel(SDPBackend.MATH):
            next_token = decode_one_token(
                model=model,
                x=cur_token,
                input_pos=input_pos,
                previous_tokens=window,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                audio_masks=audio_masks,
                audio_parts=audio_parts,
            ).clone()

        input_pos += 1
        cur_token = next_token.view(1, codebook_dim, -1)
        previous_tokens[:, i : i + 1] = next_token.view(codebook_dim, -1)

        if cur_token[0, 0, -1] == end_token_id:
            break

    del cur_token

    return previous_tokens[:, : i + 1]


def decode_n_tokens_streaming(
    model: DualARTransformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    decode_one_token: Callable = decode_one_token_ar,
    show_progress: bool = False,
) -> Iterator[torch.Tensor]:
    """Decode multiple tokens with streaming - yields each token as generated."""
    device = cur_token.device
    codebook_dim = model.config.num_codebooks + 1

    previous_tokens = torch.zeros(
        (codebook_dim, model.config.max_seq_len),
        dtype=torch.int,
        device=device,
    )

    # Pre-compute end token ID
    end_token_id = model.tokenizer.get_token_id(IM_END_TOKEN)

    iterator = tqdm(range(num_new_tokens)) if show_progress else range(num_new_tokens)

    for i in iterator:
        win_size = 16
        if i < win_size:
            window = previous_tokens[:, :win_size]
        else:
            window = previous_tokens[:, i - win_size : i]

        with sdpa_kernel(SDPBackend.MATH):
            next_token = decode_one_token(
                model=model,
                x=cur_token,
                input_pos=input_pos,
                previous_tokens=window,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                audio_masks=audio_masks,
                audio_parts=audio_parts,
            ).clone()

        input_pos += 1
        cur_token = next_token.view(1, codebook_dim, -1)
        previous_tokens[:, i : i + 1] = next_token.view(codebook_dim, -1)

        # Yield the token (codebooks only, excluding first row which is semantic)
        yield next_token[1:, :]  # Shape: (num_codebooks, 1)

        if cur_token[0, 0, -1] == end_token_id:
            break

    del cur_token


@torch.no_grad()
@torch.inference_mode()
def generate(
    *,
    model: DualARTransformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    decode_one_token: Callable = decode_one_token_ar,
    num_samples: int = 1,
    **sampling_kwargs,
):
    """Generate tokens from prompt."""
    T = prompt.size(1)
    prompt = prompt[None].repeat(num_samples, 1, 1)

    if T >= model.config.max_seq_len:
        raise ValueError(
            f"Input sequence length {T} exceeds max_seq_len {model.config.max_seq_len}"
        )

    if max_new_tokens:
        if T + max_new_tokens > model.config.max_seq_len:
            max_new_tokens = model.config.max_seq_len - T
        T_new = T + max_new_tokens
    else:
        T_new = model.config.max_seq_len
        max_new_tokens = T_new - T

    device, dtype = prompt.device, prompt.dtype

    if not hasattr(model, "_cache_setup_done") or not model._cache_setup_done:
        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )
        model._cache_setup_done = True

    codebook_dim = 1 + model.config.num_codebooks

    input_pos = torch.arange(0, T, device=device, dtype=torch.long)
    empty = torch.empty(
        (codebook_dim, model.config.max_seq_len), dtype=dtype, device=device
    )
    empty[:, :T] = prompt
    seq = empty

    # Use pre-created fixed parameter tensors
    temperature = getattr(
        model, "fixed_temperature", torch.tensor(0.8, device=device, dtype=torch.float)
    )
    top_p = getattr(
        model, "fixed_top_p", torch.tensor(0.8, device=device, dtype=torch.float)
    )
    repetition_penalty = getattr(
        model,
        "fixed_repetition_penalty",
        torch.tensor(1.1, device=device, dtype=torch.float),
    )

    temp_val = sampling_kwargs.get("temperature", 0.7)
    top_p_val = sampling_kwargs.get("top_p", 0.7)
    rep_val = sampling_kwargs.get("repetition_penalty", 1.5)

    if abs(temperature.item() - temp_val) > 1e-6:
        temperature.fill_(temp_val)
    if abs(top_p.item() - top_p_val) > 1e-6:
        top_p.fill_(top_p_val)
    if abs(repetition_penalty.item() - rep_val) > 1e-6:
        repetition_penalty.fill_(rep_val)

    first_token = decode_one_token_ar(
        model,
        prompt.view(1, codebook_dim, -1),
        input_pos,
        temperature,
        top_p,
        repetition_penalty,
        audio_masks,
        audio_parts,
    )
    seq[:, T : T + 1] = first_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    x = decode_n_tokens(
        model,
        first_token.view(1, codebook_dim, -1),
        input_pos,
        max_new_tokens - 1,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        audio_masks=audio_masks,
        audio_parts=audio_parts,
        decode_one_token=decode_one_token,
    )
    seq = seq[:, : T + 1 + x.size(1)]
    seq[:, T + 1 :] = x

    del first_token, x, prompt, empty, input_pos

    return seq


def init_model(
    checkpoint_path: str,
    device: str,
    precision: torch.dtype,
    compile: bool = False,
):
    """Initialize model and optionally compile decode function."""
    model = DualARTransformer.from_pretrained(checkpoint_path, load_weights=True)
    model = model.to(device=device, dtype=precision)

    decode_one_token = decode_one_token_ar

    # Pre-create fixed parameter tensors
    model.fixed_temperature = torch.tensor(0.7, device=device, dtype=torch.float)
    model.fixed_top_p = torch.tensor(0.7, device=device, dtype=torch.float)
    model.fixed_repetition_penalty = torch.tensor(1.5, device=device, dtype=torch.float)

    model._cache_setup_done = False

    if compile:
        decode_one_token = torch.compile(
            decode_one_token,
            backend="inductor" if torch.cuda.is_available() else "aot_eager",
            mode="reduce-overhead" if torch.cuda.is_available() else None,
            fullgraph=True,
        )

    return model.eval(), decode_one_token


@dataclass
class GenerateResponse:
    action: Literal["sample", "next"]
    codes: Optional[torch.Tensor] = None
    text: Optional[str] = None


# Content sequence classes
@dataclass
class BasePart:
    type: Literal["text", "vq", "audio"] | None = None
    cal_loss: bool = False


@dataclass(kw_only=True)
class VQPart(BasePart):
    type: str = "vq"
    codes: torch.Tensor

    def __post_init__(self):
        self.type = "vq"
        if isinstance(self.codes, np.ndarray):
            self.codes = torch.from_numpy(self.codes.copy())


@dataclass(kw_only=True)
class TextPart(BasePart):
    type: str = "text"
    text: str | None = None
    tokens: list[int] | None = None

    def __post_init__(self):
        self.type = "text"
        if self.text is None and self.tokens is None:
            raise ValueError("Either text or tokens must be provided")


@dataclass
class EncodedMessage:
    tokens: torch.Tensor
    labels: torch.Tensor
    vq_mask_tokens: torch.Tensor | None = None
    vq_mask_labels: torch.Tensor | None = None
    vq_parts: list[torch.Tensor] = field(default_factory=list)
    vq_require_losses: torch.Tensor | None = None
    audio_parts: list[torch.Tensor] = field(default_factory=list)
    audio_masks: torch.Tensor | None = None
    metadata: dict | None = None


class ContentSequence:
    """Flexible sequence of content parts for interleaved multimodal format."""

    def __init__(
        self,
        parts: list = None,
        modality: Literal["text", "voice", "interleave"] | None = None,
        metadata: dict | None = None,
    ):
        self.modality = modality
        self.metadata = metadata or {}

        fixed_parts = []
        for part in parts or []:
            if isinstance(part, dict):
                if part["type"] == "vq":
                    part = VQPart(**part)
                elif part["type"] == "text":
                    part = TextPart(**part)
                else:
                    raise ValueError(f"Unsupported part type: {part['type']}")
            fixed_parts.append(part)

        self.parts = fixed_parts

        if self.modality and not (
            len(self.parts) > 0
            and isinstance(self.parts[0], TextPart)
            and self.parts[0].text is not None
            and self.parts[0].text.startswith(MODALITY_TOKENS[self.modality])
        ):
            modality_token = MODALITY_TOKENS[self.modality]
            self.parts.insert(0, TextPart(text=modality_token))

    def append(
        self,
        part_or_parts: Union[BasePart, List[BasePart]],
        add_end: bool = False,
        speaker: Union[str, int] | None = None,
    ):
        """Append parts to the sequence."""
        parts_to_add = (
            [part_or_parts] if not isinstance(part_or_parts, list) else part_or_parts
        )

        if speaker is not None:
            speaker_token = f"<|speaker:{speaker}|>"
            self.parts.append(TextPart(text=speaker_token))

        self.parts.extend(parts_to_add)

        if add_end:
            self.parts.append(
                TextPart(text=IM_END_TOKEN, cal_loss=self.parts[-1].cal_loss)
            )

    def encode(
        self,
        tokenizer: FishTokenizer,
        add_shift: bool = True,
        ignore_loss_tokens: list[str] = [],
    ) -> EncodedMessage:
        """Encode the sequence parts into tokens."""
        all_tokens = []
        all_labels = []

        vq_parts = []
        vq_masks = []
        vq_require_losses = []

        audio_parts = []
        audio_masks = []

        ignore_loss_token_ids = [tokenizer.get_token_id(i) for i in ignore_loss_tokens]

        for part in self.parts:
            if isinstance(part, TextPart):
                if part.tokens is None:
                    assert part.text is not None
                    tokens = tokenizer.encode(part.text)
                else:
                    tokens = part.tokens

                tokens = torch.tensor(tokens, dtype=torch.int)
            elif isinstance(part, VQPart):
                curr_codes = part.codes.clone().to(torch.int)
                tokens = torch.tensor(
                    [
                        tokenizer.semantic_id_to_token_id[int(i.item())]
                        for i in curr_codes[0].int()
                    ],
                    dtype=torch.int,
                )
                vq_parts.append(curr_codes)
                vq_require_losses.append(part.cal_loss)
            else:
                raise ValueError(f"Unsupported part type: {type(part)}")

            all_tokens.append(tokens)

            if isinstance(part, VQPart):
                vq_masks.append(torch.ones_like(tokens, dtype=torch.bool))
                audio_masks.append(torch.zeros_like(tokens, dtype=torch.bool))
            else:
                vq_masks.append(torch.zeros_like(tokens, dtype=torch.bool))
                audio_masks.append(torch.zeros_like(tokens, dtype=torch.bool))

            if part.cal_loss:
                all_labels.append(tokens.clone())
            else:
                all_labels.append(torch.full_like(tokens, -100))

        tokens = torch.cat(all_tokens, dim=0)
        labels = torch.cat(all_labels, dim=0)
        vq_masks = torch.cat(vq_masks, dim=0)
        audio_masks = torch.cat(audio_masks, dim=0)
        vq_require_losses = torch.tensor(vq_require_losses, dtype=torch.bool)

        vq_mask_tokens = vq_masks
        vq_mask_labels = vq_masks

        if add_shift:
            tokens = tokens[:-1]
            labels = labels[1:]
            vq_masks = vq_masks[:-1]
            vq_mask_tokens = vq_mask_tokens[:-1]
            vq_mask_labels = vq_mask_labels[1:]
            audio_masks = audio_masks[:-1]

        for i in ignore_loss_token_ids:
            labels[labels == i] = -100

        return EncodedMessage(
            tokens=tokens,
            labels=labels,
            vq_parts=vq_parts,
            vq_mask_tokens=vq_mask_tokens,
            vq_mask_labels=vq_mask_labels,
            vq_require_losses=vq_require_losses,
            audio_parts=audio_parts,
            audio_masks=audio_masks,
            metadata=self.metadata,
        )

    def encode_for_inference(
        self,
        tokenizer: FishTokenizer,
        num_codebooks: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode for inference (no shift)."""
        encoded = self.encode(tokenizer, add_shift=False)
        tokens = encoded.tokens
        values = torch.zeros((num_codebooks + 1, len(tokens)), dtype=torch.int)
        values[0] = tokens

        if (encoded.vq_parts is None or len(encoded.vq_parts) == 0) and (
            encoded.audio_parts is None or len(encoded.audio_parts) == 0
        ):
            return values, None, None

        audio_parts = audio_masks = None
        if encoded.vq_parts is not None and len(encoded.vq_parts) > 0:
            vq_parts = encoded.vq_parts
            vq_parts = torch.cat(vq_parts, dim=1)
            values[0, encoded.vq_mask_tokens] = (
                vq_parts[0] + tokenizer.semantic_begin_id
            )
            values[1:, encoded.vq_mask_tokens] = vq_parts

        if encoded.audio_parts is not None and len(encoded.audio_parts) > 0:
            audio_parts = torch.cat(encoded.audio_parts, dim=0)
            audio_masks = encoded.audio_masks[None, :]

        return values, audio_masks, audio_parts


@torch.no_grad()
@torch.inference_mode()
def generate_streaming(
    *,
    model: DualARTransformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    decode_one_token: Callable = decode_one_token_ar,
    **sampling_kwargs,
) -> Iterator[torch.Tensor]:
    """Generate tokens from prompt with streaming - yields tokens as generated."""
    T = prompt.size(1)
    prompt = prompt[None].repeat(1, 1, 1)

    if T >= model.config.max_seq_len:
        raise ValueError(
            f"Input sequence length {T} exceeds max_seq_len {model.config.max_seq_len}"
        )

    if max_new_tokens:
        if T + max_new_tokens > model.config.max_seq_len:
            max_new_tokens = model.config.max_seq_len - T
    else:
        max_new_tokens = model.config.max_seq_len - T

    device = prompt.device

    if not hasattr(model, "_cache_setup_done") or not model._cache_setup_done:
        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )
        model._cache_setup_done = True

    codebook_dim = 1 + model.config.num_codebooks

    input_pos = torch.arange(0, T, device=device, dtype=torch.long)

    # Use pre-created fixed parameter tensors
    temperature = getattr(
        model, "fixed_temperature", torch.tensor(0.8, device=device, dtype=torch.float)
    )
    top_p = getattr(
        model, "fixed_top_p", torch.tensor(0.8, device=device, dtype=torch.float)
    )
    repetition_penalty = getattr(
        model,
        "fixed_repetition_penalty",
        torch.tensor(1.1, device=device, dtype=torch.float),
    )

    temp_val = sampling_kwargs.get("temperature", 0.7)
    top_p_val = sampling_kwargs.get("top_p", 0.7)
    rep_val = sampling_kwargs.get("repetition_penalty", 1.5)

    if abs(temperature.item() - temp_val) > 1e-6:
        temperature.fill_(temp_val)
    if abs(top_p.item() - top_p_val) > 1e-6:
        top_p.fill_(top_p_val)
    if abs(repetition_penalty.item() - rep_val) > 1e-6:
        repetition_penalty.fill_(rep_val)

    first_token = decode_one_token_ar(
        model,
        prompt.view(1, codebook_dim, -1),
        input_pos,
        temperature,
        top_p,
        repetition_penalty,
        audio_masks,
        audio_parts,
    )

    # Yield first token
    yield first_token[1:, :]  # Exclude semantic token row

    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    # Stream remaining tokens
    for token in decode_n_tokens_streaming(
        model,
        first_token.view(1, codebook_dim, -1),
        input_pos,
        max_new_tokens - 1,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        audio_masks=audio_masks,
        audio_parts=audio_parts,
        decode_one_token=decode_one_token,
    ):
        yield token


def generate_long(
    *,
    model: DualARTransformer,
    device: Union[str, torch.device],
    decode_one_token: Callable,
    text: str,
    num_samples: int = 1,
    max_new_tokens: int = 0,
    top_p: float = 0.8,
    repetition_penalty: float = 1.1,
    temperature: float = 0.8,
    compile: bool = False,
    prompt_text: Optional[Union[str, list[str]]] = None,
    prompt_tokens: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
    streaming: bool = False,
) -> Iterator[GenerateResponse]:
    """High-level generation API with streaming support.

    Args:
        streaming: If True, yields each token as it's generated (for real-time streaming).
                  If False, yields all tokens at once after generation completes.
    """
    assert 0 < top_p <= 1, "top_p must be in (0, 1]"
    assert 0 < repetition_penalty < 2, "repetition_penalty must be in (0, 2)"
    assert 0 < temperature < 2, "temperature must be in (0, 2)"

    use_prompt = prompt_text is not None and prompt_tokens is not None
    if use_prompt and isinstance(prompt_text, str):
        prompt_text = [prompt_text]
        prompt_tokens = [prompt_tokens]

    if use_prompt:
        assert len(prompt_text) == len(prompt_tokens)

    if prompt_tokens:
        prompt_tokens = [i.cpu() for i in prompt_tokens]

    tokenizer = model.tokenizer
    base_content_sequence = ContentSequence(modality="interleave")

    max_length = model.config.max_seq_len
    if use_prompt:
        for t, c in zip(prompt_text, prompt_tokens):
            base_content_sequence.append(
                [TextPart(text=t), VQPart(codes=c)],
                add_end=True,
                speaker=0,
            )
    base_content_sequence.append([TextPart(text=text)], add_end=False, speaker=0)

    encoded, audio_masks, audio_parts = base_content_sequence.encode_for_inference(
        tokenizer, num_codebooks=model.config.num_codebooks
    )
    if encoded.size(1) > max_length - 2048:
        raise ValueError(f"Prompt is too long: {encoded.size(1)} > {max_length - 2048}")

    encoded = encoded.to(device=device)

    for sample_idx in range(num_samples):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if streaming:
            # True streaming: yield each token as generated
            for token in generate_streaming(
                model=model,
                prompt=encoded,
                max_new_tokens=max_new_tokens,
                audio_masks=audio_masks,
                audio_parts=audio_parts,
                decode_one_token=decode_one_token,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            ):
                # Each token is (num_codebooks, 1), need to ensure non-negative
                codes = token.clone()
                codes[codes < 0] = 0
                yield GenerateResponse(action="sample", codes=codes, text=text)
        else:
            # Batch mode: generate all tokens then yield
            prompt_length = encoded.size(1)

            y = generate(
                model=model,
                prompt=encoded,
                max_new_tokens=max_new_tokens,
                audio_masks=audio_masks,
                audio_parts=audio_parts,
                decode_one_token=decode_one_token,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            codes = y[1:, prompt_length:-1].clone()
            assert (codes >= 0).all(), f"Negative code found"

            yield GenerateResponse(action="sample", codes=codes, text=text)

            del y, codes

        yield GenerateResponse(action="next")
