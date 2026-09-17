# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""YuE2-3B text-to-music model for vllm-omni: single-stage native-AR.

The checkpoint is one Mixture-of-Transformers: a Qwen3-shaped AR path (which
is exactly a Qwen3-1.7B-class backbone over an extended vocabulary) plus a
parallel NAR path (``nar_self_attn``/``nar_mlp`` per layer) that a 32-step
midpoint flow-matching solver walks over 64-dim VAE latents. Both paths and
the projection heads live in the same ``model.safetensors``, so a single
vLLM stage loads everything once and no weights are duplicated.

Sampling is model-owned (``prefer_model_sampler``), reproducing the upstream
request-local arithmetic: per-phase vocabulary masking, windowed repetition
penalty, seeded multinomial, ``min_tokens`` end-token suppression. One song
is one or two engine requests — the abc phase (``cot=full|melody``) first,
then the semantic phase; the driver stitches them. When the semantic phase's
end token is drawn (or its frame budget is hit), the model solves the ODE and
decodes 48 kHz stereo audio inside the step and ships it as the request's
final multimodal payload, so the whole song arrives in one piece.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.v1.outputs import SamplerOutput

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .constants import (
    ABC_END,
    ABC_SAMPLING,
    CODEC_OFFSET,
    LATENT_DIM,
    MUSIC_END,
    ODE_STEPS,
    SAMPLE_RATE,
    SEMANTIC_SAMPLING,
    VAE_CORE_FRAMES,
    VAE_HALO_FRAMES,
    DEFAULT_VAE_ID,
    KEY_MAX_AUDIO_FRAMES,
    KEY_MIN_TOKENS,
    KEY_PENALTY_WINDOW,
    KEY_PHASE,
    KEY_REPETITION_PENALTY,
    KEY_SEED,
    KEY_SKIP_SYNTHESIS,
    KEY_TEMPERATURE,
    KEY_TOP_K,
    KEY_TOP_P,
)
from .nar import synthesize
from .sampling import distribution, sample_row
from .vae import YuE2VAE

logger = init_logger(__name__)

__all__ = ["Yue2ForCausalLM"]

DEFAULT_SEED = 831001


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class _RotaryEmbedding(nn.Module):
    """Checkpoint-compatible RoPE: cos/sin over [B, T, head_dim/2]."""

    def __init__(self, head_dim: int, base: float = 1000000.0):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self._inv_freq: torch.Tensor | None = None

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = position_ids.device
        if self._inv_freq is None or self._inv_freq.device != device:
            self._inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim)
            )
        angles = position_ids.float().unsqueeze(-1) * self._inv_freq
        return angles.cos(), angles.sin()


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class _NARAttention(nn.Module):
    """NAR-path attention with checkpoint-native separate projections."""

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = _RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = _RMSNorm(self.head_dim, config.rms_norm_eps)

    def project_qkv(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        rc, rs = cos.unsqueeze(2), sin.unsqueeze(2)
        return _apply_rotary(q, rc, rs), _apply_rotary(k, rc, rs), v


class _NARMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class _NARLayer(nn.Module):
    """One NAR path layer; checkpoint ``model.layers.N.nar_*`` remaps onto it."""

    def __init__(self, config):
        super().__init__()
        self.input_layernorm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = _NARAttention(config)
        self.pre_mlp_layernorm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = _NARMLP(config)


class _TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb.to(next(self.parameters()).dtype))


class _AudioPositionEmbedding(nn.Module):
    def __init__(self, max_frames: int, hidden_size: int):
        super().__init__()
        pe = torch.zeros(max_frames, hidden_size)
        position = torch.arange(0, max_frames, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, position_ids):
        return self.pe[position_ids]


def _request_key(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return "" if value is None else str(value)


@dataclass
class _RowConstants:
    """Per-request values fixed at admission (a prefill step, never replayed)."""

    request_id: str
    phase: str
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    penalty_window: int
    min_tokens: int
    max_audio_frames: int
    seed: int
    skip_synthesis: bool


@dataclass
class _RequestState:
    """Everything one song request accumulates across decode steps."""

    request_id: str
    constants: _RowConstants
    prompt_tokens: int = 0
    prefix_ids: list[int] = field(default_factory=list)
    history: list[int] = field(default_factory=list)
    generator: torch.Generator | None = None
    finished: bool = False
    truncated: bool = False


class Yue2ForCausalLM(nn.Module):
    """YuE2-3B: vLLM-native Qwen3 AR backbone + NAR flow-matching + VAE."""

    have_multimodal_outputs = True
    prefer_model_sampler = True
    has_postprocess = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        config = vllm_config.model_config.hf_config
        self.config = config
        hidden = int(config.hidden_size)

        self.model = Qwen3Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                int(config.vocab_size),
                hidden,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        self.logits_processor = LogitsProcessor(int(config.vocab_size))

        self.nar_layers = nn.ModuleList([_NARLayer(config) for _ in range(int(config.num_hidden_layers))])
        self.vae2llm = nn.Linear(int(getattr(config, "latent_dim", LATENT_DIM)), hidden)
        self.llm2vae = nn.Linear(hidden, int(getattr(config, "latent_dim", LATENT_DIM)))
        self.time_embedder = _TimestepEmbedder(hidden)
        self.latent_pos_embed = _AudioPositionEmbedding(
            int(getattr(config, "max_latent_frames", config.max_position_embeddings)),
            hidden,
        )
        self.rotary_emb = _RotaryEmbedding(int(config.head_dim), float(config.rope_theta))

        self.vocab_size = int(config.vocab_size)
        self.max_position_embeddings = int(config.max_position_embeddings)
        self.max_latent_frames = int(getattr(config, "max_latent_frames", self.max_position_embeddings))
        self._timestep_shift = float(getattr(config, "timestep_shift", 1.0))

        self._states: dict[str, _RequestState] = {}
        self._row_constants: dict[str, _RowConstants] = {}
        self._audio_queue: list[tuple[str, torch.Tensor, bool]] = []
        self._deferred_cleanup_ids: set[str] = set()
        self._step_rows: list[tuple[str, int, int]] = []  # (req_id, computed, scheduled)
        self._vae: YuE2VAE | None = None
        self._vae_device: torch.device | None = None

    # ------------------------------------------------------------ sampling

    def shift_t(self, raw_t: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t_sig = torch.sigmoid(torch.tensor(raw_t, dtype=dtype, device=device))
        shift = self._timestep_shift
        return shift * t_sig / (1 + (shift - 1) * t_sig)

    def rotary(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rotary_emb(positions)

    def ar_project(self, layer, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """AR-path Q/K/V from the vLLM backbone's fused projection."""
        attn = layer.self_attn
        qkv = attn.qkv_proj(x)
        if isinstance(qkv, tuple):
            qkv = qkv[0]
        num_heads, num_kv_heads, head_dim = attn.num_heads, attn.num_kv_heads, attn.head_dim
        q, k, v = torch.split(
            qkv,
            [num_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim],
            dim=-1,
        )
        T = x.shape[-2]
        q = attn.q_norm(q.view(T, num_heads, head_dim))
        k = attn.k_norm(k.view(T, num_kv_heads, head_dim))
        rc, rs = cos.unsqueeze(2), sin.unsqueeze(2)
        q = _apply_rotary(q, rc, rs)
        k = _apply_rotary(k, rc, rs)
        return q, k, v.view(T, num_kv_heads, head_dim)

    # ------------------------------------------------------------ weights

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        ar_pairs: list[tuple[str, torch.Tensor]] = []
        side: dict[str, torch.Tensor] = {}
        for name, tensor in weights:
            if name == "latent_pos_embed.pe":
                continue  # deterministic sinusoid, rebuilt in __init__
            if ".nar_" in name or name.split(".", 1)[0] in {"vae2llm", "llm2vae", "time_embedder"}:
                side[name] = tensor
                continue
            ar_pairs.append((name, tensor))

        # Remap checkpoint NAR names onto the parallel module list:
        # model.layers.N.nar_self_attn.X -> nar_layers.N.self_attn.X, etc.
        remapped: list[tuple[str, torch.Tensor]] = []
        for name, tensor in side.items():
            new = name
            if name.startswith("model.layers."):
                rest = name[len("model.layers.") :]
                layer_no, _, tail = rest.partition(".")
                if tail.startswith("nar_"):
                    tail = tail[len("nar_") :]
                new = f"nar_layers.{layer_no}.{tail}"
            remapped.append((new, tensor))

        missing, unexpected = self.load_state_dict(
            dict(remapped), strict=False
        )
        missing = [k for k in missing if not k.startswith(("model.", "lm_head"))]
        if missing or unexpected:
            raise RuntimeError(f"YuE2 NAR/projection weight mismatch: missing={missing}, unexpected={unexpected}")

        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(iter(ar_pairs))
        # The NAR/projection modules were populated by hand above;
        # DefaultModelLoader.track_weights_loading diffs every named
        # parameter against the returned set, so the side keys must be
        # reported too or the loader flags them as uninitialized.
        return loaded | {name for name, _ in remapped}

    # ------------------------------------------------------------ hooks

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def prepare_runner_inputs(
        self,
        *,
        req_ids: list[str],
        num_computed_tokens: Any,
        num_scheduled_tokens: Any,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bind rows to requests for the step; prompt tokens are captured in
        ``forward``, where the state is created (see ``_capture_constants``).
        Decode rows feed tokens the model itself sampled, so they never
        contribute to the prompt prefix."""
        computed = [int(v) for v in num_computed_tokens]
        scheduled = [int(v) for v in num_scheduled_tokens]
        self._step_rows = [(str(rid), computed[i], scheduled[i]) for i, rid in enumerate(req_ids)]
        return input_ids, positions

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del intermediate_tensors
        self._capture_constants(kwargs, input_ids)
        hidden = self.model(
            input_ids=input_ids if inputs_embeds is None else None,
            positions=positions,
            inputs_embeds=inputs_embeds,
        )
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        self._flush_deferred_cleanup()
        return hidden

    def _capture_constants(self, kwargs: dict[str, Any], input_ids: torch.Tensor | None) -> None:
        """Create per-request state at the (never-replayed) prefill step.

        This is the only hook that sees both the sampling extra args and the
        scheduled prompt tokens, so it is where the request's constants, its
        seeded generator, and its prompt prefix are captured once. Later
        decode steps may replay forward from a CUDA graph, skipping this
        body; ``prepare_runner_inputs`` keeps the row binding alive instead.
        """
        extra_args = kwargs.get("sampling_extra_args")
        if extra_args is None or input_ids is None:
            return
        offset = 0
        for row, (req_id, comp, span) in enumerate(self._step_rows):
            if row >= len(extra_args):
                break
            if comp == 0 and span > 1 and req_id not in self._states:
                args = extra_args[row] or {}
                phase = str(args.get(KEY_PHASE, "semantic"))
                preset = ABC_SAMPLING if phase == "abc" else SEMANTIC_SAMPLING
                seed = args.get(KEY_SEED, DEFAULT_SEED)
                constants = _RowConstants(
                    request_id=req_id,
                    phase=phase,
                    temperature=float(args.get(KEY_TEMPERATURE, preset["temperature"])),
                    top_p=float(args.get(KEY_TOP_P, preset["top_p"])),
                    top_k=int(args.get(KEY_TOP_K, preset["top_k"])),
                    repetition_penalty=float(args.get(KEY_REPETITION_PENALTY, preset["repetition_penalty"])),
                    penalty_window=int(args.get(KEY_PENALTY_WINDOW, preset["penalty_window"])),
                    min_tokens=int(args.get(KEY_MIN_TOKENS, preset["min_tokens"])),
                    max_audio_frames=int(args.get(KEY_MAX_AUDIO_FRAMES, preset["max_tokens"])),
                    seed=int(seed),
                    skip_synthesis=bool(args.get(KEY_SKIP_SYNTHESIS, phase == "abc")),
                )
                device = input_ids.device
                generator = torch.Generator(device=device if device.type != "mps" else "cpu")
                generator.manual_seed(constants.seed)
                self._row_constants[req_id] = constants
                self._states[req_id] = _RequestState(
                    request_id=req_id,
                    constants=constants,
                    prompt_tokens=span,
                    prefix_ids=input_ids[offset : offset + span].tolist(),
                    generator=generator,
                )
            offset += span

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: Any = None) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    # ------------------------------------------------------------ sampler

    def sample(self, logits: torch.Tensor, sampling_metadata: Any) -> SamplerOutput | None:
        """Draw one token per decoding row with the request's own arithmetic."""
        del sampling_metadata
        rows = int(logits.shape[0])
        token_ids = torch.zeros((rows, 1), dtype=torch.long, device=logits.device)
        if not self._step_rows:
            return SamplerOutput(sampled_token_ids=token_ids, logprobs_tensors=None)

        for row, (req_id, comp, span) in enumerate(self._step_rows):
            if row >= rows:
                break
            state = self._states.get(req_id)
            if state is None or state.finished:
                continue
            if span != 1 and not (comp == 0 and span > 1):
                # Mid-chunk prefill rows: their sampled token is discarded.
                continue
            constants = state.constants
            scores = distribution(
                logits[row],
                temperature=constants.temperature,
                top_p=constants.top_p,
                top_k=constants.top_k,
                repetition_penalty=constants.repetition_penalty,
                penalty_window=constants.penalty_window,
                history=state.history,
                step=len(state.history),
                min_tokens=constants.min_tokens,
                phase=constants.phase,
            )
            token = sample_row(scores, state.generator, greedy=constants.temperature == 0)
            end = ABC_END if constants.phase == "abc" else MUSIC_END
            if token == end:
                state.finished = True
                state.truncated = False
                self._finish_request(state, hit_end=True)
                token_ids[row, 0] = end
            else:
                state.history.append(token)
                budget = constants.max_audio_frames
                if constants.phase == "semantic" and len(state.history) >= budget:
                    state.finished = True
                    state.truncated = True
                    self._finish_request(state, hit_end=False)
                    token_ids[row, 0] = end
                else:
                    token_ids[row, 0] = token
        return SamplerOutput(sampled_token_ids=token_ids, logprobs_tensors=None)

    # ------------------------------------------------------------ audio

    def _finish_request(self, state: _RequestState, *, hit_end: bool) -> None:
        """Solve the ODE and decode the song; abc-phase requests skip this."""
        constants = state.constants
        if constants.skip_synthesis or not state.history:
            return
        from .constants import CODEC_SIZE

        codec = [t - CODEC_OFFSET for t in state.history]
        if min(codec) < 0 or max(codec) >= CODEC_SIZE:
            raise RuntimeError("semantic history contains non-codec tokens")
        latents = synthesize(self, state.prefix_ids, codec, constants.seed, steps=ODE_STEPS)
        audio = self._decode_latents(latents)
        self._audio_queue.append((state.request_id, audio, state.truncated))
        del hit_end

    def _vae_model(self, device: torch.device) -> YuE2VAE:
        if self._vae is None:
            vae_path = os.environ.get("YUE2_VAE", DEFAULT_VAE_ID)
            logger.info("Loading YuE2 VAE decoder from %s", vae_path)
            self._vae = YuE2VAE.from_pretrained(vae_path, decoder_only=True, device="cpu")
        if self._vae_device != device:
            self._vae.to(device)
            self._vae_device = device
        return self._vae

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """[frames, 64] FP32 CPU latents -> interleaved stereo [samples, 2]."""
        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        model = self._vae_model(torch.device(device))
        z = latents.T.unsqueeze(0)  # [1, 64, T]
        with torch.inference_mode():
            audio = model.decode_tiled(
                z,
                core_frames=VAE_CORE_FRAMES,
                halo_frames=VAE_HALO_FRAMES,
                output_device="cpu",
            )
        if not torch.isfinite(audio).all():
            raise RuntimeError("VAE produced non-finite audio")
        return audio[0].float().clamp(-1, 1).T.contiguous().reshape(-1)

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        by_req: dict[str, torch.Tensor] = {}
        truncated: dict[str, bool] = {}
        for req_id, audio, is_truncated in self._audio_queue:
            by_req[req_id] = audio
            truncated[req_id] = is_truncated
        self._audio_queue.clear()
        if not by_req:
            return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs=None)
        ready = list(by_req)
        sr = torch.tensor(SAMPLE_RATE, dtype=torch.int32)
        mm: dict[str, Any] = {
            "model_outputs": [by_req[r] for r in ready],
            "sr": [sr for _ in ready],
            "meta": {
                "req_id": ready,
                "sparse_audio": ["1" for _ in ready],
                "truncated": [str(int(truncated[r])) for r in ready],
            },
        }
        return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs=mm)

    def on_requests_finished(self, finished_req_ids: Iterable[str]) -> None:
        # Fires before forward; defer the free so the in-flight step can read.
        for rid in finished_req_ids:
            self._deferred_cleanup_ids.add(_request_key(rid))

    def _flush_deferred_cleanup(self) -> None:
        for req_id in self._deferred_cleanup_ids:
            state = self._states.pop(req_id, None)
            if state is not None and not state.finished:
                logger.warning(
                    "YuE2 request %s ended without an end token or its frame "
                    "budget (aborted or preempted); no audio was produced.",
                    req_id,
                )
            self._row_constants.pop(req_id, None)
        self._deferred_cleanup_ids.clear()
