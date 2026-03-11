from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL_VERSION = "default_v2"


@dataclass
class SentenceTransformerEncoder:
    """Small wrapper around a single shared sentence-transformers encoder."""

    model_name: str = DEFAULT_MODEL_NAME
    model_version: str = DEFAULT_MODEL_VERSION
    local_files_only: bool = False
    device: str | None = None
    _model: Any | None = None

    def encode(self, text: str) -> np.ndarray:
        model = self._load_model()
        vector = model.encode(
            str(text),
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(vector, dtype="float32").reshape(-1)

    def token_count(self, text: str) -> int:
        model = self._load_model()
        tokenized = model.tokenize([str(text)])
        attention_mask = tokenized.get("attention_mask")
        if attention_mask is not None:
            return int(attention_mask[0].sum().item())
        input_ids = tokenized.get("input_ids")
        if input_ids is not None:
            first_row = input_ids[0]
            if hasattr(first_row, "shape"):
                return int(first_row.shape[-1])
            return int(len(first_row))
        raise RuntimeError("Could not determine token count from the encoder tokenizer output.")

    def max_tokens(self) -> int | None:
        model = self._load_model()
        raw = getattr(model, "max_seq_length", None)
        try:
            return int(raw) if raw not in (None, "") else None
        except Exception:
            return None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEncoder. "
                "Install it and make sure the configured model is available."
            ) from exc
        kwargs: dict[str, Any] = {"local_files_only": self.local_files_only}
        if self.device:
            kwargs["device"] = self.device
        self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model


_DEFAULT_ENCODER: SentenceTransformerEncoder | None = None


def get_default_encoder(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    model_version: str = DEFAULT_MODEL_VERSION,
    local_files_only: bool = False,
    device: str | None = None,
) -> SentenceTransformerEncoder:
    global _DEFAULT_ENCODER
    if _DEFAULT_ENCODER is None:
        _DEFAULT_ENCODER = SentenceTransformerEncoder(
            model_name=model_name,
            model_version=model_version,
            local_files_only=local_files_only,
            device=device,
        )
    elif (
        _DEFAULT_ENCODER.model_name != model_name
        or _DEFAULT_ENCODER.model_version != model_version
        or _DEFAULT_ENCODER.local_files_only != local_files_only
        or _DEFAULT_ENCODER.device != device
    ):
        _DEFAULT_ENCODER = SentenceTransformerEncoder(
            model_name=model_name,
            model_version=model_version,
            local_files_only=local_files_only,
            device=device,
        )
    return _DEFAULT_ENCODER


def encode_family(text: str, *, encoder: Any | None = None) -> np.ndarray:
    active_encoder = encoder or get_default_encoder()
    if not hasattr(active_encoder, "encode"):
        raise TypeError("encoder must expose an encode(text) method")
    vector = active_encoder.encode(str(text))
    return np.asarray(vector, dtype="float32").reshape(-1)


def count_text_tokens(text: str, *, encoder: Any | None = None) -> int | None:
    active_encoder = encoder or get_default_encoder()
    token_count_fn = getattr(active_encoder, "token_count", None)
    if callable(token_count_fn):
        return int(token_count_fn(str(text)))
    return None


def max_text_tokens(*, encoder: Any | None = None) -> int | None:
    active_encoder = encoder or get_default_encoder()
    max_tokens_fn = getattr(active_encoder, "max_tokens", None)
    if callable(max_tokens_fn):
        return max_tokens_fn()
    raw = getattr(active_encoder, "max_seq_length", None)
    try:
        return int(raw) if raw not in (None, "") else None
    except Exception:
        return None


def measure_text_tokens(text: str, *, encoder: Any | None = None) -> dict[str, Any]:
    token_count = count_text_tokens(str(text), encoder=encoder)
    max_tokens = max_text_tokens(encoder=encoder)
    within_limit = None
    if token_count is not None and max_tokens is not None:
        within_limit = bool(token_count <= max_tokens)
    return {
        "token_count": token_count,
        "max_tokens": max_tokens,
        "within_limit": within_limit,
    }


def fit_text_to_token_limit(
    text: str,
    *,
    encoder: Any | None = None,
    preserve_header_lines: int = 3,
) -> str:
    rendered = str(text)
    budget = max_text_tokens(encoder=encoder)
    token_count = count_text_tokens(rendered, encoder=encoder)
    if budget is None or token_count is None or token_count <= budget:
        return rendered

    lines = rendered.splitlines()
    if len(lines) <= max(0, int(preserve_header_lines)):
        return rendered

    kept_lines = lines[: max(0, int(preserve_header_lines))]
    for line in lines[max(0, int(preserve_header_lines)) :]:
        candidate = "\n".join([*kept_lines, line])
        candidate_tokens = count_text_tokens(candidate, encoder=encoder)
        if candidate_tokens is None or candidate_tokens > budget:
            break
        kept_lines.append(line)
    return "\n".join(kept_lines)


def resolve_encoder_identity(encoder: Any | None = None) -> tuple[str, str]:
    active_encoder = encoder or get_default_encoder()
    model_name = str(getattr(active_encoder, "model_name", active_encoder.__class__.__name__))
    model_version = str(getattr(active_encoder, "model_version", DEFAULT_MODEL_VERSION))
    return model_name, model_version
