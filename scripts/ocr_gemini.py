"""Free vision API client: Gemini image-to-text (the building block of `ocr_ensemble`).

Ported from sipher v0.1.7 (commit dc21664), core/llm_free.py.
MIT License, Copyright (c) 2026 stepbyjason-lab -- see THIRD_PARTY_NOTICES.md.

Changes from the original: HTTP goes through urllib (this project has no third-party
dependency, so `requests` is not used); the key and settings come from the video-note key
file (`scripts/keys.py`) and the environment instead of sipher's `.env.local`; sipher's
language helper is replaced by `VIDEO_NOTE_LANG`; the model is always passed in by the
caller (no config fallback).

Rules kept from the original: a missing key is reported, never raised, by `is_available()`;
the model that actually answered is always returned; an API key never appears in a log or an
exception message (it travels in the `x-goog-api-key` header, and error text is redacted).
"""
from __future__ import annotations

import base64
import json
import logging
import math
import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

from scripts import keys as keys_mod

__all__ = ["ocr_image", "is_available", "OcrError", "settings"]

_log = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 60
_MAX_RETRIES = 2
_RETRY_DELAY_SECONDS = 3

# Extraction covers every language on the image; `lang` only picks the language the
# instruction is written in (the Korean text is the verified wording from sipher).
_PROMPT_KO = (
    "이 이미지에 있는 모든 텍스트를 언어에 상관없이(한국어·영어·숫자·기호 모두) "
    "빠짐없이·정확히 추출해라. 카드 내 순서/구조 유지. 설명·해석 없이 텍스트만 출력."
)
_PROMPT_GENERIC = (
    "Extract ALL text in this image exactly and completely. "
    "Preserve the original order and structure. "
    "Output only the extracted text, no commentary or interpretation."
)


def settings() -> dict[str, str]:
    """Key-file entries first, then the process environment for names the file lacks."""
    merged = dict(keys_mod.load_key_file())
    for name, value in os.environ.items():
        merged.setdefault(name, value)
    return merged


def resolve_lang() -> str:
    """`VIDEO_NOTE_LANG` (default "ko"): the language of the OCR instructions."""
    return (settings().get("VIDEO_NOTE_LANG") or "ko").strip().lower()


def _build_prompt(lang: str) -> str:
    return _PROMPT_KO if lang == "ko" else _PROMPT_GENERIC


class OcrError(RuntimeError):
    """An OCR call failed (network, timeout, rate limit, API error, ...)."""


class HttpTimeout(Exception):
    """The HTTP request timed out."""


class HttpNetworkError(Exception):
    """The HTTP request failed before any response (DNS, refused, reset, ...)."""


class _Response:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text

    def json(self):
        return json.loads(self.text)


def post_json(url: str, *, headers: dict[str, str], body: dict, timeout: float) -> _Response:
    """POST a JSON body. A non-2xx answer is returned, not raised; transport failures raise
    `HttpTimeout` / `HttpNetworkError`."""
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return _Response(resp.status, resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        return _Response(e.code, e.read().decode("utf-8", errors="replace"))
    except (TimeoutError, socket.timeout) as e:
        raise HttpTimeout(str(e)) from e
    except urllib.error.URLError as e:
        if isinstance(e.reason, (TimeoutError, socket.timeout)):
            raise HttpTimeout(str(e.reason)) from e
        raise HttpNetworkError(str(e.reason)) from e
    except OSError as e:
        raise HttpNetworkError(str(e)) from e


def api_key() -> str | None:
    return settings().get("GEMINI_API_KEY") or None


def is_available() -> bool:
    """True when a Gemini key is configured (no network call)."""
    return bool(api_key())


class _QuotaExhausted(OcrError):
    """Quota gone with no hint of recovery (a per-day quota without `RetryInfo.retryDelay`).

    The ensemble demotes the provider for the rest of the process and falls down its ladder.
    """


class _RateLimited(OcrError):
    """Temporary rate limit (per-minute, or a per-day quota that names a retry delay).

    Not a demotion: the ensemble cools the provider down for `retry_after` seconds (or an
    exponential backoff) and brings it back. This module raises at once and never retries a
    429 itself, so there is a single place that waits.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _clamp_delay(value: float | None) -> float | None:
    """A usable delay in seconds, or None: zero, negative, NaN and inf would cancel or bypass
    the cooldown, so the caller falls back to its backoff instead."""
    if value is None or not math.isfinite(value) or value <= 0:
        return None
    return value


def _parse_retry_delay(raw: str | int | float | None) -> float | None:
    """Google `RetryInfo.retryDelay` ("44s", "3.5s", "1m", ...) -> seconds, or None."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return _clamp_delay(float(raw))
    text = str(raw).strip()
    if not text:
        return None
    try:
        if text.endswith("ms"):
            parsed = float(text[:-2]) / 1000.0
        elif text.endswith("s"):
            parsed = float(text[:-1])
        elif text.endswith("m"):
            parsed = float(text[:-1]) * 60.0
        elif text.endswith("h"):
            parsed = float(text[:-1]) * 3600.0
        else:
            parsed = float(text)
    except (TypeError, ValueError):
        return None
    return _clamp_delay(parsed)


def _classify_429(resp) -> tuple[type[OcrError], float | None]:
    """A 429/403 response -> `(_QuotaExhausted, None)` or `(_RateLimited, retry_after)`.

    Only a per-day quota that carries no `RetryInfo.retryDelay` counts as exhausted: field
    evidence showed `PerDay` and `retryDelay=44s` together, and service returned minutes
    later. A retryDelay that exists but cannot be parsed is still a temporary limit. When in
    doubt the provider is not killed.
    """
    retry_after: float | None = None
    has_retry_delay = False
    is_perday = False
    try:
        body = resp.json().get("error", {}) or {}
        for det in body.get("details", []):
            at = det.get("@type", "")
            if "QuotaFailure" in at:
                for v in det.get("violations", []):
                    qid = (v.get("quotaId", "") + v.get("quotaMetric", "")).lower()
                    if "perday" in qid:
                        is_perday = True
            if "RetryInfo" in at:
                if "retryDelay" in det:
                    has_retry_delay = True
                retry_after = _parse_retry_delay(det.get("retryDelay"))
    except Exception:  # noqa: BLE001 -- an unparsable body is treated as a temporary limit
        pass
    if is_perday and not has_retry_delay:
        return _QuotaExhausted, None
    return _RateLimited, retry_after


def _redact(text: str, secret: str | None) -> str:
    """Mask `secret` in `text` (defence in depth: the key is never in the URL anyway)."""
    return text.replace(secret, "***REDACTED***") if secret else text


def _call_gemini(image_path: Path, *, key: str, model: str, prompt: str) -> tuple[str, bool]:
    """One image + instruction -> `(text, truncated)`; `truncated` = finishReason MAX_TOKENS."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    body = {"contents": [{"parts": [
        {"text": prompt},
        {"inline_data": {"mime_type": mime,
                         "data": base64.b64encode(image_path.read_bytes()).decode("ascii")}},
    ]}]}

    last_err: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = post_json(url, headers=headers, body=body, timeout=_TIMEOUT_SECONDS)
        except HttpTimeout as e:
            last_err = e
            _log.warning("Gemini OCR timeout (try %d/%d): %s", attempt + 1, _MAX_RETRIES,
                         image_path.name)
        except HttpNetworkError as e:
            last_err = e
            _log.warning("Gemini OCR network error (try %d/%d): %s", attempt + 1, _MAX_RETRIES,
                         image_path.name)
        else:
            if resp.status_code == 429 or (
                resp.status_code == 403 and "RESOURCE_EXHAUSTED" in resp.text
            ):
                exc_cls, retry_after = _classify_429(resp)
                msg = f"Gemini rate/quota (HTTP {resp.status_code}): {image_path.name}"
                if exc_cls is _RateLimited:
                    raise _RateLimited(msg, retry_after=retry_after)
                raise exc_cls(msg)
            elif resp.status_code >= 500:
                last_err = OcrError(f"Gemini server error (HTTP {resp.status_code}): "
                                    f"{image_path.name}")
                _log.warning("Gemini OCR server error (try %d/%d) HTTP %d: %s", attempt + 1,
                             _MAX_RETRIES, resp.status_code, image_path.name)
            elif resp.status_code != 200:
                # other 4xx: a bad key or request; retrying will not help
                raise OcrError(f"Gemini OCR failed (HTTP {resp.status_code}): "
                               f"{_redact(resp.text[:300], key)}")
            else:
                try:
                    data = resp.json()
                except ValueError as e:
                    raise OcrError(f"Gemini OCR JSON parse failed: {_redact(str(e), key)}") from e
                try:
                    candidate = data["candidates"][0]
                    parts = candidate["content"]["parts"]
                    if not parts:
                        raise IndexError("candidates[0].content.parts is empty")
                    # Models with a thinking channel send reasoning first as `thought: true`
                    # parts; only the answer parts are the OCR text.
                    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                except (KeyError, IndexError, TypeError, AttributeError) as e:
                    raise OcrError(f"Gemini response malformed: {e}") from e
                return text, candidate.get("finishReason") == "MAX_TOKENS"

        if attempt < _MAX_RETRIES - 1:
            time.sleep(_RETRY_DELAY_SECONDS)

    raise OcrError(
        f"Gemini OCR retries exhausted ({_MAX_RETRIES}): {_redact(str(last_err), key)}"
    ) from last_err


def ocr_image(path: str | Path, *, model: str, prompt: str | None = None) -> dict:
    """Image file -> `{"text", "model", "truncated"}` from the given Gemini-API model.

    Raises `OcrError` for a missing file or key and for any API failure; a quota or rate-limit
    answer raises `_QuotaExhausted` / `_RateLimited` so the ensemble can manage its ladder.
    """
    image_path = Path(path)
    if not image_path.exists():
        raise OcrError(f"image file not found: {image_path}")
    key = api_key()
    if not key:
        raise OcrError("GEMINI_API_KEY is not set")
    text, truncated = _call_gemini(
        image_path, key=key, model=model,
        prompt=prompt if prompt is not None else _build_prompt(resolve_lang()))
    _log.info("Gemini OCR done: %s (model=%s, truncated=%s)", image_path.name, model, truncated)
    return {"text": text, "model": model, "truncated": truncated}
