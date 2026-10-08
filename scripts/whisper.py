"""Stdlib HTTP clients for Groq and OpenAI Whisper APIs + local-command bridge."""
from __future__ import annotations

import json
import mimetypes
import os
import shlex
import shutil
import uuid
from pathlib import Path
from typing import Callable, Literal, Optional, TypedDict, Union
from urllib.error import HTTPError
from urllib.request import Request, urlopen


GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = "whisper-large-v3"
GROQ_TURBO_MODEL = "whisper-large-v3-turbo"
OPENAI_URL = "https://api.openai.com/v1/audio/transcriptions"
OPENAI_MODEL = "whisper-1"

# Groq's endpoint sits behind Cloudflare, which returns 403 (error 1010) to
# requests with no User-Agent (stdlib urllib sends none by default). A plain UA
# clears the block; harmless for OpenAI, which shares this client.
_USER_AGENT = "claude-watch/1.0 (+https://github.com/devinilabs/claude-watch)"

# Standard local Whisper CLIs probed on PATH, in preference order. Both
# openai-whisper (`whisper`) and the faster-whisper CLI (`whisper-ctranslate2`)
# share the same `--model / --output_dir / --output_format / --language` flags.
_PATH_WHISPER_CLIS = ("whisper", "whisper-ctranslate2")
DEFAULT_LOCAL_MODEL = "base"


class CliSpec(TypedDict):
    """A standard Whisper CLI auto-detected on PATH."""
    kind: Literal["cli"]
    bin: str
    model: str


class CustomSpec(TypedDict):
    """A user-configured command from WHISPER_LOCAL_CMD."""
    kind: Literal["custom"]
    template: str


# How transcribe.transcribe_local should invoke a local Whisper.
LocalSpec = Union[CliSpec, CustomSpec]


class WhisperError(Exception):
    status_code: int | None = None   # set from the HTTP status when the failure was an HTTPError
    retry_after: float | None = None  # seconds from a Retry-After header, when present


def parse_local_cmd(raw: Optional[str]) -> Optional[list[str]]:
    """Split a WHISPER_LOCAL_CMD string into argv tokens, or None if unset.

    On Windows we split with posix=False so backslash paths survive intact
    (e.g. ``C:\\venv\\python.exe C:\\tools\\transcribe.py``).
    """
    if not raw or not raw.strip():
        return None
    try:
        tokens = shlex.split(raw.strip(), posix=(os.name != "nt"))
    except ValueError:
        # Unbalanced quote etc. — treat as unparseable rather than crashing the
        # pipeline; the caller surfaces this as a WhisperError and falls back.
        return None
    return tokens or None


def detect_local_whisper(
    env: dict, *, which: Callable[[str], Optional[str]] = shutil.which
) -> Optional[LocalSpec]:
    """Resolve how to run a local Whisper, or None if none is available.

    Precedence:
      1. ``WHISPER_LOCAL_CMD`` env var — a custom command. May contain
         ``{audio}`` / ``{outdir}`` placeholders; otherwise the audio path and
         ``--outdir <dir>`` are appended. Works with any wrapper or CLI.
      2. A standard Whisper CLI on PATH (``whisper`` then ``whisper-ctranslate2``),
         invoked with ``--model <m> --output_dir <dir> --output_format srt`` and
         automatic language detection.

    Returns a spec dict consumed by ``transcribe.transcribe_local``:
      ``{"kind": "custom", "template": str}`` or
      ``{"kind": "cli", "bin": str, "model": str}``.
    """
    raw = env.get("WHISPER_LOCAL_CMD")
    if raw and raw.strip():
        return {"kind": "custom", "template": raw.strip()}
    for exe in _PATH_WHISPER_CLIS:
        found = which(exe)
        if found:
            return {
                "kind": "cli",
                "bin": found,
                "model": env.get("WHISPER_MODEL") or DEFAULT_LOCAL_MODEL,
            }
    return None


def pick_backend(
    *,
    groq_key: Optional[str],
    openai_key: Optional[str],
    forced: Optional[str],
    local_available: bool = False,
) -> Optional[str]:
    """Return 'local', 'groq', 'openai', or None.

    Forced backend wins iff its prerequisite is present (a local Whisper for
    'local', the matching key for the cloud backends). Otherwise prefer a local
    Whisper, then Groq, then OpenAI.
    """
    if forced == "local":
        return "local" if local_available else None
    if forced == "groq":
        return "groq" if groq_key else None
    if forced == "openai":
        return "openai" if openai_key else None
    if local_available:
        return "local"
    if groq_key:
        return "groq"
    if openai_key:
        return "openai"
    return None


def _build_multipart(audio: Path, model: str) -> tuple[bytes, str]:
    """Encode a multipart/form-data body for the audio + model fields."""
    boundary = f"----whisper-{uuid.uuid4().hex}"
    crlf = b"\r\n"
    parts: list[bytes] = []
    # Field: model
    parts.append(f"--{boundary}".encode())
    parts.append(b'Content-Disposition: form-data; name="model"')
    parts.append(b"")
    parts.append(model.encode())
    # Field: response_format = verbose_json (gives us segments with timestamps)
    parts.append(f"--{boundary}".encode())
    parts.append(b'Content-Disposition: form-data; name="response_format"')
    parts.append(b"")
    parts.append(b"verbose_json")
    # Field: file
    mime = mimetypes.guess_type(audio.name)[0] or "application/octet-stream"
    parts.append(f"--{boundary}".encode())
    parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{audio.name}"'.encode()
    )
    parts.append(f"Content-Type: {mime}".encode())
    parts.append(b"")
    parts.append(audio.read_bytes())
    parts.append(f"--{boundary}--".encode())
    parts.append(b"")
    body = crlf.join(parts)
    return body, boundary


def _parse_retry_after(headers) -> float | None:
    """Seconds from an HTTP ``Retry-After`` header, or None. Only the numeric-seconds
    form is honored; the HTTP-date form (and anything unparseable/absent) returns None."""
    raw = headers.get("Retry-After") if headers else None
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _post(url: str, audio: Path, *, model: str, api_key: str) -> list[dict]:
    """POST audio + model + response_format=verbose_json to a Whisper endpoint.

    Returns: [{"t_start": float, "t_end": float, "text": str}, ...]
    Raises: WhisperError on any network, HTTP, JSON, or response-shape failure.
    """
    body, boundary = _build_multipart(audio, model)
    req = Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
            "User-Agent": _USER_AGENT,
        },
    )
    try:
        with urlopen(req, timeout=300) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        segs = payload.get("segments") or []
        return [
            {
                "t_start": float(s["start"]),
                "t_end": float(s["end"]),
                "text": s["text"].strip(),
            }
            for s in segs
        ]
    except HTTPError as e:
        # Surface the API's JSON error body so users see "Invalid API Key" etc.
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        err = WhisperError(f"HTTP {e.code} {e.reason}: {detail}")
        err.status_code = e.code
        err.retry_after = _parse_retry_after(e.headers)
        raise err from e
    except Exception as e:
        # Network, JSON parse, missing keys on segments — all surface as WhisperError.
        raise WhisperError(str(e)) from e


def transcribe_groq(audio: Path, *, api_key: str, model: str = GROQ_MODEL) -> list[dict]:
    return _post(GROQ_URL, audio, model=model, api_key=api_key)


def transcribe_openai(audio: Path, *, api_key: str) -> list[dict]:
    return _post(OPENAI_URL, audio, model=OPENAI_MODEL, api_key=api_key)
