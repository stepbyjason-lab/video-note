"""Paid OCR through the user's own subscription CLIs (Claude Code or Codex).

Only used when the user asks for it (`--ocr-provider claude|codex`); the free path never turns
paid by itself. One call per image, no judge step. The call is a child process with no stdin and
no console prompt; a missing CLI, a non-zero exit or an empty answer raises `OcrError` with a
`reason` (`no_cli`, `cli_failed`, `timeout`, `empty_response`), so a failed frame is reported.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from scripts import ocr_gemini as gemini
from scripts.ocr_ensemble import _ocr_error, _looks_empty
from scripts.ocr_gemini import OcrError

VENDORS = ("claude", "codex")
DEFAULT_CLAUDE_MODEL = "sonnet"  # an alias the CLI resolves; `OCR_CLAUDE_MODEL` overrides
DEFAULT_CODEX_EFFORT = "low"     # reading text off an image needs no deep reasoning; `OCR_CODEX_EFFORT` overrides
TIMEOUT_S = 240

PROMPT = (
    "Extract ALL text in the image file {path} exactly and completely, in every language, "
    "preserving the original order and structure. Output only the extracted text, with no "
    "commentary, no analysis and no code fences."
)


def is_available(vendor: str) -> bool:
    return vendor in VENDORS and shutil.which(vendor) is not None


def ocr_image(path: str | Path, vendor: str) -> dict:
    """Image -> `{"text", "model", "mode": "paid"}`; `model` names the vendor and the model
    the CLI reports (or `default` when it does not say)."""
    image = Path(path).resolve()
    if vendor not in VENDORS:
        raise _ocr_error(f"unknown paid OCR vendor: {vendor}", "no_cli")
    if not image.exists():
        raise _ocr_error(f"image file not found: {image}", "file_missing")
    exe = shutil.which(vendor)
    if exe is None:
        raise _ocr_error(f"`{vendor}` CLI not found on PATH", "no_cli")
    prompt = PROMPT.format(path=image)
    with tempfile.TemporaryDirectory(prefix="video-note-ocr-") as tmp:
        if vendor == "codex":
            text, model = _run_codex(exe, image, prompt, Path(tmp))
        else:
            text, model = _run_claude(exe, image, prompt, Path(tmp))
    if _looks_empty(text):
        raise _ocr_error(f"{vendor} answered empty/refused", "empty_response")
    return {"text": text.strip(), "model": f"{vendor}:{model}", "mode": "paid"}


def _run(cmd: list[str], cwd: Path, vendor: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, cwd=cwd, stdin=subprocess.DEVNULL, capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        raise _ocr_error(f"{vendor} timed out after {TIMEOUT_S}s", "timeout") from e
    except OSError as e:
        raise _ocr_error(f"{vendor} could not start: {e}", "no_cli") from e


def _run_codex(exe: str, image: Path, prompt: str, tmp: Path) -> tuple[str, str]:
    out = tmp / "answer.txt"
    model = (gemini.settings().get("OCR_CODEX_MODEL") or "").strip()
    effort = (gemini.settings().get("OCR_CODEX_EFFORT") or "").strip() or DEFAULT_CODEX_EFFORT
    cmd = [exe, "exec", "--skip-git-repo-check", "--sandbox", "read-only", "--ephemeral",
           "-c", f'model_reasoning_effort="{effort}"',
           "-i", str(image), "--output-last-message", str(out)]
    if model:
        cmd += ["-m", model]
    cmd.append(prompt)
    proc = _run(cmd, tmp, "codex")
    if proc.returncode != 0:
        raise _ocr_error(f"codex exited {proc.returncode}: {proc.stderr.strip()[-200:]}",
                         "cli_failed")
    text = out.read_text(encoding="utf-8", errors="replace") if out.exists() else proc.stdout
    return text, model or _codex_configured_model() or "default"


def _codex_configured_model() -> str:
    """The model Codex runs when none is given: `model = "..."` in `~/.codex/config.toml`
    (the CLI does not print it). Empty when the file or key is absent."""
    try:
        text = (Path.home() / ".codex" / "config.toml").read_text(encoding="utf-8")
    except OSError:
        return ""
    m = re.search(r'^model\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return m.group(1) if m else ""


def _run_claude(exe: str, image: Path, prompt: str, tmp: Path) -> tuple[str, str]:
    model = (gemini.settings().get("OCR_CLAUDE_MODEL") or "").strip() or DEFAULT_CLAUDE_MODEL
    cmd = [exe, "-p", prompt, "--output-format", "json", "--allowedTools", "Read",
           "--add-dir", str(image.parent), "--no-session-persistence", "--model", model]
    proc = _run(cmd, tmp, "claude")
    if proc.returncode != 0:
        raise _ocr_error(f"claude exited {proc.returncode}: {proc.stderr.strip()[-200:]}",
                         "cli_failed")
    try:
        data = json.loads(proc.stdout)
        text = data["result"]
    except (ValueError, KeyError, TypeError) as e:
        raise OcrError(f"claude output not understood: {e}") from e
    reported = list((data.get("modelUsage") or {}).keys())
    return text, reported[0] if reported else (model or "default")
