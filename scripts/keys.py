"""The one key file video-note reads API keys from (OCR, notes, transcription).

Which file is "the" key file depends on the install:
  - a development checkout (the repo root has `.git`): `<repo>/.env.local`
  - a live install (plugin cache copy, no `.git`): `~/.config/video-note/.env.local`,
    a spot the plugin updater never rewrites, so the keys survive updates
`VIDEO_NOTE_KEY_FILE` overrides both (tests, unusual layouts).

Non-key settings (library location, local Whisper command) stay in
`~/.config/claude-watch/.env`; see `scripts/setup.py`.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LIVE_KEY_FILE = Path.home() / ".config" / "video-note" / ".env.local"


def key_file_path() -> Path:
    override = (os.environ.get("VIDEO_NOTE_KEY_FILE") or "").strip()
    if override:
        return Path(override).expanduser()
    if (REPO_ROOT / ".git").exists():
        return REPO_ROOT / ".env.local"
    return LIVE_KEY_FILE


def parse_env_text(text: str) -> dict[str, str]:
    """`NAME=value` lines; blanks and `#` comments skipped, matching quotes stripped.

    The last definition of a name wins (shell semantics).
    """
    env: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        env[name.strip()] = value
    return env


def load_key_file(path: Path | None = None) -> dict[str, str]:
    """Entries with a non-empty value from the key file; {} if it is absent or unreadable."""
    path = path or key_file_path()
    try:
        text = path.read_text(encoding="utf-8-sig")  # utf-8-sig: tolerate a Windows BOM
    except (OSError, UnicodeDecodeError):
        return {}
    return {k: v for k, v in parse_env_text(text).items() if v}
