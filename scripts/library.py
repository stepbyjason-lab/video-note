"""Slug, cache, and manifest helpers for claude-watch's persistent library."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CONFIG_ENV_PATH = Path.home() / ".config" / "claude-watch" / ".env"
LEGACY_LIBRARY_ROOT = Path.home() / "claude-watch" / "library"


def default_library_root() -> Path:
    """Platform-standard app-data location for the library."""
    sysname = platform.system()
    if sysname == "Windows":
        base = os.environ.get("LOCALAPPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
    elif sysname == "Darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_DATA_HOME")
        root = Path(base) if base else Path.home() / ".local" / "share"
    return root / "claude-watch" / "library"


def _env_file_override() -> str | None:
    """CLAUDE_WATCH_LIBRARY value from ~/.config/claude-watch/.env, if set.

    Last definition wins (shell semantics); quotes and trailing `# comments`
    are stripped. An unreadable or non-UTF-8 file is treated as absent rather
    than crashing at import time.
    """
    if not CONFIG_ENV_PATH.exists():
        return None
    try:
        raw = CONFIG_ENV_PATH.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    value: str | None = None
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == "CLAUDE_WATCH_LIBRARY":
            v = re.split(r"\s+#", v, maxsplit=1)[0]
            v = v.strip().strip('"').strip("'")
            value = v or None
    return value


def resolve_library_root() -> Path:
    """Resolve the library root.

    Priority (highest first): the CLAUDE_WATCH_LIBRARY environment variable,
    the same key in ~/.config/claude-watch/.env, the legacy
    ~/claude-watch/library if it already exists (pre-relocation installs keep
    working untouched), then the platform-standard app-data dir. The --out-dir
    CLI flag overrides all of these (handled in watch.py).
    """
    env_value = (os.environ.get("CLAUDE_WATCH_LIBRARY") or "").strip()
    if env_value:
        return Path(env_value).expanduser()
    file_value = _env_file_override()
    if file_value:
        return Path(file_value).expanduser()
    if LEGACY_LIBRARY_ROOT.is_dir():
        return LEGACY_LIBRARY_ROOT
    return default_library_root()


LIBRARY_ROOT = resolve_library_root()

_SLUG_BAD = re.compile(r"[^a-z0-9]+")


def sanitize_title(title: str) -> str:
    """Lowercase, replace any non-[a-z0-9] run with a single dash, strip ends."""
    s = _SLUG_BAD.sub("-", title.lower()).strip("-")
    return s or "untitled"


def slug_for(meta: dict) -> str:
    """Return a stable library slug.

    Default mode keeps the upstream hash exactly (source + focus). Slides mode
    folds the full detection profile into the hash so any flag change gets a
    fresh library directory.
    """
    date = meta.get("watched_at") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    title = sanitize_title(meta.get("title", "untitled"))
    src = meta.get("source", "")
    focus = meta.get("focus_range_str", "")
    mode = meta.get("mode", "default")
    if mode == "default":
        key = src + "|" + focus
    else:
        res = meta.get("dl_resolution", "best")
        profile = meta.get("slides_profile", "")
        key = "|".join([src, focus, mode, res, profile])
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:4]
    return f"{date}-{title}-{h}"


def cache_lookup(slug: str, source_hash: str) -> Path | None:
    """Return the library dir if it exists and meta.source_hash matches; else None."""
    d = LIBRARY_ROOT / slug
    meta_path = d / "meta.json"
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return None
    return d if data.get("source_hash") == source_hash else None


def write_manifest(
    *,
    path: Path,
    meta: dict[str, Any],
    scenes: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    transcript_path: str,
    focus_range: tuple[float, float] | None,
) -> None:
    """Write manifest.json that Claude consumes."""
    payload = {
        "meta": meta,
        "scenes": scenes,
        "frames": frames,
        "transcript_path": transcript_path,
        "focus_range": (
            None
            if focus_range is None
            else {"start_s": focus_range[0], "end_s": focus_range[1]}
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
