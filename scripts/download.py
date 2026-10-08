"""yt-dlp download wrapper + local file linker."""
from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path


_FORMATS = {
    "best": "best[ext=mp4]/best",
    "720p": "bv*[height<=720]+ba/b[height<=720]/best",
    "1080p": "bv*[height<=1080]+ba/b[height<=1080]/best",
}


def format_selector(fmt: str) -> str:
    """Map an internal format enum to a yt-dlp selector."""
    return _FORMATS.get(fmt, _FORMATS["best"])


@lru_cache(maxsize=None)
def ytdlp_cmd() -> tuple[str, ...]:
    """Base `yt-dlp` argv, plus `--js-runtimes node` when it helps.

    YouTube extraction needs a JS runtime and yt-dlp enables only deno by default.
    Node is added (deno keeps priority) only if it is on PATH and this yt-dlp knows
    the flag -- an older yt-dlp would reject it, which would be worse than today.
    """
    if not shutil.which("node"):
        return ("yt-dlp",)
    try:
        probe = subprocess.run(
            ["yt-dlp", "--js-runtimes", "node", "--version"],
            capture_output=True, text=True, check=False, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return ("yt-dlp",)
    return ("yt-dlp", "--js-runtimes", "node") if probe.returncode == 0 else ("yt-dlp",)


def download_video(
    url: str, out_dir: Path, *, basename: str = "video", fmt: str = "best"
) -> Path:
    """Download to `out_dir/<basename>.<ext>` via yt-dlp. Returns the downloaded file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    template = str(out_dir / f"{basename}.%(ext)s")
    cmd = [
        *ytdlp_cmd(),
        "--no-playlist",
        "-f", format_selector(fmt),
        "-o", template,
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {proc.stderr.strip()}")
    matches = sorted(out_dir.glob(f"{basename}.*"))
    if not matches:
        raise RuntimeError(f"yt-dlp returned 0 but no {basename}.* file in {out_dir}")
    return matches[0]


_VIDEO_EXTS = {
    ".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4",
    ".mpeg", ".mpg", ".ts", ".webm", ".wmv",
}


def copy_local(src: Path, out_dir: Path, *, basename: str = "video") -> Path:
    """For local sources, symlink (cheap, no copy) into out_dir/<basename>.<ext>.
    Falls back to a regular file copy if symlink fails.

    The created symlink is followed by every later pipeline stage, so the
    resolved source must be a regular file with a plausible video extension —
    refuse directories, devices, and links to unrelated files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    src = src.expanduser().resolve()
    if not src.is_file():
        raise RuntimeError(f"local source is not a regular file: {src}")
    if src.suffix.lower() not in _VIDEO_EXTS:
        raise RuntimeError(
            f"local source does not look like a video file "
            f"({src.suffix or 'no extension'}): {src}; "
            f"expected one of: {', '.join(sorted(_VIDEO_EXTS))}"
        )
    dst = out_dir / f"{basename}{src.suffix}"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        dst.write_bytes(src.read_bytes())
    return dst
