"""Per-scene frame extraction via ffmpeg."""
from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.scenes import Scene


def format_filename(index: int, t: float) -> str:
    """`NNNN_tMM-SS.jpg`. MM may exceed 59 for videos > 1h — that's intentional
    so filenames sort naturally."""
    total = round(t)
    mm, ss = divmod(total, 60)
    return f"{index:04d}_t{mm:02d}-{ss:02d}.jpg"


def extract_frames(
    video: Path,
    scenes: list[Scene],
    *,
    out_dir: Path,
    width_px: int = 512,
    native: bool = False,
    crop_vf: str = "",
) -> list[dict]:
    """For each scene, run a single-frame ffmpeg seek + extract.

    Returns: [{"index": int, "t": float, "path": str (relative to out_dir), "kind": str}]
    If native is true, no scale filter is applied.
    `crop_vf` (e.g. "crop=W:H:X:Y,") is prepended to the filter chain when set, so
    the written frame is cropped to the slide region. Default "" leaves the output
    full-frame, byte-identical to the prior behavior.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for i, scene in enumerate(scenes, start=1):
        name = format_filename(i, scene.t)
        out_path = out_dir / name
        # `-ss` before `-i` is fast (key-frame seek) and accurate enough for our purposes.
        # `-frames:v 1` writes exactly one frame; `scale=W:-2` keeps aspect, even-height.
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-nostdin",
            "-y",
            "-protocol_whitelist", "file",
            "-ss", f"{scene.t:.3f}",
            "-i", str(video),
            "-frames:v", "1",
        ]
        # Build the filter chain: optional crop, then optional scale. When neither
        # applies (default native, no crop) no -vf is emitted — byte-identical output.
        vf_parts: list[str] = []
        if crop_vf:
            vf_parts.append(crop_vf.rstrip(","))
        if not native:
            vf_parts.append(f"scale={width_px}:-2")
        if vf_parts:
            cmd += ["-vf", ",".join(vf_parts)]
        cmd += [
            "-q:v", "3",  # JPEG quality (2 best, 31 worst); 3 is a good balance
            str(out_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            stderr_text = (
                e.stderr.decode("utf-8", errors="replace")
                if isinstance(e.stderr, (bytes, bytearray))
                else (e.stderr or "")
            )
            raise RuntimeError(
                f"ffmpeg failed extracting frame {i} at t={scene.t:.3f}s "
                f"(exit {e.returncode}): {stderr_text.strip()}"
            ) from e
        results.append({
            "index": i,
            "t": scene.t,
            "path": name,
            "kind": scene.kind,
        })
    return results
