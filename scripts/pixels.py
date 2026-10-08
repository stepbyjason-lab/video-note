"""Small-frame pixel helpers shared by slide detection and the ground-truth tools.

Frames are 128x72 gray images (bytes), sampled from a video in one ffmpeg pass. The work is
done in pure Python on purpose: this project has no numpy dependency, and the comparisons
below only touch a few frames per candidate.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

W, H = 128, 72
CHANGE_LEVEL = 24   # gray-level difference that counts as "this pixel changed"
BLOCK = 8           # blocks of 8x8 pixels: 16 x 9 per frame
BLOCK_MIN_PX = 3    # a block counts as changed when this many of its pixels changed


def read_gray_frames(video: Path, crop_vf: str, fps: float) -> list[bytes]:
    """All frames of `video` at `fps`, cropped by `crop_vf` (e.g. "crop=W:H:X:Y,"), as 128x72
    gray byte strings."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(video),
           "-vf", f"fps={fps},{crop_vf}scale={W}:{H},format=gray", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    size = W * H
    return [raw[i:i + size] for i in range(0, len(raw) - size + 1, size)]


def gray_of_image(path: Path, crop_vf: str = "") -> bytes:
    """One image file as a 128x72 gray byte string (same scaling as the video frames)."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-protocol_whitelist",
           "file", "-i", str(path), "-vf", f"{crop_vf}scale={W}:{H},format=gray",
           "-frames:v", "1", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    if len(raw) != W * H:
        raise RuntimeError(f"expected {W * H} gray bytes, got {len(raw)} for {path}")
    return raw


def diff_stats(a: bytes, b: bytes) -> tuple[float, int]:
    """`(changed fraction, changed blocks)` of two 128x72 gray frames in one pass.

    changed fraction: share of pixels whose gray level differs by more than CHANGE_LEVEL.
    changed blocks: number of 8x8 blocks holding at least BLOCK_MIN_PX changed pixels.
    Compression noise scatters single changed pixels, so it rarely fills a block; an edited
    word or digit does. 0 blocks means "no visible edit", whatever the noise.
    """
    if a == b:
        return 0.0, 0
    mask = [abs(x - y) > CHANGE_LEVEL for x, y in zip(a, b)]
    total = sum(mask)
    if total < BLOCK_MIN_PX:
        return total / (W * H), 0
    blocks = 0
    for by in range(0, H, BLOCK):
        for bx in range(0, W, BLOCK):
            n = 0
            for y in range(by, by + BLOCK):
                base = y * W + bx
                n += sum(mask[base:base + BLOCK])
            if n >= BLOCK_MIN_PX:
                blocks += 1
    return total / (W * H), blocks


def changed_frac(a: bytes, b: bytes) -> float:
    return diff_stats(a, b)[0]


def changed_blocks(a: bytes, b: bytes) -> int:
    return diff_stats(a, b)[1]
