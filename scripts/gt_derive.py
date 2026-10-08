"""Derive ground-truth time ranges for slides a person already picked.

    python -m scripts.gt_derive <video> --id harness --times 03:00,03:45,... \\
        [--pages 05,06,...] [--cam-corner tr --caption bottom | --crop W:H:X:Y] [--fps 2]

The person picked one frame per slide. This finds, around each pick, where in the video that
slide is on screen:
  final  the frames that still look like the picked frame (nearly no pixel changed);
  slide  `final` plus the build steps before/after it -- frames whose only differences from
         the picked frame sit where the picked frame has ink (a build step shows a subset of
         the final content, a different slide has ink elsewhere).
Mid-transition frames match neither, so they fall outside every range (-> surplus when saved).
The ranges are a starting point for a person to check, not a verdict.

Pixel work runs on 128x72 gray frames sampled at `--fps` from a single ffmpeg pass, in pure
Python (this project has no numpy dependency); only frames next to a pick are examined.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts import slides as slides_mod
from scripts.pixels import CHANGE_LEVEL, H, W, changed_frac, read_gray_frames

SAME_FRAC = 0.01       # <= this fraction of pixels changed: still the picked frame
OUTSIDE_INK_FRAC = 0.004  # build step: changes outside the picked frame's ink stay below this
CUT_FRAC = 0.12        # --cuts: a frame-to-frame change above this is a cut between two slides


def parse_ts(s: str) -> float:
    parts = [float(p) for p in s.split(":")]
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return sec


def ink_mask(frame: bytes) -> bytearray:
    """Pixels that differ from the frame's most common gray (the background), dilated 1 px."""
    counts = [0] * 256
    for v in frame:
        counts[v] += 1
    bg = max(range(256), key=counts.__getitem__)
    ink = bytearray(1 if abs(v - bg) > CHANGE_LEVEL else 0 for v in frame)
    out = bytearray(W * H)
    for y in range(H):
        for x in range(W):
            if ink[y * W + x]:
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < H and 0 <= xx < W:
                            out[yy * W + xx] = 1
    return out


def is_build_state(frame: bytes, final: bytes, final_ink: bytearray) -> bool:
    outside = sum(1 for i, (x, y) in enumerate(zip(frame, final))
                  if abs(x - y) > CHANGE_LEVEL and not final_ink[i])
    return outside / (W * H) <= OUTSIDE_INK_FRAC


def _walk(frames: list[bytes], start: int, step: int, keep) -> int:
    i = start
    while 0 <= i + step < len(frames) and keep(frames[i + step]):
        i += step
    return i


def derive_row(frames: list[bytes], t0: float, fps: float) -> dict:
    i0 = min(len(frames) - 1, max(0, round(t0 * fps)))
    ref = frames[i0]
    ink = ink_mask(ref)
    f_lo = _walk(frames, i0, -1, lambda f: changed_frac(f, ref) <= SAME_FRAC)
    f_hi = _walk(frames, i0, +1, lambda f: changed_frac(f, ref) <= SAME_FRAC)
    s_lo = _walk(frames, f_lo, -1, lambda f: is_build_state(f, ref, ink))
    s_hi = _walk(frames, f_hi, +1, lambda f: is_build_state(f, ref, ink))
    rng = lambda lo, hi: [round(lo / fps, 2), round(hi / fps, 2)]
    return {"slide": [rng(s_lo, s_hi)], "final": [rng(f_lo, f_hi)]}


def derive_row_by_cuts(frames: list[bytes], t0: float, fps: float) -> dict:
    """For footage that never holds still (talking head with overlays): the slide is the
    stretch between the cuts on either side of the pick; no build analysis (final == slide)."""
    cuts = [i for i in range(1, len(frames)) if changed_frac(frames[i], frames[i - 1]) > CUT_FRAC]
    i0 = min(len(frames) - 1, max(0, round(t0 * fps)))
    lo = max([c for c in cuts if c <= i0], default=0)
    hi = min([c for c in cuts if c > i0], default=len(frames)) - 1
    return {"slide": [[round(lo / fps, 2), round(hi / fps, 2)]]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("video")
    p.add_argument("--id", required=True, help="ground-truth id for the output file")
    p.add_argument("--times", required=True, help="comma list of picked-frame times (MM:SS)")
    p.add_argument("--pages", help="comma list of page labels, same length as --times")
    p.add_argument("--cam-corner", default="none", choices=sorted(slides_mod.VALID_CAM))
    p.add_argument("--caption", default="none", choices=sorted(slides_mod.VALID_CAPTION))
    p.add_argument("--crop", help="explicit crop W:H:X:Y (overrides --cam-corner/--caption)")
    p.add_argument("--cuts", action="store_true",
                   help="segment by cuts instead of similarity (footage that never holds still)")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--out", help="write the truth JSON here (default: stdout)")
    args = p.parse_args(argv)

    times = [parse_ts(t) for t in args.times.split(",")]
    pages = args.pages.split(",") if args.pages else [f"{i + 1:02d}" for i in range(len(times))]
    if len(pages) != len(times):
        sys.exit("--pages and --times differ in length")
    video = Path(args.video)
    vw, vh = slides_mod.probe_dimensions(video)
    if args.crop:
        w, h, x, y = slides_mod.parse_crop(args.crop)
        crop_vf = f"crop={w}:{h}:{x}:{y},"
    else:
        crop_vf = slides_mod.build_crop_vf(vw, vh, args.cam_corner, args.caption)
    frames = read_gray_frames(video, crop_vf, args.fps)

    rows = []
    for page, t0 in zip(pages, times):
        row = {"id": f"p{page}", "page": page, "picked_at": t0}
        row.update((derive_row_by_cuts if args.cuts else derive_row)(frames, t0, args.fps))
        rows.append(row)
    truth = {"id": args.id, "derived": {"fps": args.fps, "crop": crop_vf.rstrip(",")},
             "slides": rows}
    text = json.dumps(truth, ensure_ascii=False, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
