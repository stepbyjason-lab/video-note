"""claude-watch orchestrator — runs the full pipeline and prints a manifest block."""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

# When invoked as `python scripts/watch.py` the repo root is not automatically
# on sys.path.  Insert it so that `from scripts import …` works correctly
# regardless of how the script is launched.
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts import library as lib
from scripts import resolve as resolve_mod
from scripts import download as download_mod
from scripts import transcribe as transcribe_mod
from scripts import scenes as scenes_mod
from scripts import frames as frames_mod
from scripts import frame_ocr
from scripts import ocr_ensemble
from scripts import ocr_paid
from scripts import slide_detect
from scripts import slides as slides_mod
from scripts import setup as setup_mod
from scripts import whisper


_WIN_DRIVE_PATH = re.compile(r"^[A-Za-z]:(?:\\|/(?!/))")
SLIDES_FLAG_DIST_OFFSET = 6


def _parse_ts(s: str) -> float:
    """Accept SS, MM:SS, or HH:MM:SS."""
    parts = s.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"bad timestamp: {s}")


def _probe_timestamp(duration_s: float, override: str | None) -> float:
    """Pick the timestamp for a --probe-frame capture.

    `override` (--probe-at) is parsed via `_parse_ts` when given; otherwise
    defaults to 25% into the video (layout chrome — cam/chat/toolbar position —
    is fixed for the whole call, so an exact timestamp doesn't matter as long as
    it avoids the intro/waiting-room start and the outro/Q&A end). Clamped into
    [0, max(duration_s - 0.5, 0)] so an out-of-range --probe-at never hands
    ffmpeg an out-of-range seek.
    """
    t = _parse_ts(override) if override is not None else duration_s * 0.25
    hi = max(duration_s - 0.5, 0.0)
    return min(max(t, 0.0), hi)


def _focus_range(args) -> tuple[float, float] | None:
    if args.start is None and args.end is None:
        return None
    s = _parse_ts(args.start) if args.start else 0.0
    e = _parse_ts(args.end) if args.end else 1e12  # clamped later by duration
    return (s, e)


def _scheme_ok(source: str) -> bool:
    """Allow http(s) URLs and local paths; reject every other URL scheme."""
    if _WIN_DRIVE_PATH.match(source):
        return True
    return urlparse(source).scheme in ("", "http", "https")


def _validate_slides_args(*, scene_threshold: float, phash_dist: int) -> None:
    if not (0.0 < scene_threshold < 1.0):
        sys.exit(f"--scene-threshold must be in (0,1); got {scene_threshold}")
    if not (0 <= phash_dist <= 64):
        sys.exit(f"--phash-dist must be in [0,64]; got {phash_dist}")


def _validate_slides_focus(*, slides: bool, focus) -> None:
    if slides and focus is not None:
        sys.exit(
            "--slides cannot be combined with --start/--end in v1 "
            "(focus-windowed slide capture is not supported yet)"
        )


def _validate_probe_args(*, probe_frame: bool, probe_at: str | None, slides: bool, focus) -> None:
    """Fail fast on bad --probe-frame/--probe-at combinations, INCLUDING a
    malformed --probe-at value.

    --probe-at only means anything alongside --probe-frame; --probe-frame only
    means anything alongside --slides (it measures the slide-region crop); and
    --probe-frame + focus is redundant (--probe-at is the one-timestamp knob).
    A --probe-at value that doesn't parse as a timestamp, or parses negative/
    non-finite (nan/inf), is rejected here too — before the download — instead
    of raising a raw ValueError out of `_parse_ts`, or crashing later in
    `frames.format_filename` ("cannot convert float NaN to integer"), deep
    inside Stage 2 (design decision: reject, don't silently clamp negatives,
    matching `_validate_freeze_args`).
    """
    if probe_at is not None and not probe_frame:
        sys.exit("--probe-at requires --probe-frame")
    if probe_frame and not slides:
        sys.exit("--probe-frame requires --slides (it measures the slide-region crop)")
    if probe_frame and focus is not None:
        sys.exit(
            "--probe-frame cannot be combined with --start/--end "
            "(use --probe-at to pick a single probe timestamp instead)"
        )
    if probe_at is not None and probe_frame:
        try:
            parsed = _parse_ts(probe_at)
        except ValueError:
            sys.exit(
                f"--probe-at: bad timestamp {probe_at!r} "
                "(expected SS, MM:SS, or HH:MM:SS)"
            )
        # math.isfinite rejects nan/inf, which slip past a bare `< 0` (nan
        # comparisons are always False; inf < 0 is False) — same reasoning as
        # `_validate_freeze_args`'s hold/light_threshold/merge_gap_s checks.
        if not math.isfinite(parsed) or parsed < 0:
            sys.exit(f"--probe-at must be a finite number >= 0; got {probe_at!r}")


def _validate_freeze_args(
    *, detect: str, crop, freeze_noise: str, hold: float, candidate_cap: int,
    light_threshold: float = 80.0, merge_gap_s: float = 15.0, merge_dist: int = 11,
) -> None:
    """Fail fast on bad slides knobs before any download/extraction work.

    candidate_cap applies to every detect mode; crop/freeze_noise/hold apply to freeze and
    auto; light_threshold/merge_gap_s/merge_dist only to freeze.
    """
    try:
        if candidate_cap <= 0:
            raise ValueError("--candidate-cap must be > 0")
        if detect in ("freeze", "auto"):
            if crop and crop != "auto":
                slides_mod.parse_crop(crop)
            slides_mod.validate_freeze_noise(freeze_noise)
            # math.isfinite rejects nan/inf, which slip past a bare `<= 0` (nan
            # comparisons are always False; inf > 0 is True).
            if not math.isfinite(hold) or hold <= 0:
                raise ValueError("--hold must be a finite number > 0")
            if not math.isfinite(light_threshold) or not (0 <= light_threshold <= 255):
                raise ValueError("--light-threshold must be in [0, 255]")
            # 0 is allowed (disables the merge pass); negative or non-finite is not.
            if not math.isfinite(merge_gap_s) or merge_gap_s < 0:
                raise ValueError("--merge-gap must be a finite number >= 0")
            if merge_dist < 0:
                raise ValueError("--merge-dist must be >= 0")
    except ValueError as e:
        sys.exit(str(e))


def _slides_advisories(args) -> list[str]:
    """Non-fatal advisories for flag combinations that are silently inert.

    Returned as strings (not printed) so they're unit-testable; main() prints them.
    """
    msgs: list[str] = []
    if args.detect == "auto":
        if args.prefer_light:
            msgs.append("--prefer-light applies to --detect freeze only "
                        "(auto decides from pixels and OCR text)")
        if args.merge_gap != 15.0 or args.merge_dist != 11:
            msgs.append("--merge-gap/--merge-dist apply to --detect freeze only "
                        "(auto merges build steps from OCR text)")
    if args.detect == "scene":
        if args.crop:
            msgs.append("--crop is ignored with --detect scene "
                        "(scene mode crops via --cam-corner/--caption)")
        if args.prefer_light:
            msgs.append("--prefer-light is ignored with --detect scene "
                        "(brightness filtering applies to freeze mode only)")
        if args.merge_gap != 15.0 or args.merge_dist != 11:
            msgs.append("--merge-gap/--merge-dist are ignored with --detect scene "
                        "(time-aware merge applies to freeze mode only)")
    return msgs


def _wipe_frames_dir(frames_dir: Path) -> None:
    # Containment guard: LIBRARY_ROOT is user-configurable (env var / .env /
    # --out-dir), so never delete outside the resolved library root.
    resolved = frames_dir.resolve()
    root = lib.LIBRARY_ROOT.resolve()
    if not resolved.is_relative_to(root):
        sys.exit(f"refusing to wipe {resolved}: outside library root {root}")
    if resolved.exists():
        for f in resolved.iterdir():
            f.unlink()


def _slides_flag_dist(drop_dist: int) -> int:
    return drop_dist + SLIDES_FLAG_DIST_OFFSET


def _prefix_frame_paths(records: list[dict]) -> list[dict]:
    return [{**fr, "path": f"frames/{fr['path']}"} for fr in records]


def _emit_slide_review_lines(
    flagged: list[tuple[float, float, int]],
    merged: list[tuple[float, float, int, float]],
    merge_flagged: list[tuple[float, float, int, float]],
) -> None:
    """Print the slides-mode audit lines: borderline-kept dedup pairs (`review:
    near-dup`), preserved exact-threshold merge pairs (`review: merge-threshold`),
    and folded-away merge pairs (`merged:`). Extracted from main() so this
    transparency output can be unit-tested via capsys without driving the whole
    pipeline — a merge/dedup fold that the user can't see in the manifest is a
    silent-recall regression.
    """
    for ta, tb, d in flagged:
        print(
            f"review: near-dup t={int(ta)//60:02d}:{int(ta)%60:02d} "
            f"~ t={int(tb)//60:02d}:{int(tb)%60:02d} (dist {d})"
        )
    for ta, tb, d, gap in merge_flagged:
        print(
            f"review: merge-threshold t={int(ta)//60:02d}:{int(ta)%60:02d} "
            f"~ t={int(tb)//60:02d}:{int(tb)%60:02d} (dist {d}, gap {gap:.1f}s)"
        )
    for ta, tb, d, gap in merged:
        print(
            f"merged: t={int(ta)//60:02d}:{int(ta)%60:02d} "
            f"~ t={int(tb)//60:02d}:{int(tb)%60:02d} (dist {d}, gap {gap:.1f}s)"
        )


def _record_slide_signal(work: Path, result: dict) -> None:
    """Keep what the OCR text signal decided next to the frames, and say how much it read."""
    ocr = result["ocr"]
    (work / "slide-signal.json").write_text(
        json.dumps({"ocr": ocr, "decisions": result["decisions"], "drafts": result["drafts"]},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"slide text signal: {ocr['calls']} OCR reads + {ocr['cached']} from cache "
          f"({ocr['failures']} failed), "
          f"models: {', '.join(ocr['models']) or '-'}", file=sys.stderr)


def _ocr_vendor(args) -> str:
    """"free" (Gemini models, the default) or the paid vendor the user chose."""
    return getattr(args, "ocr_provider", "free") or "free"


def _ocr_cache_name(stem: str, vendor: str) -> str:
    """Cache file per reader: the free models' keeps the plain name, a paid vendor's is suffixed
    so one reader's text is never served for another."""
    return f"{stem}.json" if vendor == "free" else f"{stem}-{vendor}.json"


def _ocr_ready(vendor: str) -> bool:
    return ocr_ensemble.is_available() if vendor == "free" else ocr_paid.is_available(vendor)


def _ocr_missing_msg(vendor: str) -> str:
    if vendor == "free":
        return "no OCR key (GEMINI_API_KEY in the key file); nothing paid is called instead"
    return f"the `{vendor}` CLI was not found on PATH"


def _ocr_draft_fn(vendor: str):
    """Reads one frame for the same-slide decision: the free writer alone, or the paid call."""
    if vendor == "free":
        return ocr_ensemble.ocr_draft
    return lambda path: ocr_paid.ocr_image(path, vendor)


def _ocr_final_fn(vendor: str):
    """Reads one saved frame: the free writer + judge (reusing detection's draft), or the paid
    call (its detection read is reused as is)."""
    if vendor == "free":
        return ocr_ensemble.ocr_image_ensemble

    def paid(path, draft=None):
        if draft is not None:
            return {"text": draft["text"], "model": draft["model"]}
        return ocr_paid.ocr_image(path, vendor)
    return paid


def _ocr_slide_frames(frame_records: list[dict], work: Path, vendor: str = "free") -> dict[str, dict]:
    """OCR the saved slide frames and write `ocr.json`. With no key (or no CLI) the frames are
    reported as failed (`no_key` / `no_cli`), not silently left without text, and nothing paid is
    called in place of the free models."""
    if not _ocr_ready(vendor):
        print(f"warning: {_ocr_missing_msg(vendor)} -- OCR is stopped; the slide frames get "
              "no OCR text (listed as ocr_failed) and no paid model is called in its place",
              file=sys.stderr)
        reason = "no_key" if vendor == "free" else "no_cli"
        results = {r["path"]: {"status": "failed", "reason": reason,
                               "error": _ocr_missing_msg(vendor)} for r in frame_records}
    else:
        drafts: dict[str, dict] = {}
        signal = work / "slide-signal.json"
        if signal.exists():
            try:
                drafts = json.loads(signal.read_text(encoding="utf-8")).get("drafts", {})
            except (OSError, ValueError):
                drafts = {}
        results = frame_ocr.ocr_frames(
            frame_records, base_dir=work,
            drafts={Path(k).name: v for k, v in drafts.items()},
            cache_path=work / _ocr_cache_name("ocr-cache", vendor),
            ocr_fn=_ocr_final_fn(vendor),
            on_progress=lambda m: print(f"  {m}", file=sys.stderr, flush=True))
    (work / "ocr.json").write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    return results


def select_scenes(video, meta, args, focus, work, *, cached):
    """Mode-dispatched detect + extract step.

    Returns (frame_records, scenes, flagged, merged, merge_flagged), with frame paths
    relative to the library directory and frame files already written. `merged` and
    `merge_flagged` are only populated in freeze-detect mode (the time-aware merge
    pass); scene mode and classic mode always return an empty list for both.
    """
    frames_dir = work / "frames"

    if args.slides:
        _wipe_frames_dir(frames_dir)
        if args.detect == "scene":
            result = slides_mod.detect_slides(
                video,
                out_dir=frames_dir,
                cam_corner=args.cam_corner,
                caption=args.caption,
                threshold=args.scene_threshold,
                max_gap=min(args.max_gap, 20.0),
                drop_dist=args.phash_dist,
                flag_dist=_slides_flag_dist(args.phash_dist),
                width_px=1280,
                candidate_cap=args.candidate_cap,
            )
        else:  # auto (default) or freeze
            freeze_kwargs = dict(
                out_dir=frames_dir,
                cam_corner=args.cam_corner,
                caption=args.caption,
                crop=args.crop,
                hold=args.hold,
                freeze_noise=args.freeze_noise,
                drop_dist=args.phash_dist,
                width_px=1280,
                candidate_cap=args.candidate_cap,
            )
            use_text = (args.detect == "auto" and not getattr(args, "no_ocr", False)
                        and _ocr_ready(_ocr_vendor(args)))
            if args.detect == "auto" and not use_text and not getattr(args, "no_ocr", False):
                print(f"warning: {_ocr_missing_msg(_ocr_vendor(args))} -- OCR is stopped; the "
                      "slides are still saved, from pixel detection only and without the OCR "
                      "text signal", file=sys.stderr)
            if use_text:
                result = slide_detect.detect_slides_text(
                    video, **freeze_kwargs, ocr_fn=_ocr_draft_fn(_ocr_vendor(args)),
                    cache_path=work / _ocr_cache_name("ocr-draft-cache", _ocr_vendor(args)),
                    on_progress=lambda m: print(f"  {m}", file=sys.stderr, flush=True))
                _record_slide_signal(work, result)
            else:
                result = slides_mod.detect_slides_freeze(
                    video,
                    flag_dist=_slides_flag_dist(args.phash_dist),
                    prefer_light=args.prefer_light,
                    light_threshold=args.light_threshold,
                    merge_gap_s=args.merge_gap,
                    merge_dist=args.merge_dist,
                    **freeze_kwargs,
                )
        frame_records = _prefix_frame_paths(result["slides"])
        scenes = [{"t": fr["t"], "score": 1.0, "kind": fr["kind"]} for fr in result["slides"]]
        return (
            frame_records,
            scenes,
            result["flagged"],
            result.get("merged", []),
            result.get("merge_flagged", []),
        )

    scenes_path = work / "scenes.json"
    if cached and scenes_path.exists() and not focus:
        raw_scenes = [
            scenes_mod.Scene(t=s["t"], score=s["score"], kind=s["kind"])
            for s in json.loads(scenes_path.read_text())
        ]
    else:
        raw_scenes = scenes_mod.detect_scenes(video, threshold=args.scene_threshold)

    if focus:
        s0, s1 = focus
        s1 = min(s1, meta["duration_s"])
        raw_scenes = [s for s in raw_scenes if s0 <= s.t <= s1]
        if not raw_scenes or raw_scenes[0].t > s0:
            raw_scenes.insert(0, scenes_mod.Scene(t=s0, score=1.0, kind="detected"))
        max_gap = min(args.max_gap, 15.0)
        duration_for_floor = s1
    else:
        max_gap = args.max_gap
        duration_for_floor = meta["duration_s"]

    floored = scenes_mod.apply_coverage_floor(
        raw_scenes, duration_s=duration_for_floor, max_gap_s=max_gap
    )
    capped = scenes_mod.apply_budget_cap(floored, max_frames=args.max_frames)

    if not focus:
        scenes_path.write_text(json.dumps(
            [{"t": s.t, "score": s.score, "kind": s.kind} for s in capped],
            indent=2,
        ))

    _wipe_frames_dir(frames_dir)
    raw_frame_records = frames_mod.extract_frames(
        video, capped, out_dir=frames_dir, width_px=args.resolution
    )
    frame_records = _prefix_frame_paths(raw_frame_records)
    scenes = [{"t": s.t, "score": s.score, "kind": s.kind} for s in capped]
    return frame_records, scenes, [], [], []


def _emit_probe_frame_lines(
    frame_path: Path, t: float, width: int, height: int, work: Path, source: str
) -> None:
    """Print the `--probe-frame` stdout block. Extracted from `_run_probe_frame` so
    this structured block — the contract the LLM caller reads to measure a
    `--crop W:H:X:Y` rectangle — is unit-testable via capsys without driving a
    real download/extract; an untested inline print here is a silent-regression
    risk (R06 precedent, same reasoning as `_emit_slide_review_lines`).

    The displayed timestamp uses `round(t)` (via divmod), matching
    `frames.format_filename`'s rounding — using `int(t)` here would truncate and
    could drift up to 1s from the frame's actual filename.
    """
    mm, ss = divmod(round(t), 60)
    print("=== probe frame ===")
    print(f"frame: {frame_path}")
    print(f"timestamp: t={mm:02d}:{ss:02d}")
    print(f"source_resolution: {width}x{height}")
    print(f"library_dir: {work}")
    print()
    print("next: Read the frame above. Identify the slide-region rectangle in SOURCE pixel")
    print("coordinates (exclude presenter cam / chat panel / toolbar) using pixel-boundary")
    print("sampling (color-transition detection), not eyeballing. Then re-run with the")
    print("measured crop:")
    print(f'  python3 watch.py "{source}" --slides --crop W:H:X:Y [...other flags]')


def _run_probe_frame(video: Path, meta: dict, work: Path, args) -> int:
    """Download-only probe: extract one native-resolution frame + report source
    pixel dimensions, then exit. Lets Claude measure a `--crop W:H:X:Y` rectangle
    without paying for a full detect/extract/transcribe pass with the wrong
    (default cam-corner) crop already baked into the frames.

    Writes meta.json (so a re-probe with the same flags reuses the cached
    download) but does NOT write manifest.json — the pipeline never ran to
    completion, and a manifest would misrepresent an unfinished run as done.
    """
    t = _probe_timestamp(meta["duration_s"], args.probe_at)
    if args.probe_at is not None and abs(_parse_ts(args.probe_at) - t) > 1e-9:
        print(
            f"note: --probe-at {args.probe_at} clamped to t={t:.1f}s "
            f"(duration {meta['duration_s']:.1f}s)",
            file=sys.stderr,
        )
    probe_dir = work / "probe"
    _wipe_frames_dir(probe_dir)
    frame_records = frames_mod.extract_frames(
        video, [scenes_mod.Scene(t=t, score=1.0, kind="probe")], out_dir=probe_dir, native=True
    )
    width, height = slides_mod.probe_dimensions(video)

    (work / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    frame_path = probe_dir / frame_records[0]["path"]
    _emit_probe_frame_lines(frame_path, t, width, height, work, meta["source"])
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows consoles/pipes often default to a legacy codec (e.g. cp949) that
    # cannot encode the en/em-dashes and non-ASCII titles we print, which would
    # crash the run after work is already done. Force UTF-8 output where supported.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(prog="watch")
    p.add_argument("source", help="URL or local path")
    p.add_argument("--start", help="focus start (SS, MM:SS, or HH:MM:SS)")
    p.add_argument("--end", help="focus end (SS, MM:SS, or HH:MM:SS)")
    p.add_argument("--max-frames", type=int, default=80)
    p.add_argument("--resolution", type=int, default=512, help="frame width in px")
    p.add_argument("--scene-threshold", type=float, default=0.30)
    p.add_argument("--max-gap", type=float, default=45.0, help="coverage floor seconds")
    p.add_argument("--whisper", choices=["local", "groq", "openai"],
                   help="force Whisper backend (local = PATH whisper CLI or "
                        "WHISPER_LOCAL_CMD; preferred automatically when available)")
    p.add_argument("--no-whisper", action="store_true", help="disable Whisper fallback")
    p.add_argument("--out-dir", help="library root override — takes precedence over the "
                   "CLAUDE_WATCH_LIBRARY env var/config (default: the OS app-data dir, "
                   "e.g. %%LOCALAPPDATA%%\\claude-watch\\library on Windows)")
    p.add_argument("--slides", action="store_true",
                   help="slide-deck mode: high-recall capture of a prepared deck")
    p.add_argument("--cam-corner", choices=["tr", "tl", "br", "bl", "none"], default="tr",
                   help="presenter-cam corner to exclude from slide detection")
    p.add_argument("--caption", choices=["bottom", "top", "none"], default="bottom",
                   help="burned-in caption band to exclude from slide detection")
    p.add_argument("--hi-res", action="store_true",
                   help="slides mode: download 1080p instead of 720p")
    p.add_argument("--phash-dist", type=int, default=4,
                   help="slides dedup drop distance (<= this = duplicate)")
    p.add_argument("--ocr-provider", choices=["free", "claude", "codex"], default="free",
                   help="slides mode: who reads the text -- free Gemini models (default) or, "
                        "only when you pick it, your Claude Code / Codex CLI (a subscription "
                        "call per frame). A failed free call never turns into a paid one")
    p.add_argument("--no-ocr", action="store_true",
                   help="slides mode: no OCR at all -- slides are detected from pixels only and the "
                        "saved frames get no text")
    p.add_argument("--detect", choices=["auto", "freeze", "scene"], default="auto",
                   help="slides detection: auto (default: held screens + cuts, with the OCR "
                        "text as a second signal; pixels only when no OCR key), freeze "
                        "(held-screen capture, pixels only) or scene (legacy scene-cut + "
                        "coverage floor)")
    p.add_argument("--crop", help="slides freeze: explicit slide-region crop W:H:X:Y, "
                   "or 'auto' to detect the static slide region automatically "
                   "(overrides --cam-corner/--caption; needed for non-corner cams/side chat)")
    p.add_argument("--hold", type=float, default=5.0,
                   help="slides freeze: min seconds a screen must hold to count as a slide")
    p.add_argument("--freeze-noise", default="-50dB",
                   help="slides freeze: pixel-change tolerance (ffmpeg freezedetect noise)")
    p.add_argument("--candidate-cap", type=int, default=800,
                   help="slides: safety cap on candidate frames before extraction")
    p.add_argument("--prefer-light", action="store_true",
                   help="slides freeze: drop dark (demo/terminal) frames by mean brightness "
                        "(opt-in; assumes light-background slides — leave off for dark decks)")
    p.add_argument("--light-threshold", type=float, default=80.0,
                   help="slides freeze: mean-grayscale cutoff 0-255 for --prefer-light (default 80)")
    p.add_argument("--merge-gap", type=float, default=15.0,
                   help="slides freeze: max seconds between held frames to merge as an "
                        "animation/scroll build-step (freeze-only; merge requires BOTH "
                        "this AND --merge-dist to match, so setting EITHER to 0 disables "
                        "the whole merge pass -> R05 behavior, not just this half of it)")
    p.add_argument("--merge-dist", type=int, default=11,
                   help="slides freeze: max hash distance to merge as a build-step -- "
                        "strictly below this merges; exactly this is kept and flagged as "
                        "`review: merge-threshold` instead of folded away (freeze-only; "
                        "merge requires BOTH this AND --merge-gap to match, so setting "
                        "EITHER to 0 disables the whole merge pass -> R05 behavior, not "
                        "just this half of it)")
    p.add_argument("--probe-frame", action="store_true",
                   help="slides only: download the source, extract ONE native-resolution "
                        "frame + report its pixel dimensions, then exit (skips detect/"
                        "extract/transcribe) -- for measuring a --crop W:H:X:Y rectangle "
                        "before the full run")
    p.add_argument("--probe-at", help="timestamp for --probe-frame (SS, MM:SS, or "
                   "HH:MM:SS); default is 25%% into the video")
    args = p.parse_args(argv)

    if not _scheme_ok(args.source):
        sys.exit(f"refusing non-http(s)/local scheme: {args.source}")
    focus = _focus_range(args)
    _validate_probe_args(
        probe_frame=args.probe_frame, probe_at=args.probe_at, slides=args.slides, focus=focus
    )
    _validate_slides_focus(slides=args.slides, focus=focus)
    if args.slides:
        _validate_slides_args(scene_threshold=args.scene_threshold, phash_dist=args.phash_dist)
        _validate_freeze_args(
            detect=args.detect, crop=args.crop, freeze_noise=args.freeze_noise,
            hold=args.hold, candidate_cap=args.candidate_cap,
            light_threshold=args.light_threshold,
            merge_gap_s=args.merge_gap, merge_dist=args.merge_dist,
        )
        for msg in _slides_advisories(args):
            print(f"warning: {msg}", file=sys.stderr)

    if args.out_dir:
        lib.LIBRARY_ROOT = Path(args.out_dir).expanduser().resolve()
    lib.LIBRARY_ROOT.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1: resolve ----
    meta = resolve_mod.resolve_source(args.source, focus_range=focus)
    meta["watched_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    meta["mode"] = "slides" if args.slides else "default"
    if args.slides:
        meta["dl_resolution"] = "1080p" if args.hi_res else "720p"
        meta["slides_profile"] = "|".join([
            args.detect,
            args.cam_corner,
            args.caption,
            args.crop or "-",
            f"{args.hold}",
            args.freeze_noise,
            f"{args.scene_threshold}",
            f"{min(args.max_gap, 20.0)}",
            f"{args.phash_dist}",
            f"{_slides_flag_dist(args.phash_dist)}",
            f"{args.candidate_cap}",
            f"{int(args.prefer_light)}",
            # threshold only affects output when prefer_light is on — keep it out of
            # the cache key otherwise so tweaking it alone doesn't force re-extraction.
            f"{args.light_threshold}" if args.prefer_light else "-",
            # merge knobs only affect output in freeze mode (scene mode never runs
            # time_aware_merge) — keep them out of the cache key otherwise, same
            # reasoning as light_threshold above.
            f"{args.merge_gap}|{args.merge_dist}" if args.detect == "freeze" else "-",
        ])
    else:
        meta["dl_resolution"] = "best"
        meta["slides_profile"] = ""
    slug = lib.slug_for(meta)
    work = lib.LIBRARY_ROOT / slug
    work.mkdir(parents=True, exist_ok=True)

    # Cache check — skip download/transcribe/scenes if same source_hash
    cached = lib.cache_lookup(slug, meta["source_hash"])

    # ---- Stage 2: download ----
    src_dir = work / "source"
    cached_videos = list(src_dir.glob("video.*"))
    if cached and cached_videos:
        video = cached_videos[0]
    else:
        if meta["is_url"]:
            video = download_mod.download_video(
                meta["source"], src_dir, basename="video", fmt=meta["dl_resolution"]
            )
        else:
            video = download_mod.copy_local(Path(meta["source"]), src_dir, basename="video")

    if args.probe_frame:
        return _run_probe_frame(video, meta, work, args)

    # ---- Stage 3: transcribe ----
    transcript_path = work / "transcript.json"
    if cached and transcript_path.exists():
        transcript = json.loads(transcript_path.read_text())
    else:
        transcript: list[dict] = []
        if meta["is_url"]:
            vtt = transcribe_mod.fetch_native_captions(meta["source"], work / "subs")
            if vtt:
                transcript = transcribe_mod.dedupe_cues(
                    transcribe_mod.parse_vtt(vtt.read_text(encoding="utf-8"))
                )
                # Keep the raw VTT alongside transcript.json for grepability
                (work / "transcript.vtt").write_bytes(vtt.read_bytes())
        if not transcript and not args.no_whisper:
            env = setup_mod._read_env()
            local_spec = whisper.detect_local_whisper(env)
            backend = whisper.pick_backend(
                groq_key=env.get("GROQ_API_KEY"),
                openai_key=env.get("OPENAI_API_KEY"),
                forced=args.whisper,
                local_available=bool(local_spec),
            )
            if backend:
                audio = work / "audio.m4a"
                try:
                    transcribe_mod.extract_audio_for_whisper(video, audio)
                except (subprocess.CalledProcessError, OSError) as e:
                    print(f"Audio extraction failed ({e}); skipping transcription "
                          "(use --no-whisper to silence).", file=sys.stderr)
                else:
                    transcript = transcribe_mod.transcribe_with_fallback(
                        audio,
                        backend=backend,
                        groq_key=env.get("GROQ_API_KEY"),
                        openai_key=env.get("OPENAI_API_KEY"),
                        forced=args.whisper,
                        local_spec=local_spec,
                        work_dir=work,
                        on_message=lambda m: print(m, file=sys.stderr),
                    )
            elif args.whisper:
                print(f"--whisper {args.whisper} requested but unavailable "
                      "(no key/binary); skipping transcription.", file=sys.stderr)
        transcript = transcribe_mod.insert_speaker_breaks(transcript)
        transcript_path.write_text(json.dumps(transcript, indent=2, ensure_ascii=False))

    if focus:
        transcript_for_window = transcribe_mod.slice_to_window(
            transcript, start_s=focus[0], end_s=focus[1]
        )
    else:
        transcript_for_window = transcript

    # ---- Stage 4+5: detect + extract (mode-dispatched) ----
    # The candidate-cap error carries an actionable message ("raise --hold ...") —
    # surface it cleanly. Catch ONLY that specific type: a broad `except RuntimeError`
    # would also swallow genuine ffmpeg/programming failures into a terse exit, hiding
    # bugs. Real extraction failures keep their traceback (a loud signal, not silent).
    try:
        frame_records, scenes, flagged, merged, merge_flagged = select_scenes(
            video, meta, args, focus, work, cached=cached
        )
    except slides_mod.CandidateCapExceeded as e:
        sys.exit(str(e))

    # ---- Stage 6: read the text of every saved slide frame ----
    ocr_results: dict[str, dict] = {}
    if args.slides and not args.no_ocr and frame_records:
        ocr_results = _ocr_slide_frames(frame_records, work, _ocr_vendor(args))
        for rec in frame_records:
            rec["ocr"] = ocr_results[rec["path"]]

    # ---- Stage 7: emit manifest + meta + structured stdout block ----
    transcript_window_path = work / "transcript.window.json"
    if focus:
        transcript_window_path.write_text(
            json.dumps(transcript_for_window, indent=2, ensure_ascii=False)
        )
        transcript_consumer_path = "transcript.window.json"
    else:
        transcript_consumer_path = "transcript.json"

    (work / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    lib.write_manifest(
        path=work / "manifest.json",
        meta=meta,
        scenes=scenes,
        frames=frame_records,
        transcript_path=transcript_consumer_path,
        focus_range=focus,
    )

    # Stdout: the contract Claude consumes
    duration_str = f"{int(meta['duration_s']) // 60:02d}:{int(meta['duration_s']) % 60:02d}"
    focus_str = "full" if focus is None else f"{args.start or '0:00'}–{args.end or 'end'}"
    transcript_kind = (
        "captions" if (work / "transcript.vtt").exists()
        else "whisper" if transcript
        else "none"
    )
    print("=== claude-watch manifest ===")
    print(f"title: {meta['title']!r}")
    print(f"source: {meta['source']}")
    print(f"duration: {duration_str}")
    print(f"focus: {focus_str}")
    print(f"transcript_source: {transcript_kind}")
    print(f"scenes_detected: {sum(1 for s in scenes if s['kind'] == 'detected')}")
    print(f"frames_extracted: {len(frame_records)}")
    print(f"library_dir: {work}")
    print()
    print("=== frames ===")
    for fr in frame_records:
        mm = int(fr["t"]) // 60
        ss = int(fr["t"]) % 60
        print(f"{fr['index']:04d}  t={mm:02d}:{ss:02d}  {fr['path']}  ({fr['kind']})")
    if args.slides:
        print(f"slides_extracted: {len(frame_records)}")
        if ocr_results:
            failed = sorted(p for p, r in ocr_results.items() if r["status"] != "ok")
            print(f"ocr_frames: {len(ocr_results) - len(failed)} ok, {len(failed)} failed "
                  f"(text in manifest.json and ocr.json)")
            for p in failed:
                print(f"ocr_failed: {p} ({ocr_results[p]['reason']})")
        _emit_slide_review_lines(flagged, merged, merge_flagged)
    print()
    print("=== transcript ===")
    print(f"{work / transcript_consumer_path}  (load this — too long to inline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
