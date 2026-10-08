"""Slide detection with a second signal: OCR text next to the pixels.

`detect_slides_freeze` (scripts/slides.py) keeps one frame per held screen and decides
"same slide?" from an image hash alone. That fails two ways: a slide that differs by a line of
text hashes like its neighbour (the second one is lost), and a build slide is saved at its first
step rather than its last. This module adds what the pixels cannot tell:

  candidates   the frames the freeze pass would capture, plus the last frame of every stable
               run, a capture at the end of each long held screen (an edited line inside one
               long freeze shows only there) and the last stable frame
               before every hard cut (so footage that never holds still -- a talking head with
               overlays -- still yields its slides)
  same slide?  pixels first (only noise changed -> same; changed a lot -> different);
               in between the OCR text decides: same lines -> same slide; the later frame has
               extra lines -> a build step, keep the later (final) one; each has lines the other
               lacks -> a different slide. Every kept frame is compared with the *nearest* kept
               frame, not only its neighbour, so a slide that comes back is not saved twice.

Without a usable OCR key the caller falls back to `detect_slides_freeze` (pixels only).
"""
from __future__ import annotations

import hashlib
import json
import time
import warnings
from collections.abc import Callable
from pathlib import Path

from scripts import frames as frames_mod
from scripts import ocr_ensemble
from scripts import pixels
from scripts import slides as slides_mod
from scripts.scenes import Scene
from scripts.slide_text import compare_texts

SAMPLE_FPS = 2.0        # pixel sampling rate for cut detection
CUT_HARD = 0.12         # frame-to-frame changed fraction that is a cut between two slides
MIN_SEGMENT_S = 2.0     # a stretch between cuts shorter than this is a flash, a transition or an animation
RUN_MIN_S = 3.0         # a stable run (no visible edit) shorter than this is not a slide state
AMBIGUOUS = 0.30        # nearest-frame changed fraction above this: different, no OCR needed
NOISE_BLOCKS = 3        # at most this many changed 8x8 blocks (and few changed pixels) is noise:
NOISE_FRAC = 0.01       # compression shimmer on text edges, measured at 1.3 blocks on average
CANDIDATE_MERGE_S = 1.0  # a freeze capture this close to a cut-end capture is the same moment


def frame_changes(frames: list[bytes]) -> list[tuple[float, int]]:
    """`(changed fraction, changed blocks)` between each frame and the one before it
    (`pixels.diff_stats`); entry `i - 1` is the pair `(frames[i - 1], frames[i])`."""
    return [pixels.diff_stats(frames[i - 1], frames[i]) for i in range(1, len(frames))]


def _stretch_ends(n: int, boundaries: list[int], fps: float, min_s: float) -> list[float]:
    """Time of the last frame of every stretch (between `boundaries`, frame indices where a
    new stretch starts) that lasts at least `min_s` seconds."""
    starts = [0] + boundaries
    return [(b - 1) / fps for a, b in zip(starts, starts[1:] + [n]) if (b - a) / fps >= min_s]


def cut_end_times(changes: list[tuple[float, int]], fps: float, *, hard: float = CUT_HARD,
                  min_segment_s: float = MIN_SEGMENT_S) -> list[float]:
    """Time of the last frame before each cut, for every stretch at least `min_segment_s` long
    (the last stretch ends with the last frame). A cut is a frame-to-frame changed fraction
    above `hard`."""
    cuts = [i + 1 for i, (frac, _b) in enumerate(changes) if frac > hard]
    return _stretch_ends(len(changes) + 1, cuts, fps, min_segment_s)


def run_end_times(changes: list[tuple[float, int]], fps: float, *,
                  min_run_s: float = RUN_MIN_S) -> list[float]:
    """Time of the last frame of every stable run at least `min_run_s` long.

    A run is consecutive samples with no visible edit between them (0 changed blocks), so it
    sees an edited line of text that the freeze pass (a noise-level filter) lets through, and a
    build step that adds a bullet. The last frame of a run is the state before the next edit.
    Footage that never holds still (a moving presenter) has no such runs.
    """
    edits = [i + 1 for i, (_f, blocks) in enumerate(changes) if blocks > 0]
    return _stretch_ends(len(changes) + 1, edits, fps, min_run_s)


def merge_candidate_times(*groups: tuple[str, list[float]],
                          within: float = CANDIDATE_MERGE_S) -> list[tuple[float, str]]:
    """Sorted `(time, kind)` candidates from `(kind, times)` groups given in priority order; a
    time within `within` s of one from an earlier group is dropped as the same moment."""
    out: list[tuple[float, str]] = []
    for kind, times in groups:
        taken = [t for t, _k in out]
        out += [(t, kind) for t in times if all(abs(t - u) > within for u in taken)]
    return sorted(out)


def dedup_with_text(
    records: list[dict],
    *,
    gray_of: Callable[[dict], bytes],
    text_of: Callable[[dict], str | None],
    hash_dist: Callable[[dict, dict], int],
    drop_dist: int,
    ambiguous: float = AMBIGUOUS,
) -> tuple[list[dict], list[dict]]:
    """Fold time-ordered candidate records into the slides to keep.

    Each record needs a `t`. Returns `(kept, decisions)`; `decisions` has one entry per record
    that was dropped, replaced a kept frame, or needed the text signal:
    `{"t", "against_t", "changed", "blocks", "relation", "action"}`.
    `text_of` returns the OCR text of a record or None when OCR failed; None falls back to the
    hash distance (`hash_dist` <= `drop_dist` = same).
    """
    kept: list[dict] = []
    grays: dict[int, bytes] = {}
    decisions: list[dict] = []

    def gray(rec: dict) -> bytes:
        key = id(rec)
        if key not in grays:
            grays[key] = gray_of(rec)
        return grays[key]

    for rec in records:
        g = gray(rec)
        if not kept:
            kept.append(rec)
            continue
        nearest = min(kept, key=lambda k: pixels.changed_frac(gray(k), g))
        changed = pixels.changed_frac(gray(nearest), g)
        blocks = pixels.changed_blocks(gray(nearest), g)
        entry = {"t": rec["t"], "against_t": nearest["t"], "changed": round(changed, 4),
                 "blocks": blocks}
        if blocks <= NOISE_BLOCKS and changed <= NOISE_FRAC:
            decisions.append({**entry, "relation": "identical", "action": "drop"})
            continue
        if changed > ambiguous:
            kept.append(rec)
            continue
        a_text, b_text = text_of(nearest), text_of(rec)
        relation = "unknown" if a_text is None or b_text is None else compare_texts(a_text, b_text)
        if relation == "unknown":
            same = hash_dist(nearest, rec) <= drop_dist
            action = "drop" if same else "keep"
        elif relation == "extends":
            action = "replace"
        elif relation in ("same", "within"):
            action = "drop"
        else:
            action = "keep"
        decisions.append({**entry, "relation": relation, "action": action})
        if action == "replace":
            kept[kept.index(nearest)] = rec
        elif action == "keep":
            kept.append(rec)
    kept.sort(key=lambda r: r["t"])
    return kept, decisions


def _load_cache(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_cache(path: Path | None, cache: dict[str, dict]) -> None:
    if path is not None:
        path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def detect_slides_text(
    video: Path,
    *,
    out_dir: Path,
    cam_corner: str = "tr",
    caption: str = "bottom",
    crop: str | None = None,
    hold: float = 5.0,
    freeze_noise: str = "-50dB",
    drop_dist: int = 4,
    width_px: int = 1280,
    candidate_cap: int = 800,
    ocr_fn: Callable[[Path], dict] | None = None,
    cache_path: Path | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    """Capture the slides of `video` into `out_dir` with pixels + OCR text (see module doc).

    Returns the shape of `detect_slides_freeze` (`slides`, `flagged`, `merged`,
    `merge_flagged`) plus `decisions` (every text-signal verdict) and `ocr` (`calls`,
    `cached`, `failures`, `models`). With `cache_path`, OCR drafts are kept in a json file
    keyed by the SHA-256 of the frame, so running the same video again reads no frame twice.
    """
    ocr_fn = ocr_fn or ocr_ensemble.ocr_draft
    started = time.monotonic()
    progress = on_progress or (lambda _m: None)

    def say(msg: str) -> None:
        progress(f"[{time.monotonic() - started:5.0f}s] {msg}")

    cache = _load_cache(cache_path)
    crop_vf = slides_mod.resolve_crop_vf(video, crop, cam_corner, caption)

    periods = slides_mod._freeze_periods(video, crop_vf=crop_vf, hold=hold, noise=freeze_noise)
    # one capture mid-freeze (as the freeze pass does) and one at its end: a line edited
    # inside one long freeze is only on screen at the end
    held = [start + min((dur if dur is not None else hold) / 2, 3.0) for start, dur in periods]
    held_end = [start + dur - 0.5 for start, dur in periods if dur is not None and dur > hold]
    say(f"{len(periods)} held screens")
    small = pixels.read_gray_frames(video, crop_vf, SAMPLE_FPS)
    changes = frame_changes(small)
    cuts = cut_end_times(changes, SAMPLE_FPS)
    runs = run_end_times(changes, SAMPLE_FPS)
    say(f"{len(cuts)} stretches between cuts, {len(runs)} stable runs")
    cands = merge_candidate_times(("cut-end", cuts), ("run-end", runs),
                                  ("held-end", held_end), ("held", held))
    if len(cands) > candidate_cap:
        raise slides_mod.CandidateCapExceeded(
            f"{len(cands)} candidate frames exceeds cap {candidate_cap}; "
            "raise --hold, tighten --crop, or use --start/--end")

    out_dir.mkdir(parents=True, exist_ok=True)
    frame_glob = "[0-9][0-9][0-9][0-9]_t*.jpg"
    pre_existing = set(out_dir.glob(frame_glob))
    try:
        records = frames_mod.extract_frames(
            video, [Scene(t=t, score=1.0, kind="detected") for t, _k in cands],
            out_dir=out_dir, width_px=width_px, native=True, crop_vf=crop_vf)
        for rec in records:
            rec["abs"] = (out_dir / rec["path"]).resolve()

        texts: dict[str, str | None] = {}
        draft_models: dict[str, str] = {}
        stats = {"calls": 0, "cached": 0, "failures": 0, "models": set(), "stopped": False}
        say(f"{len(records)} candidate frames extracted")

        def text_of(rec: dict) -> str | None:
            key = rec["path"]
            if stats["stopped"] and key not in texts:
                return None  # the key was rejected: no more calls, pixels decide
            if key not in texts:
                digest = hashlib.sha256(rec["abs"].read_bytes()).hexdigest()
                if digest in cache:
                    stats["cached"] += 1
                    stats["models"].add(cache[digest]["model"])
                    texts[key] = cache[digest]["text"]
                    draft_models[key] = cache[digest]["model"]
                    return texts[key]
                stats["calls"] += 1
                try:
                    r = ocr_fn(rec["abs"])
                except ocr_ensemble.OcrError as e:
                    stats["failures"] += 1
                    if ocr_ensemble.is_key_error(e):
                        stats["stopped"] = True
                        warnings.warn("OCR stopped: the key was rejected (HTTP 4xx) -- fix the key "
                                      "file and run again; the rest is decided from pixels",
                                      RuntimeWarning, stacklevel=2)
                    warnings.warn(f"OCR failed for {rec['path']} ({ocr_ensemble.classify_failure(e)}); "
                                  "using the pixel hash for this comparison", RuntimeWarning,
                                  stacklevel=2)
                    texts[key] = None
                else:
                    stats["models"].add(r["model"])
                    texts[key] = r["text"]
                    draft_models[key] = r["model"]
                    cache[digest] = {"text": r["text"], "model": r["model"]}
                    _save_cache(cache_path, cache)
                    say(f"ocr {rec['path']}")
            return texts[key]

        kept, decisions = dedup_with_text(
            records,
            gray_of=lambda rec: pixels.gray_of_image(rec["abs"]),
            text_of=text_of,
            hash_dist=lambda a, b: slides_mod.hamming(
                slides_mod.dhash(a["abs"]), slides_mod.dhash(b["abs"])),
            drop_dist=drop_dist,
        )
    except Exception:
        for leftover in set(out_dir.glob(frame_glob)) - pre_existing:
            leftover.unlink(missing_ok=True)
        raise

    kept_paths = {rec["path"] for rec in kept}
    for rec in records:
        if rec["path"] not in kept_paths:
            rec["abs"].unlink(missing_ok=True)
    slides = [{"index": i, "t": rec["t"], "path": rec["path"], "kind": "detected"}
              for i, rec in enumerate(kept, start=1)]
    merged = [(d["against_t"], d["t"], d["blocks"], round(abs(d["t"] - d["against_t"]), 1))
              for d in decisions if d["action"] in ("drop", "replace")]
    return {
        "slides": slides, "flagged": [], "merged": merged, "merge_flagged": [],
        "decisions": decisions,
        "ocr": {"calls": stats["calls"], "cached": stats["cached"],
                "failures": stats["failures"], "models": sorted(stats["models"]),
                "stopped": stats["stopped"]},
        "drafts": {rec["path"]: {"text": texts[rec["path"]], "model": draft_models[rec["path"]]}
                   for rec in kept if texts.get(rec["path"]) is not None},
    }
