"""Read the text of every saved slide frame (free vision models, writer + judge).

`ocr_frames` runs in one process, one frame at a time: the ladder state (cooldowns, pacing)
lives in the process and a free model's requests-per-minute limit is per model, so frames are
never read in parallel. A frame whose OCR failed is reported as failed with the reason; it is
never dropped and never turned into empty text.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from scripts import ocr_ensemble


def ocr_frames(
    frames: list[dict],
    *,
    base_dir: Path,
    drafts: dict[str, dict] | None = None,
    cache_path: Path | None = None,
    ocr_fn: Callable[..., dict] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, dict]:
    """OCR each frame record (`{"path": "frames/0001_t00-03.jpg", ...}`, relative to `base_dir`).

    `drafts` maps a frame's file name to the writer draft detection already made (`{"text",
    "model"}`), so the writer is not called twice. Results are cached by the SHA-256 of the
    frame in `cache_path`, so re-running a video reads nothing again. Returns
    `{path: {"status": "ok", "text", "model", ["partial"]} | {"status": "failed", "reason",
    "error"}}`.
    """
    ocr_fn = ocr_fn or ocr_ensemble.ocr_image_ensemble
    say = on_progress or (lambda _m: None)
    drafts = drafts or {}
    cache = _load(cache_path)
    ocr_ensemble.reset_wait_budget()
    results: dict[str, dict] = {}
    stopped = False
    for rec in frames:
        if stopped:  # the key was rejected: stop calling, say so for every remaining frame
            results[rec["path"]] = {"status": "failed", "reason": "key_error",
                                    "error": "OCR stopped after the key was rejected"}
            continue
        path = base_dir / rec["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in cache:
            results[rec["path"]] = cache[digest]
            continue
        try:
            r = ocr_fn(path, draft=drafts.get(path.name))
        except ocr_ensemble.OcrError as e:
            results[rec["path"]] = {"status": "failed", "reason": ocr_ensemble.classify_failure(e),
                                    "error": str(e)}
            say(f"ocr FAILED {rec['path']}: {results[rec['path']]['reason']}")
            if ocr_ensemble.is_key_error(e):
                stopped = True
                say("OCR stopped: the key was rejected (HTTP 4xx) -- fix the key file and run again")
            continue
        entry = {"status": "ok", "text": r["text"], "model": r["model"]}
        if r.get("partial"):
            entry["partial"] = True
        results[rec["path"]] = entry
        cache[digest] = entry
        _save(cache_path, cache)
        say(f"ocr {rec['path']}")
    return results


def _load(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save(path: Path | None, cache: dict[str, dict]) -> None:
    if path is not None:
        path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
