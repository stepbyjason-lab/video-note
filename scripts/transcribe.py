"""Caption extraction (VTT) + dedupe + speaker-break heuristic + Whisper orchestration."""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

from scripts import whisper
from scripts.download import ytdlp_cmd

# Cloud Whisper endpoints reject a single request body over 25 MB. Mono 16 kHz
# ~64 kbps audio is ~0.5 MB/min, so anything past ~50 min must be split. Local
# Whisper has no such cap and is never chunked.
CLOUD_MAX_BYTES = 24_000_000   # a small margin under the 25 MB per-request cap
CHUNK_SECONDS = 1200           # 20 min ~= 10 MB/chunk at ~64 kbps: safely under the cap
# A chunk's cue times must be chunk-relative (0-based); if a backend ever returned
# original-timeline stamps, adding the offset would double them. Reject any cue whose
# end exceeds the chunk's own duration by more than this slack (whisper can round a
# final segment a hair past the audio end).
_CHUNK_TS_TOLERANCE_S = 2.0

_TS_RX = re.compile(
    r"(?:(\d+):)?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?\s*-->\s*"
    r"(?:(\d+):)?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?"
)


def _ts_to_s(h: str | None, m: str, s: str, ms: str | None) -> float:
    return (int(h) if h else 0) * 3600 + int(m) * 60 + int(s) + (int(ms) / 1000.0 if ms else 0.0)


def parse_vtt(text: str) -> list[dict]:
    """Parse a WebVTT file into cues. Strips formatting tags like <c.colorXXXX>."""
    cues: list[dict] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = _TS_RX.search(lines[i])
        if not m:
            i += 1
            continue
        t_start = _ts_to_s(m.group(1), m.group(2), m.group(3), m.group(4))
        t_end = _ts_to_s(m.group(5), m.group(6), m.group(7), m.group(8))
        i += 1
        text_lines: list[str] = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(re.sub(r"<[^>]+>", "", lines[i]).strip())
            i += 1
        joined = " ".join(t for t in text_lines if t).strip()
        if joined:
            cues.append({"t_start": t_start, "t_end": t_end, "text": joined})
    return cues


_ROLLING_OVERLAP_MIN = 10  # chars of suffix/prefix overlap to count as a rolling caption


def dedupe_cues(cues: list[dict]) -> list[dict]:
    """Collapse adjacent rolling/duplicate cues from VTT (especially YouTube auto-caps).

    Handles four patterns:
    1. Identical text → extend prev's t_end.
    2. `c.text` starts with `prev.text` (rolling extension) → replace prev's text with c's longer text.
    3. `prev.text` ends with `c.text` (c is a tail already shown) → extend prev's t_end, drop c.
    4. Suffix of prev matches prefix of c (>= 10 chars) → emit only the new tail of c.
    Otherwise keep both as separate cues.
    """
    out: list[dict] = []
    for c in cues:
        if not out:
            out.append(dict(c))
            continue
        prev = out[-1]
        # 1. Identical
        if prev["text"] == c["text"]:
            prev["t_end"] = max(prev["t_end"], c["t_end"])
            continue
        # 2. Rolling extension: c is the longer continuation of prev
        if len(c["text"]) > len(prev["text"]) and c["text"].startswith(prev["text"]):
            prev["t_end"] = c["t_end"]
            prev["text"] = c["text"]
            continue
        # 3. c is already contained at the tail of prev
        if prev["text"].endswith(c["text"]):
            prev["t_end"] = max(prev["t_end"], c["t_end"])
            continue
        # 4. Suffix of prev = prefix of c → emit only the new tail
        max_check = min(len(prev["text"]), len(c["text"]))
        overlap = 0
        for k in range(max_check, _ROLLING_OVERLAP_MIN - 1, -1):
            if prev["text"][-k:] == c["text"][:k]:
                overlap = k
                break
        if overlap > 0:
            tail = c["text"][overlap:].lstrip()
            if tail:
                out.append({"t_start": c["t_start"], "t_end": c["t_end"], "text": tail})
            else:
                prev["t_end"] = max(prev["t_end"], c["t_end"])
            continue
        # 5. Unrelated → keep both
        out.append(dict(c))
    return out


def insert_speaker_breaks(cues: list[dict], threshold_s: float = 2.0) -> list[dict]:
    """Mark cues that follow a pause > threshold as a speaker_break.

    This is a heuristic — it doesn't identify WHO is speaking, just that
    something changed. Useful for interviews/panels where downstream notes
    benefit from a section boundary.
    """
    out: list[dict] = []
    for i, c in enumerate(cues):
        marked = dict(c)
        if i > 0:
            gap = c["t_start"] - cues[i - 1]["t_end"]
            marked["speaker_break"] = gap > threshold_s
        else:
            marked["speaker_break"] = False
        out.append(marked)
    return out


def slice_to_window(
    cues: list[dict], start_s: Optional[float], end_s: Optional[float]
) -> list[dict]:
    """Keep only cues that overlap the [start, end] window. None bounds = open."""
    if start_s is None and end_s is None:
        return cues
    s = start_s if start_s is not None else float("-inf")
    e = end_s if end_s is not None else float("inf")
    return [c for c in cues if c["t_end"] >= s and c["t_start"] <= e]


def fetch_native_captions(video_url: str, work_dir: Path) -> Optional[Path]:
    """Try to pull native + auto-generated subs via yt-dlp. Returns the .vtt path or None."""
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        *ytdlp_cmd(),
        "--skip-download",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", "en.*",
        "--sub-format", "vtt",
        "-o", str(work_dir / "%(id)s.%(ext)s"),
        video_url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return None
    vtts = sorted(work_dir.glob("*.vtt"))
    return vtts[0] if vtts else None


def extract_audio_for_whisper(video: Path, out_audio: Path) -> None:
    """Mono 16kHz audio for Whisper. ~0.5MB/min — well under the 25MB limit for ~50 min."""
    out_audio.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-nostdin",
            "-i", str(video),
            "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k",
            str(out_audio),
        ],
        check=True,
    )


_CTRL_RX = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _clean_output(s: str, *, limit: int = 500) -> str:
    """Strip control chars (ANSI escapes, CR/LF) from a subprocess's output before
    it lands in an error message — prevents terminal/log injection from a
    misbehaving local Whisper command."""
    return _CTRL_RX.sub(" ", s).strip()[:limit]


def _build_local_cmd(spec: whisper.LocalSpec, audio: Path, work_dir: Path) -> list[str]:
    """Turn a detect_local_whisper() spec into an argv list.

    - kind="cli": a standard Whisper CLI (openai-whisper / whisper-ctranslate2),
      invoked with --model/--output_dir/--output_format and auto language.
    - kind="custom": a user command. The template is tokenised first, then
      {audio}/{outdir} are substituted *per token* (so a path with spaces never
      re-splits and cannot inject extra argv elements). Whichever placeholder is
      absent is appended: missing {audio} → the audio path; missing {outdir} →
      `--outdir <dir>`.
    """
    if spec.get("kind") == "cli":
        return [
            spec["bin"], str(audio),
            "--model", spec.get("model") or whisper.DEFAULT_LOCAL_MODEL,
            "--output_dir", str(work_dir),
            "--output_format", "srt",
        ]
    template = spec.get("template") or ""
    if not template.strip():
        raise whisper.WhisperError("WHISPER_LOCAL_CMD is set but empty")
    tokens = whisper.parse_local_cmd(template)
    if not tokens:
        raise whisper.WhisperError(
            f"WHISPER_LOCAL_CMD could not be parsed (unbalanced quotes?): {template[:80]}"
        )
    has_audio = any("{audio}" in t for t in tokens)
    has_outdir = any("{outdir}" in t for t in tokens)
    cmd = [t.replace("{audio}", str(audio)).replace("{outdir}", str(work_dir)) for t in tokens]
    if not has_audio:
        cmd.append(str(audio))
    if not has_outdir:
        cmd += ["--outdir", str(work_dir)]
    return cmd


def _find_srt(work_dir: Path, stem: str) -> Optional[Path]:
    """Locate the SRT a local Whisper wrote: <stem>.srt first, else the most
    recently modified *.srt in the work dir."""
    exact = work_dir / f"{stem}.srt"
    if exact.exists():
        return exact
    srts = list(work_dir.glob("*.srt"))
    return max(srts, key=lambda p: p.stat().st_mtime) if srts else None


def transcribe_local(audio: Path, work_dir: Path, *, spec: whisper.LocalSpec) -> list[dict]:
    """Run a local Whisper (custom command or standard CLI) and parse its SRT.

    The produced SubRip is parsed by ``parse_vtt`` — its timestamp grammar
    already accepts comma millisecond separators, so SRT cues parse identically
    to WebVTT.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_local_cmd(spec, audio, work_dir)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False,
            encoding="utf-8", errors="replace",
        )
    except OSError as e:
        # Missing/unrunnable binary (FileNotFoundError on Windows, etc.) — surface
        # as WhisperError so callers fall back to cloud instead of crashing.
        raise whisper.WhisperError(f"local whisper command not runnable: {cmd[0]} ({e})") from e
    if proc.returncode != 0:
        detail = _clean_output(proc.stderr or proc.stdout or "")
        raise whisper.WhisperError(f"local whisper exited {proc.returncode}: {detail}")
    srt = _find_srt(work_dir, audio.stem)
    if srt is None:
        hint = _clean_output(proc.stdout or "", limit=300)
        raise whisper.WhisperError(
            f"local whisper produced no SRT in {work_dir}" + (f"; stdout: {hint}" if hint else "")
        )
    try:
        text = srt.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        # SRT vanished/unreadable between detection and read (concurrent cleanup,
        # broken symlink, network fs) — keep it inside the WhisperError chain so
        # the caller can fall back instead of crashing.
        raise whisper.WhisperError(f"could not read SRT output {srt}: {e}") from e
    cues = parse_vtt(text)
    if not cues:
        raise whisper.WhisperError("local whisper SRT contained no cues")
    return cues


def _segment_audio(audio: Path, work_dir: Path, *, seconds: int = CHUNK_SECONDS) -> list[Path]:
    """Split `audio` into <=`seconds`-long pieces with ffmpeg's segment muxer.

    Returns the chunk paths in play order (`<stem>_chunk_000<suffix>`, ...). Uses
    `-c copy` (no re-encode) so it's fast and lossless, and `-reset_timestamps 1` so
    each chunk's PTS starts at 0 — the cloud backend then transcribes it as standalone
    0-based audio, which is what the caller's offset arithmetic assumes. Chunks are
    enumerated by their deterministic `%03d` names (not a glob) so a stem containing
    glob metacharacters can't drop chunks. An ffmpeg failure is re-raised as a
    `WhisperError` (mirroring `transcribe_local`) so the cloud attempt degrades through
    the normal fallback path instead of crashing the run.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(work_dir / f"{audio.stem}_chunk_%03d{audio.suffix}")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-nostdin",
                "-i", str(audio),
                "-f", "segment", "-segment_time", str(seconds),
                "-reset_timestamps", "1", "-c", "copy",
                pattern,
            ],
            check=True,
        )
    except (subprocess.CalledProcessError, OSError) as e:
        raise whisper.WhisperError(f"audio segmentation failed for {audio.name}: {e}") from e
    chunks: list[Path] = []
    i = 0
    while True:
        chunk = work_dir / f"{audio.stem}_chunk_{i:03d}{audio.suffix}"
        if not chunk.exists():
            break
        chunks.append(chunk)
        i += 1
    return chunks


def _probe_duration(path: Path) -> float:
    """Seconds of audio in `path` via ffprobe. Raises WhisperError on any failure — a
    bad/missing duration would silently misalign stitched cue timestamps, so fail loudly
    instead (the cloud attempt then surfaces the error and returns no cues)."""
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=nokey=1:noprint_wrappers=1", str(path),
            ],
            capture_output=True, text=True, check=False,
        )
    except OSError as e:
        # missing/unrunnable ffprobe (mirrors _segment_audio) -> degrade via fallback,
        # not a raw OSError crash.
        raise whisper.WhisperError(f"ffprobe not runnable for {path.name}: {e}") from e
    raw = (proc.stdout or "").strip()
    try:
        return float(raw)
    except ValueError as e:
        detail = raw or (proc.stderr or "").strip()[:200]
        raise whisper.WhisperError(
            f"ffprobe could not read duration of chunk {path.name}: {detail!r}"
        ) from e


def transcribe_cloud(
    audio: Path,
    *,
    transcribe_one: Callable[[Path], list[dict]],
    work_dir: Path,
) -> list[dict]:
    """Transcribe via a cloud backend, splitting first if the audio exceeds the
    per-request size cap. Small files (or a path that doesn't exist yet) take the
    single-request path unchanged. For a split, each chunk is transcribed on its own
    (Whisper returns chunk-relative times) and its cues are shifted by the running sum
    of preceding chunk durations, then concatenated in order.

    `transcribe_one` is the single-file cloud call (e.g. `whisper.transcribe_groq` bound
    to its key). A chunk failure propagates (the whole attempt fails, consistent with the
    existing all-or-nothing cloud semantics); chunk files are always cleaned up.
    """
    try:
        oversize = audio.stat().st_size > CLOUD_MAX_BYTES
    except OSError:
        # Any stat failure (missing path, permission, locked/flaky FS) -> take the
        # single-request path. That's a safe degrade: the read failure (if real) then
        # surfaces as a WhisperError inside the cloud client instead of a raw OSError
        # crash here, matching how _segment_audio/_probe_duration also wrap OS errors.
        oversize = False
    if not oversize:
        return transcribe_one(audio)

    chunks = _segment_audio(audio, work_dir)
    if not chunks:
        raise whisper.WhisperError(f"audio segmentation produced no chunks for {audio}")
    cues: list[dict] = []
    offset = 0.0
    try:
        for chunk in chunks:
            part = transcribe_one(chunk)
            dur = _probe_duration(chunk)
            for c in part:
                if c["t_end"] > dur + _CHUNK_TS_TOLERANCE_S:
                    raise whisper.WhisperError(
                        f"chunk {chunk.name}: cue end {c['t_end']:.1f}s exceeds chunk "
                        f"duration {dur:.1f}s — backend returned non-chunk-relative "
                        "timestamps, stitched offsets would be wrong"
                    )
                cues.append({
                    "t_start": c["t_start"] + offset,
                    "t_end": c["t_end"] + offset,
                    "text": c["text"],
                })
            offset += dur
    finally:
        for chunk in chunks:
            # best-effort cleanup: a locked/undeletable chunk must not clobber a real
            # WhisperError raised in the try above (which the fallback layer catches).
            try:
                chunk.unlink(missing_ok=True)
            except OSError:
                pass
    return cues


GROQ_ASPH_MODELS = (whisper.GROQ_MODEL, whisper.GROQ_TURBO_MODEL)
RETRY_AFTER_CAP_S = 30.0  # honor a Retry-After wait only up to this; longer (ASPH-hourly) -> switch buckets instead


class _GroqAsphSpreader:
    """Per-chunk Groq transcribe that spreads load across the two Whisper ASPH buckets.

    Groq's free tier limits audio-seconds-per-hour (~7200s = 2h) *per model*, and
    ``whisper-large-v3`` / ``whisper-large-v3-turbo`` have separate buckets. A lecture
    long enough to exhaust one model's bucket (429) is retried on the other, roughly
    doubling the free-tier budget. A rate-limited model is moved to the back of the
    preference order so later chunks don't waste a request on the exhausted bucket.
    Non-429 errors (bad key, network, other HTTP) propagate immediately — falling back
    wouldn't help. If every bucket is rate-limited, raises a WhisperError.
    """

    def __init__(self, api_key: str, models: tuple[str, ...] = GROQ_ASPH_MODELS) -> None:
        self._api_key = api_key
        self._order = list(models)

    def _attempt(self, audio: Path, model: str) -> list[dict]:
        """Transcribe one chunk on one model. On a 429 carrying a short Retry-After
        (<= RETRY_AFTER_CAP_S — a transient RPM/TPM limit), wait it out once and retry
        the SAME model. A long or absent Retry-After (ASPH's hourly reset) re-raises
        immediately so __call__ switches buckets instead of hanging."""
        try:
            return whisper.transcribe_groq(audio, api_key=self._api_key, model=model)
        except whisper.WhisperError as e:
            if e.status_code != 429:
                raise
            ra = e.retry_after
            if ra is None or not (0 < ra <= RETRY_AFTER_CAP_S):
                raise
            time.sleep(ra)
            return whisper.transcribe_groq(audio, api_key=self._api_key, model=model)

    def __call__(self, audio: Path) -> list[dict]:
        errors: list[str] = []
        for model in list(self._order):  # iterate a copy; __call__ may reorder self._order
            try:
                return self._attempt(audio, model)
            except whisper.WhisperError as e:
                if e.status_code != 429:
                    raise
                errors.append(f"{model}: {e}")
                # demote the rate-limited bucket so the next chunk tries a fresh one first
                self._order.remove(model)
                self._order.append(model)
        raise whisper.WhisperError(
            f"all Groq ASPH buckets rate-limited (429) for {audio.name}: " + "; ".join(errors)
        )


def transcribe_via_whisper(
    audio: Path,
    *,
    backend: str,
    groq_key: Optional[str],
    openai_key: Optional[str],
    local_spec: Optional[whisper.LocalSpec] = None,
    work_dir: Optional[Path] = None,
) -> list[dict]:
    if backend == "local":
        if not local_spec:
            raise whisper.WhisperError("local backend selected but no local Whisper is available")
        return transcribe_local(audio, work_dir or audio.parent, spec=local_spec)
    if backend == "groq":
        if not groq_key:
            raise whisper.WhisperError("Groq backend selected but GROQ_API_KEY is unset")
        return transcribe_cloud(
            audio,
            transcribe_one=_GroqAsphSpreader(groq_key),
            work_dir=work_dir or audio.parent,
        )
    if backend == "openai":
        if not openai_key:
            raise whisper.WhisperError("OpenAI backend selected but OPENAI_API_KEY is unset")
        return transcribe_cloud(
            audio,
            transcribe_one=lambda a: whisper.transcribe_openai(a, api_key=openai_key),
            work_dir=work_dir or audio.parent,
        )
    raise whisper.WhisperError(f"Unknown backend: {backend}")


def transcribe_with_fallback(
    audio: Path,
    *,
    backend: str,
    groq_key: Optional[str],
    openai_key: Optional[str],
    forced: Optional[str],
    local_spec: Optional[whisper.LocalSpec],
    work_dir: Optional[Path] = None,
    on_message: Optional[Callable[[str], None]] = None,
) -> list[dict]:
    """Transcribe with the chosen backend; if an *auto-picked* local Whisper
    fails, fall back to a cloud backend. Returns cues (empty list if everything
    fails). A forced backend (``--whisper ...``) never falls back.

    ``on_message`` receives human-readable progress/error lines (e.g. a stderr
    printer); failures are reported through it rather than raised.
    """
    def _say(msg: str) -> None:
        if on_message:
            on_message(msg)

    try:
        return transcribe_via_whisper(
            audio, backend=backend, groq_key=groq_key, openai_key=openai_key,
            local_spec=local_spec, work_dir=work_dir,
        )
    except whisper.WhisperError as e:
        _say(f"Whisper failed ({backend}): {e}")
        if backend == "local" and forced is None:
            fb = whisper.pick_backend(
                groq_key=groq_key, openai_key=openai_key, forced=None, local_available=False,
            )
            if fb:
                _say(f"Falling back to {fb}.")
                try:
                    return transcribe_via_whisper(
                        audio, backend=fb, groq_key=groq_key, openai_key=openai_key,
                    )
                except whisper.WhisperError as e2:
                    _say(f"Whisper fallback failed ({fb}): {e2}")
        return []
