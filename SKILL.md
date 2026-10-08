---
name: video-note
description: Watch a tutorial or lecture video (URL or local path) and produce concept-first study notes. Downloads with yt-dlp, detects scene or slide changes with ffmpeg, pulls a timestamped transcript (captions, or local/cloud Whisper fallback), and writes synthesized markdown notes with inline visual evidence and timestamped traceability to the claude-watch library (OS app-data dir by default; --out-dir or CLAUDE_WATCH_LIBRARY overrides).
argument-hint: "<video-url-or-path> [topic-or-question]"
allowed-tools: Bash, Read, Write, AskUserQuestion
homepage: https://github.com/stepbyjason-lab/video-note
repository: https://github.com/stepbyjason-lab/video-note
license: MIT
user-invocable: true
---

# /video-note — Claude turns a video into study notes

You don't have a video input. This skill gives you one *and* turns each viewing into a saved notes artifact.

## Step 0 — Setup preflight (silent on success)

Run on every `/video-note` invocation:

```bash
python3 "${CLAUDE_SKILL_DIR}/scripts/setup.py" --check
```

Exit codes: `0` ready (silent — proceed), `2` missing binaries, `3` missing API key, `4` both. On non-zero, run the installer:

```bash
python3 "${CLAUDE_SKILL_DIR}/scripts/setup.py"
```

On macOS this auto-`brew install`s ffmpeg + yt-dlp. On Linux/Windows it prints the right commands. It scaffolds `~/.config/claude-watch/.env` (mode 0600 on Unix; not enforced on Windows) with commented placeholders, and the key file (`~/.config/video-note/.env.local` for an installed plugin; a development checkout uses `.env.local` in the repo root — `setup.py --json` prints the path as `key_file`, and `has_ocr_key` says whether a Gemini key is set). **All API keys live in that one key file** (see README "Bring your own keys").

A Whisper key is optional: if a local Whisper is installed (`whisper` or `whisper-ctranslate2` on PATH, or `WHISPER_LOCAL_CMD` set), it's used first and `setup.py --check` reports `whisper_backend: local` / `status: ready` with no key. Only if no local Whisper AND no key is found, use `AskUserQuestion` to ask whether the user has a Groq key (preferred — cheaper, faster) or an OpenAI key, and write it to the key file (`key_file` in `setup.py --json`). If they want neither, run with `--no-whisper`; videos without native captions will come back frames-only.

## When to use

- User pastes a tutorial / lecture / talk URL and asks to study it
- User points at a local screen recording or video and wants notes
- User types `/video-note <url-or-path> [topic]`

## How to invoke

**Step 1 — parse input.** Separate the source (URL or path) from any topic the user mentioned. The topic shapes which sections you emphasize in the notes — pass it through to your synthesis, not to the script.

**Step 1.5 — classify the video and choose extraction flags (before running).** The `--slides` decision changes how frames are extracted, so make it **now**, not after the script has run:

- **Slide lecture / seminar (prepared deck):** run with `--slides` for high-recall capture of the prepared deck. Do **not** run plain scene mode on a slide deck — that defeats the whole point of slides mode. (It is high-recall, not exhaustive — see the caveat under *Slides mode*.) Group the slides by concept when you write the notes.
- **Reel / short-form / fast-flip cards (≲90s, slide changes every few seconds):** `--slides --cam-corner none --caption none` (the card text *is* the slide). The default `--detect auto` takes the last stable frame before every cut and checks the OCR text, so fast flips are caught without tuning; on the 41s reel it caught all 7 picked cards. Only with the legacy `--detect scene` does a reel need the dense floor `--scene-threshold 0.15 --phash-dist 2` (its default misses fast transitions: 3 of 7 cards).
- **Conceptual talk without slides:** plain scene mode; organize the notes by thesis, concepts, arguments, examples, caveats, and applications.
- **Code walkthrough:** plain scene mode; organize the notes by implementation milestones, intent, design decisions, reusable patterns, and caveats.
- **Product / UI demo:** plain scene mode; organize the notes by workflow or feature area, with screenshots explaining UI states and decisions.

When you cannot tell whether a lecture uses a prepared deck, prefer `--slides`. This classification determines **both** the extraction flags here and the note structure in Step 5; after the script runs, the returned frames and transcript are treated as the raw evidence you synthesize.

**Step 2 — run the watch script** with the flags for the mode you chose in Step 1.5 (add `--slides` for slide lectures):

```bash
python3 "${CLAUDE_SKILL_DIR}/scripts/watch.py" "<source>"           # scene mode
python3 "${CLAUDE_SKILL_DIR}/scripts/watch.py" "<source>" --slides   # slide lecture
```

Optional flags:
- `--start T` / `--end T` — focus on a section (`SS`, `MM:SS`, or `HH:MM:SS`)
- `--max-frames N` — lower budget (default 80)
- `--resolution W` — bump frame width to 1024 px when on-screen text is tiny
- `--scene-threshold X` — sensitivity (default 0.30; raise for fewer cuts, lower for more)
- `--max-gap S` — coverage floor in seconds (default 45)
- `--whisper local|groq|openai` — force backend (`local` = a PATH whisper CLI or `WHISPER_LOCAL_CMD`; auto-preferred over the cloud when available)
- `--no-whisper` — disable Whisper entirely
- `--out-dir DIR` — override library root
- `--slides` — **slide-deck mode**: high-recall capture of a prepared deck (see *Slides mode* below)
- `--detect {auto,freeze,scene}` — slides detection (default `auto`): held screens + the last stable frame before every cut, then the **OCR text of the candidates as a second signal** (same slide? build step? a different slide with a near-identical look?); pixels only when no Gemini key is set. `freeze` captures one frame per held/static screen from pixels alone (skips demo scroll-noise, output ∝ held screens not duration); `scene` = legacy scene-cut + coverage floor
- `--no-ocr` — slides: no OCR at all (pixels-only detection, saved frames get no text)
- `--crop W:H:X:Y` (or `auto`) — slides freeze: explicit slide-region crop, or `auto` to detect the static slide region automatically (needed for Zoom-style recordings where the cam/chat/taskbar aren't in a corner; without it freeze can't see the slide as "frozen")
- `--hold SECONDS` — slides freeze: min seconds a screen must hold to count (default 5; lower = more recall + some held demo, higher = stricter)
- `--freeze-noise -50dB` — slides freeze: change tolerance (must be negative dB or 0..1 ratio)
- `--candidate-cap N` — slides safety cap on candidate frames (default 800)
- `--prefer-light` — slides freeze (opt-in): drop dark frames (IDE/terminal demos) by mean brightness; assumes light-background slides — leave off for dark-themed decks
- `--light-threshold N` — mean-grayscale cutoff 0–255 for `--prefer-light` (default 80)
- `--cam-corner {tr,tl,br,bl,none}` — presenter-cam corner to exclude (slides mode; default `tr`)
- `--caption {bottom,top,none}` — burned-in caption band to exclude (slides mode; default `bottom`)
- `--hi-res` — slides mode: download 1080p instead of 720p (tiny-text decks)
- `--phash-dist N` — slides dedup distance; lower keeps more near-duplicates (default 4)
- `--merge-gap S` / `--merge-dist N` — slides freeze: **time-aware merge** of animation build-steps (defaults 15s / 11). Folds a frame into the previous kept one only if it's both within `S` seconds AND *strictly* within `N` hash-distance AND the previous frame's content still LOOKS THE SAME in the new frame — a value-level pixel check, not just "still present" at the same position (a slide change that merely *hashes* close no longer merges — field evidence: uniform-template decks where two distinct slides share the same title position, dhash dist 5-10, which a presence-only check would wrongly treat as a survival) — the same screen's scroll/build progression, not a new slide. A pair landing **exactly** at `N` is not folded — it's the most fragile, crop-sensitive call this pass makes, so it's kept and surfaced as a `review: merge-threshold` line instead (field evidence: an exact-threshold fold lost a real slide). Lets you lower `--hold` for recall without build-steps inflating the count. Merges print as `merged:` lines; set EITHER to 0 to disable
- `--probe-frame` — slides only: download the source, extract ONE native-resolution frame + report its pixel dimensions, then exit (skips detect/extract/transcribe) — for measuring a `--crop W:H:X:Y` rectangle before the full run. See *LLM-assisted crop* below.
- `--probe-at TIMESTAMP` — timestamp for `--probe-frame` (`SS`, `MM:SS`, or `HH:MM:SS`); default is 25% into the video

**Step 3 — read every frame.** The script ends with a structured `=== frames ===` block listing each frame's path and timestamp. `Read` them all in parallel — they render as images in your context. In slides mode the script has already read each saved frame's text with free Gemini models: `manifest.json` `frames[].ocr` (and `ocr.json`) hold `{status, text, model}` per frame, and the `ocr_frames:` / `ocr_failed:` stdout lines say how many frames were read or failed (a failed frame is reported, not left silently empty).

**Step 4 — load the transcript.** The `=== transcript ===` block points to `transcript.json` (or `transcript.window.json` for focused mode). `Read` it — it's a list of `{t_start, t_end, text, speaker_break}`.

**Step 5 — write `notes.md` to the library directory.** Use the **concept-first study notes contract** below. The frames and transcript are raw evidence; your job is to synthesize them into a learning document. Save to `<library_dir>/notes.md`. Then print a 3-line summary to chat:
1. Title and slug
2. Number of sections + key concepts
3. Path to the notes file

Do **not** delete the library dir. It is the artifact.

## Slides mode (`--slides`)

For lecture/seminar videos where the speaker presents slides, add `--slides`:

```bash
python3 "${CLAUDE_SKILL_DIR}/scripts/watch.py" "<source>" --slides
```

This **aims to capture every prepared slide** (page 1 → last — high recall, not a guarantee).
Default detection is `--detect auto`: it captures the held (static ≥ `--hold`s) screens and the last stable
frame before every cut, so demo scroll-noise is skipped and the count tracks the slides (not video length) —
and it reads the candidates' text with the free Gemini models to decide *same slide / build step / different
slide* where pixels alone cannot (a slide that differs by a line of text; a build slide is kept at its last
step; a slide that comes back is not saved twice; talking-head footage with overlays still yields its
slides). `--detect freeze` keeps the pixels-only behavior.

- `--detect {auto,freeze,scene}` (default `auto`) — `auto` = held screens + cuts + OCR text signal (needs `GEMINI_API_KEY` in the key file, else it warns and uses pixels only); `freeze` = pixels-only held-screen capture; `scene` = legacy scene-cut + coverage floor.
- `--no-ocr` — no OCR: pixels-only detection, and the saved frames get no text.
- `--crop W:H:X:Y` — explicit slide-region crop. **Needed for Zoom-style screen recordings** where the presenter cam / chat / taskbar aren't in a corner: freeze can't detect a "frozen" slide while that chrome keeps moving. Measure the slide rectangle from one extracted frame. Overrides `--cam-corner`/`--caption`.
- `--crop auto` — instead of measuring `W:H:X:Y` by hand, sample frames and auto-detect the static slide region by trimming high-motion edge bands (presenter cam / chat / toolbar). Zero-dependency and best-effort: it works best when the moving chrome forms a clear band at the frame edges and the slide is comparatively static. It **falls back to `--cam-corner`/`--caption` (with a warning) when it can't localize a stable region** — full-screen sources, demo-heavy stretches, or **Zoom gallery layouts where the slides change continuously next to talking-head cams** (a moving face reads as slide-like content; motion and edge signals can't separate it). On those, set `--crop W:H:X:Y` from one frame — it stays the precise option, and auto never emits a *wrong* crop, it declines.
- **LLM-assisted crop — the reliable fallback when `--crop auto` declines.** A single frame is trivial for a vision model to read, even where pixel-motion can't tell a talking-head cam from the slide. So when `--crop auto` falls back, or the extracted frames still include the presenter cam / chat / toolbar, **read one extracted frame, identify the slide rectangle (the static content area, excluding cam/chat/toolbar), and re-run `--slides --crop W:H:X:Y` with those measured coordinates.** Use `--slides --probe-frame` (optionally `--probe-at TIMESTAMP`, default 25% into the video) to get that one frame cheaply — it downloads the source, extracts a single native-resolution frame, and prints the frame path + `source_resolution: WxH` (for coordinate-range sanity) then exits, skipping the full detect/extract/transcribe pass so you're not paying for a wrong (default cam-corner) crop baked into throwaway frames. This is the recommended path for Zoom gallery recordings — the automation of the "measure from one frame" step above, done by looking rather than by motion thresholds. Measure by **pixel-boundary sampling (color-transition detection), not grid eyeballing** — a blind eyeball measurement on a real recording was off by tens of pixels on two edges (~50px into the cam bar, ~40px past the chat boundary) versus a pixel-sampled re-measurement; a Sonnet subagent is sufficient for this step and was in fact more precise than the original eyeball pass, so this can be delegated cheaply. Verified end to end on a Zoom forum recording that `--crop auto` declined: the measured crop cleanly isolated the slide region (chat/toolbar/cam-bar excluded), confirming the crop-geometry problem is solved by this workflow. That said, the frame count from that run is **not** a validated unique-slide count — `--hold`-based freeze detection can still miss short slides shown near/under the hold threshold, which is a separate, unsolved recall limitation. Don't read a clean crop as proof that slide recall is also complete. `--probe-frame` only automates "get one frame"; measuring the crop rectangle from it is still your (or a subagent's) job.
- `--hold N` (default 5) — min seconds a screen must hold. Lower → more recall (also keeps held demo screens); higher → stricter (may miss briefly-shown slides).
- `--freeze-noise -50dB` (default) — freeze change tolerance; must be negative dB or a 0..1 ratio.
- `--candidate-cap N` (default 800) — safety cap on candidate frames before extraction.
- `--prefer-light` (opt-in) — after capture, drop frames whose mean brightness < `--light-threshold` to discard dark IDE/terminal **demo** screens. Assumes light-background slides; **leave off for dark-themed decks**. Removes dark demos only (light-background demos still pass — final selection is the notes step's job).
- `--light-threshold N` (default 80) — mean grayscale 0–255 cutoff for `--prefer-light`.
- `--cam-corner {tr,tl,br,bl,none}` (default `tr`) — which corner the presenter cam occupies; `none` if there is no cam. (Used when `--crop` is not given.)
- `--caption {bottom,top,none}` (default `bottom`) — burned-in caption band to ignore; `none` if there are no captions.
- `--hi-res` — download 1080p (only for decks with very small text).
- `--phash-dist N` (default 4) — dedup aggressiveness; lower keeps more near-duplicates.
- `--merge-gap S` (default 15) / `--merge-dist N` (default 11) — **time-aware merge** (freeze only): after dedup, fold a frame into the previous kept one iff it is BOTH within `S` seconds AND *strictly* within `N` hash-distance of it AND the previous frame's content still LOOKS THE SAME in the new frame (a value-level pixel check, not merely "content present at the same position" — a slide change that merely *hashes* close, e.g. two distinct slides on a uniform template sharing the same title position, no longer merges just because something is still lit up at that spot) — the animation/scroll build-steps of one screen (a bullet appearing, a panel scrolling), not a distinct slide. This lets you **lower `--hold` for recall** (catching short 3–6s slides) without the build-steps inflating the count; a plain `--phash-dist` can't express this because it is time-blind. Merges are surfaced as `merged: t=A ~ t=B (dist D, gap Gs)` lines so nothing vanishes untracked. A pair landing **exactly** at `--merge-dist` is *not* folded — it is the most fragile, crop-sensitive decision this pass makes (one hamming-distance unit separates "same build step" from "different slide", and a few pixels of crop drift is enough to flip it). Instead it's kept and surfaced as a `review: merge-threshold t=A ~ t=B (dist D, gap Gs)` line so the notes step can judge it directly. Field evidence: an exact-threshold fold silently dropped a real prepared slide on a Zoom recording, while looser folds on the same clip were safe. Set either knob to 0 to disable (restores pre-merge behavior). **Defaults are tuned on one Zoom deck** — for other presentation styles, watch the `merged:` and `review: merge-threshold` lines and tune (raise `--merge-dist` to merge more aggressively, lower it if distinct slides get merged). Field-validated: a 12-min Zoom clip at `--hold 3` went 22 → 19 (3 animation build-pairs merged, the 4 short slides `--hold 6` had missed preserved).

> **Freeze removes scroll-noise, not demo screens.** A held demo screen (terminal/IDE shown ≥ `--hold`s) is captured alongside real slides — separating "prepared slide" from "held app screen" is a content judgment for the notes step, not the extractor. For a demo-heavy seminar expect slides + some held demo frames (still far fewer than scene mode's per-interval noise).
- **Dense white-text decks, reels, and fast page-flips:** if subtle text-only slides still merge, or slides flip every few seconds (reels / short-form), run `--scene-threshold 0.15 --phash-dist 2`. A lower detection floor surfaces faint *and* fast changes as candidates and the edge-hash dedup keeps the noise bounded — validated to lift a 38-min white deck from 25 → 31 slides, and to recover a 41-second reel from **3/7 → 7/7** captured cards, neither ballooning. (Do not raise this to the global default; it is a content-specific recommendation.)
- `--slides` **cannot** be combined with `--start`/`--end` in v1 (the script errors out).

> **High-recall, not exhaustive.** The dedup uses an **edge (difference) hash** that tells monochrome / white-text slides apart far better than a plain average hash (which used to over-merge them on text decks). Slides mode also anchors the **final seconds of the video**, so a slide shown only at the very end (past the last coverage step, too late to register as a scene cut) is still extracted. It is still not a guarantee — *fast page-flips* (a slide shown for a second or two) and *near-identical build steps* can be skipped or merged. So: keep the **transcript as a parallel evidence source**, record any gaps honestly in the Slide Coverage Ledger, and for dense text decks lower `--scene-threshold` (e.g. 0.15) and/or `--phash-dist` (e.g. 2) if similar slides still merge. Field evidence: real runs on a slide lecture and the 38-min white deck above, plus embed-first coverage on a demo-heavy 155-min seminar (90 frames → ~40 inline, no "6-slide" collapse).

**Reading slides-mode output:** read every extracted frame and classify/cover it per the contract below (most prepared slides embedded `inline`), but do **not** make "one slide = one section" the default note structure. Frames are ordered by timestamp = deck order, and `slides_extracted: N` is the **raw count of extracted frames** — prepared slides plus held demo screens and the occasional non-content still (high recall, not a pre-filtered slide count). Use the slides as evidence, then group adjacent slides into the concepts, claims, frameworks, examples, or caveats they support.

Completeness and structure are separate: `--slides` protects coverage; concept-first writing protects learning quality. Do not drop prepared slides just to avoid a chronological-looking note. Instead, move each slide under the concept it supports and use an evidence caption that explains what the slide proves or exemplifies.

The stdout may print `review: near-dup t=A ~ t=B (dist D)` lines. These are borderline pairs the tool deliberately **kept** rather than risk dropping a real slide. Glance at both frames and merge their evidence only if they are genuinely the same slide.

The stdout may also print `review: merge-threshold t=A ~ t=B (dist D, gap Gs)` lines — pairs that landed *exactly* on `--merge-dist` and were preserved rather than folded by the time-aware merge pass (the most fragile call it makes). **Read both frames of the pair.** If they're the same screen's build/scroll step (a bullet appearing, a panel scrolling), treat it like a near-dup: fold it into a single evidence entry with a `ledger` disposition in the coverage ledger. If they're genuinely distinct slides, treat both as real content — cover each on its own.

Every `review:`/`merged:` line references only frames still present in the output — a line whose frame was folded away by the time-aware merge or dropped by `--prefer-light` is suppressed, so you never chase a missing frame.

## Notes template (concept-first study notes contract)

The default output is **not** a scene-by-scene screen log. Read every frame and transcript segment, then reorganize the video into a study document. The prose should teach the core thesis and concepts; screenshots should appear inline where they prove, illustrate, or clarify the idea.

For slide lectures, every frame `--slides` extracts must be accounted for and most prepared slides embedded — grouped by concept rather than by timestamp, never dropped just to make the note look less chronological, and never embedded twice. The rules below make this precise.

`--slides` output mixes three kinds of frame, and the note must classify — not silently drop — each one:

- **Prepared slide** — a deck page. **Default to embedding it `inline`** beside the concept it supports. Use `ledger` (reference-only, no image) only when an inline image adds nothing a nearby embedded slide and the prose don't already show — e.g. a near-duplicate build step, or the third near-identical slide of one concept. The test is "would the reader lose information without this image?" — if yes, `inline`.
- **Held demo / app screen** — an IDE, terminal, browser, or doc held ≥ `--hold`s, captured alongside slides (see the "Freeze removes scroll-noise, not demo screens" note above). Embed **one representative** `inline` where it concretely supports a concept; record the other near-identical frames of that same cluster as `non-slide` pointing at the representative's timestamp. **A doc / browser / Notion screen the speaker teaches *from* is content — treat it as a prepared slide (embed or `ledger`), not a demo.**
- **Non-content still** — a venue/room shot, a break screen, or a blurry transition the freeze detector happened to catch. Record it once as `non-slide` with a one-line reason; never embed it.

The three ledger statuses are **dispositions, not frame kinds**: `inline` = embedded (a prepared slide, or the one representative of a demo cluster); `ledger` = a prepared slide that is covered but not embedded; `non-slide` = an extracted frame that is not a prepared slide and is not embedded (its row carries a one-line reason). `non-slide` is not a fourth kind.

Account for **every extracted frame** so the count is auditable (`extracted = inline + ledger + non-slide`) — but **accounting is not coverage**. Most prepared slides should be `inline`; `ledger` is for genuine near-duplicates; `non-slide` is a narrow exception, **not a dumping ground**. If `non-slide` holds the bulk of the frames you have under-embedded — re-check which "demos" are actually taught-from content. When unsure whether a frame is a slide or a demo, treat it as a prepared slide. Never silently drop an extracted frame, and never pad the note with redundant demo or venue frames.

````markdown
# <Video Title>

**Source:** <URL or path>  ·  **Duration:** MM:SS  ·  **Watched:** YYYY-MM-DD

## TLDR
<3-5 sentences: the core thesis, why it matters, and the most useful takeaway.>

## Core Thesis
<The main argument or lesson of the video in plain language.>

## Concept Map
- **<concept>** — <definition and why it matters> · Evidence: `[t=MM:SS]`
- **<concept>** — <definition and relationship to another concept> · Evidence: `[t=MM:SS]`

## Learning Path

### <Concept / claim / framework name>

<Explain the idea. Start from what the speaker is teaching, not from what appears on screen.>

![](frames/0001_t00-04.jpg)

**Evidence caption:** <What this frame proves, illustrates, contrasts, or makes concrete. Do not merely describe what is visible.>

**Why it matters:** <Practical or strategic importance.>

**How to apply it:** <Concrete usage guidance, checklist, or decision rule.>

**Traceability:** `[t=MM:SS]`; transcript segment `<short excerpt or paraphrase>`.

**Additional supporting slides:** `[t=MM:SS]` `frames/0002_t00-31.jpg`; `[t=MM:SS]` `frames/0003_t01-12.jpg`. <Use this when multiple slides support the same concept but do not all need inline embeds.>

<The Slide Coverage Ledger is the single source of truth for slide accounting. `Additional supporting slides` is only a convenience pointer inside a concept section; it must not replace or duplicate ledger accounting.>

### <Next concept / claim / framework name>
...

## Frameworks, Methods, and Decision Rules
- **<framework>** — <steps, when to use it, when not to use it> · Evidence: `[t=MM:SS]`

## Examples and Applications
- **<example>** — <what it demonstrates and how to reuse it> · Evidence: `[t=MM:SS]`

## Code & Commands
<Every important code-on-screen frame's content as a runnable fenced block, language-tagged, with [t=MM:SS] back-link. Omit this section if no code appears.>

```python
# [t=03:45]
def forward(x):
    return x @ W + b
```

## Caveats and Open Questions
- <Things mentioned but not fully covered, risks, assumptions, or follow-ups.>

## Slide Coverage Ledger

<Recommended for slide lectures and dense demos. Use this to prove coverage and traceability, not as the main narrative. If using `--slides`, every **extracted** frame is accounted for exactly once: a prepared slide as `inline` (the default) or `ledger` (a genuine near-duplicate), and a held demo or non-content still as `non-slide` (with a one-line reason). The extracted count therefore equals `inline + ledger + non-slide`, so coverage is auditable and nothing is silently dropped. Statuses are dispositions, not frame kinds — most prepared slides should be `inline`; do not push slides into `ledger`/`non-slide` to shrink the note. In the last column, `inline`/`ledger` rows name the concept the slide supports; `non-slide` rows hold the one-line reason.>

| Time | Frame | Status | Supports / Reason |
|---|---|---|---|
| `[t=00:04]` | `frames/0001_t00-04.jpg` | inline | <concept/claim this embedded slide supports> |
| `[t=00:18]` | `frames/0002_t00-18.jpg` | inline | <concept/claim this embedded slide supports> |
| `[t=00:31]` | `frames/0003_t00-31.jpg` | ledger | near-duplicate build step of the `[t=00:18]` slide; covered, not re-embedded |
| `[t=20:55]` | `frames/0041_t20-55.jpg` | inline | held IDE demo, representative of the `[t=20:55]`–`[t=21:30]` cluster |
| `[t=21:30]` | `frames/0044_t21-30.jpg` | non-slide | held IDE demo; near-identical to representative `[t=20:55]` |
| `[t=75:12]` | `frames/0079_t75-12.jpg` | non-slide | venue/room shot caught by freeze; non-content |

### Optional Embedded Ledger Detail

Use this only for short decks or visually critical slides that were not already embedded inline.

![](frames/0002_t00-31.jpg)

**Evidence caption:** <What this visual supports; avoid pure screen description.>

**Transcript anchor:** <Relevant transcript excerpt, lightly cleaned.>

**Supports:** <Which concept/claim this evidence supports.>
````

## Quality gate before writing `notes.md`

Before finalizing, check the draft against this gate:

- The reader can understand the video's core thesis and concept structure from the prose; screenshots add concrete evidence and detail.
- Main sections are organized by concepts, claims, workflows, frameworks, examples, or applications — not by timestamp.
- Screenshots and transcript excerpts are evidence for ideas, not the main narrative. Important visual evidence may appear inline beside the concept it supports.
- If most paragraphs begin with what is visible on screen, rewrite the note.
- If old-template fields such as `On screen:` or `Synthesis:` appear in the main note body, rewrite the note using the concept-first contract.
- If evidence captions or visual descriptions are longer than the teaching prose across most of the document, rewrite the note.
- If slide titles became section titles without interpretation, rewrite the section titles as claims or lessons.
- If a section title is only a timestamp or a copied slide title, rewrite it as a concept, claim, workflow, or decision rule.
- If using `--slides`, record the `slides_extracted: N` value from the script's stdout and verify the Slide Coverage Ledger has exactly N rows — each extracted frame accounted for once, a prepared slide as `inline` (the default) or `ledger`, a held demo or non-content still as `non-slide` with a reason — so `N = inline + ledger + non-slide`. **Coverage check: most prepared slides must be `inline`; if `non-slide` holds the majority of frames, re-check for taught-from content mislabeled as a demo and re-embed it** (accounting every frame is not the same as covering the deck). Every timestamp a `non-slide` row names as its representative must itself appear as an `inline` row. Do not reduce slide coverage to avoid a scene-log shape, do not silently omit or `non-slide`-dump extracted frames, and do not pad the note with redundant demo/venue frames or duplicate embeds for the same slide.
- Every major claim has at least one timestamp or frame reference.
- Code, diagrams, UI states, and slide text are transcribed only when they materially support the learning goal.

If the source is genuinely a linear code walkthrough or UI demo, the main sections may follow the workflow order, but they still must explain intent, design decisions, reusable patterns, and caveats.

## Re-runs

If the user re-watches the same URL, the script reuses the cached download, transcript, and scenes. Only frames + notes regenerate. To force a full re-run, delete `<library_dir>/meta.json` first.

## Failure modes

- **Setup preflight non-zero** → run `setup.py`, then ask for a key via `AskUserQuestion`.
- **No transcript** → script emits `transcript_source: none`. Generate notes frames-only and tell the user.
- **Long video sparse-scan warning** → offer to re-run with `--start`/`--end` focused on the part the user cares about.
- **Whisper failure** → an auto-picked local Whisper that fails already falls back to the cloud automatically; otherwise retry with `--whisper openai` (if Groq failed) or vice versa. A forced `--whisper local` that's unavailable prints a notice and skips transcription (frames-only).

## Token budget

Frames dominate cost (~50-80k input tokens for 60 frames at 512 px). Transcripts are cheap. `--resolution 1024` quadruples per-frame cost — only when the user must read tiny on-screen text.

If the user asks a follow-up about a video you already watched in this session, do NOT re-run the script. The library directory is on disk; re-`Read` only the frames you need.

## Security

- Runs `yt-dlp`, `ffmpeg`, `ffprobe` locally
- Extracts mono 16 kHz audio only when captions are missing. A local Whisper (if installed) transcribes it on-device — nothing leaves the machine; otherwise the audio is sent to the Groq or OpenAI Whisper API. `WHISPER_LOCAL_CMD` runs a user-configured command with `subprocess` (no shell), so only commands the user themselves set in their env/config are executed
- Reads/writes `~/.config/claude-watch/.env` for keys (mode 0600 on Unix; Windows does not enforce file modes — protect the file with ACLs on shared machines)
- Persists artifacts to the library root (default: the OS app-data dir, e.g. `%LOCALAPPDATA%\claude-watch\library` on Windows; override — highest priority first — `--out-dir`, the `CLAUDE_WATCH_LIBRARY` env var, or the same key in `~/.config/claude-watch/.env`; pre-existing legacy `~/claude-watch/library` keeps working) — review the directory after first run if you're cautious
- Does NOT log or transmit API keys, video files, or the original URL outside the audio-to-Whisper call
