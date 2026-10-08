"""Compare a `--slides` output folder against a ground-truth slide list.

    python -m scripts.gt_compare <truth.json> <frames-dir> [--json]

Ground truth (one JSON file per sample): one row per slide the presenter showed.
A row gives the time ranges where that slide is fully on screen and not mid-transition
(`slide`), and the ranges where its last build state is on screen (`final`; omit it when
the slide has no build steps -- then it equals `slide`). Times are seconds.

Counting (docs/round-16 plan, "세는 법"):
  - a saved image belongs to the slide whose range holds its time (the time is in the
    file name, `NNNN_tMM-SS.jpg`, rounded to a second, so ranges get +-1 s of slack);
  - missing: no saved image shows the slide's final state;
  - duplicate: the 2nd and later saved images of one slide (a slide whose final state
    was never saved is missing only, not duplicated);
  - surplus: saved images that belong to no slide (demo screens, mid-transition frames).
The result depends only on the two inputs, so the same input always prints the same output.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

FRAME_RX = re.compile(r"^\d{4}_t(\d+)-(\d{2})\.jpg$")
SLACK_S = 1.0


def saved_times(frames_dir: Path) -> list[int]:
    """Seconds of every saved frame in `frames_dir`, sorted; other files are ignored."""
    times = []
    for p in frames_dir.iterdir():
        m = FRAME_RX.match(p.name)
        if m:
            times.append(int(m.group(1)) * 60 + int(m.group(2)))
    return sorted(times)


def _in(ranges: list[list[float]], t: float) -> bool:
    return any(s - SLACK_S <= t <= e + SLACK_S for s, e in ranges)


def compare(truth: dict, times: list[int]) -> dict:
    slides = truth["slides"]
    for row in slides:
        row.setdefault("final", row["slide"])
    mapped: dict[str, list[int]] = {row["id"]: [] for row in slides}
    surplus: list[int] = []
    for t in times:
        owner = next((r for r in slides if _in(r["final"], t)), None)
        owner = owner or next((r for r in slides if _in(r["slide"], t)), None)
        if owner is None:
            surplus.append(t)
        else:
            mapped[owner["id"]].append(t)
    missing, duplicate = [], 0
    for row in slides:
        ts = mapped[row["id"]]
        if not any(_in(row["final"], t) for t in ts):
            missing.append(row["id"])
        elif len(ts) > 1:
            duplicate += len(ts) - 1
    return {
        "truth": truth["id"],
        "slides": len(slides),
        "saved": len(times),
        "missing": len(missing),
        "missing_ids": missing,
        "duplicate": duplicate,
        "duplicate_rate": round(duplicate / len(slides), 4) if slides else 0.0,
        "surplus": len(surplus),
        "surplus_times": surplus,
    }


def format_report(r: dict) -> str:
    labels = {s["id"]: s.get("page", "") for s in r.get("_rows", [])}
    miss = ", ".join(f"{i}{'(' + labels[i] + ')' if labels.get(i) else ''}" for i in r["missing_ids"])
    return (
        f"truth: {r['truth']}  slides: {r['slides']}  saved: {r['saved']}\n"
        f"missing: {r['missing']}" + (f"  [{miss}]" if miss else "") + "\n"
        f"duplicate: {r['duplicate']}  rate: {r['duplicate_rate']:.2f}\n"
        f"surplus: {r['surplus']}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("truth", help="ground-truth JSON file")
    p.add_argument("frames_dir", help="--slides output folder (its frames/ directory)")
    p.add_argument("--json", action="store_true", help="print the result as JSON")
    args = p.parse_args(argv)
    truth = json.loads(Path(args.truth).read_text(encoding="utf-8"))
    frames_dir = Path(args.frames_dir)
    if not frames_dir.is_dir():
        sys.exit(f"not a directory: {frames_dir}")
    result = compare(truth, saved_times(frames_dir))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        result["_rows"] = truth["slides"]
        print(format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
