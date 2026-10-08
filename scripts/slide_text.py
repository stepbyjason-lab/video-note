"""Decide from two OCR texts whether two frames show the same slide.

Slide detection uses pixels first; this is the second signal. When the pixels differ only a
little, the texts say what the difference is:
  same       the same words are on both frames            -> one slide
  extends    the 2nd text is the 1st plus more words      -> a build step: the 2nd is later/final
  within     the 2nd text is part of the 1st              -> the 2nd is an earlier/partial state
  different  each has words the other lacks               -> two slides
  unknown    one side has no usable text                  -> the caller falls back to pixels

The comparison is on the set of words, not on lines: two OCR runs of one slide wrap and break
lines differently, so lines are not a stable unit. A word with a digit must match exactly (a
changed number is a changed slide); other words tolerate one OCR slip.
"""
from __future__ import annotations

import re

COVER = 0.95        # share of one text's words that must be found in the other for "same"
EXTRA_WORDS = 2     # "extends"/"within" need at least this many words the other text lacks...
EXTRA_SHARE = 0.10  # ...and this share of the shorter text
MIN_WORDS = 2       # a text with fewer usable words is "unknown"
FUZZY_MIN_LEN = 5   # words at least this long may differ by one OCR slip

_ZERO_WIDTH = str.maketrans("", "", "​‌‍﻿")
_WORD = re.compile(r"\w+", re.UNICODE)
_HAS_DIGIT = re.compile(r"\d")


def words(text: str) -> set[str]:
    """Distinct lower-cased words of `text`, two characters or more (so list numbers `1.` drop out)."""
    out = set()
    for w in _WORD.findall(text.translate(_ZERO_WIDTH).lower()):
        w = w.strip("_")
        if len(w) >= 2:
            out.add(w)
    return out


def _within_one_slip(a: str, b: str) -> bool:
    """True when `a` and `b` differ by one substitution, insertion or deletion."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    if i == len(a):
        return True  # a is a prefix of b (b is at most one longer)
    if len(a) == len(b):
        return a[i + 1:] == b[i + 1:]
    return a[i:] == b[i + 1:]


def _found(word: str, pool: set[str]) -> bool:
    if word in pool:
        return True
    if len(word) < FUZZY_MIN_LEN or _HAS_DIGIT.search(word):
        return False
    return any(len(w) >= FUZZY_MIN_LEN and not _HAS_DIGIT.search(w) and _within_one_slip(word, w)
               for w in pool)


def _missing(src: set[str], dst: set[str]) -> int:
    """How many words of `src` have no match in `dst`."""
    return sum(1 for w in src if not _found(w, dst))


def compare_texts(first: str, second: str) -> str:
    """How `second` relates to `first`: same / extends / within / different / unknown."""
    a, b = words(first), words(second)
    if len(a) < MIN_WORDS or len(b) < MIN_WORDS:
        return "unknown"
    a_lacks, b_lacks = _missing(a, b), _missing(b, a)   # a's words not in b; b's words not in a
    a_covered = 1 - a_lacks / len(a)
    b_covered = 1 - b_lacks / len(b)
    if a_covered >= COVER and b_covered >= COVER:
        return "same"
    if a_covered >= COVER and b_lacks >= max(EXTRA_WORDS, EXTRA_SHARE * len(a)):
        return "extends"
    if b_covered >= COVER and a_lacks >= max(EXTRA_WORDS, EXTRA_SHARE * len(b)):
        return "within"
    return "different"
