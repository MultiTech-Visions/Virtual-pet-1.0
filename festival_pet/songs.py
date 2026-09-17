"""Songs: drumline-style rhythms played on the pet's "sing" blip.

Not melodies. Two "hands" on two pitches (a high and a low blip), and the interest is all in
the rhythm: quarters, eighths, triplets, paradiddles, flams, rolls, rests. A song is a tempo,
a short list of bar patterns (with structure, AABA-ish) and its two pitches. ``compose`` makes a
new one; ``render`` turns one into audio; songs are plain dicts so they can be saved as JSON
and kept in a repertoire.
"""

from __future__ import annotations

import random

import numpy as np

from festival_pet import sounds

# One bar = 4 beats. Each hit: (beat offset, hand "R"/"L", accent 0/1). Flams are a grace hit ~35 ms before.
PATTERNS: dict[str, list[tuple[float, str, int]]] = {
    "quarters": [(0, "R", 1), (1, "L", 0), (2, "R", 0), (3, "L", 0)],
    "eighths": [(i / 2, "R" if i % 2 == 0 else "L", 1 if i % 4 == 0 else 0) for i in range(8)],
    "triplets": [(b + k / 3, ("R", "L", "R")[k] if b % 2 == 0 else ("L", "R", "L")[k], 1 if k == 0 else 0) for b in range(4) for k in range(3)],
    "paradiddle": [(i / 4, h, 1 if i % 4 == 0 else 0) for i, h in enumerate("RLRRLRLLRLRRLRLL")],
    "flams": [(0, "F", 1), (1, "L", 0), (2, "F", 1), (3, "L", 0)],  # F = flam (grace + accented hit)
    "hemiola": [(0, "R", 1), (0.75, "L", 0), (1.5, "R", 1), (2.25, "L", 0), (3, "R", 1), (3.5, "L", 0)],
    "roll_and_stop": [(i / 4, "R" if i % 2 == 0 else "L", 1 if i == 0 else 0) for i in range(12)] + [(3, "F", 1)],
    "call": [(0, "R", 1), (0.5, "R", 0), (1, "L", 1), (2.5, "R", 0), (3, "L", 1)],
    "answer": [(0, "L", 1), (1, "R", 0), (1.5, "R", 0), (2, "L", 1), (3.5, "R", 0)],
    "six_stroke": [(0, "R", 1), (0.25, "L", 0), (0.5, "L", 0), (0.75, "R", 1), (1, "R", 0), (1.25, "L", 0),
                   (2, "R", 1), (2.25, "L", 0), (2.5, "L", 0), (2.75, "R", 1), (3, "R", 0), (3.25, "L", 0)],
    "rest": [(0, "R", 1)],  # a bar of air after one hit
}
FORMS = [["A", "A", "B", "A"], ["A", "B", "A", "B"], ["A", "A", "B", "B", "A", "C"], ["A", "B", "A", "C", "A", "B", "A", "C"], ["A", "B", "C", "A"]]


def compose(rng: random.Random) -> dict:
    """A new song: tempo, form filled with patterns, two pitches, a big finish."""
    names = [n for n in PATTERNS if n not in ("rest", "roll_and_stop")]
    form = rng.choice(FORMS)
    parts = {p: rng.choice(names) for p in sorted(set(form))}
    # B (or C) is allowed to be a breath
    if len(parts) > 1 and rng.random() < 0.3:
        parts[sorted(parts)[-1]] = "rest"
    bars = [parts[p] for p in form] + ["roll_and_stop"]
    return {
        "bpm": rng.choice([88, 96, 104, 112, 120, 128]),
        "bars": bars,
        "hi": rng.choice([1500, 1600, 1700, 1800]),
        "lo": rng.choice([950, 1000, 1100, 1200]),
        "name": rng.choice(["rat-a-tat", "boop cadence", "tenor line", "flam city", "little march", "roll call", "pip pip"]) + f" #{rng.randint(10, 99)}",
    }


def hits(song: dict) -> list[tuple[float, str, int]]:
    """(time s, hand, accent) for the whole song, flams expanded into grace + accented hit."""
    period = 60.0 / song["bpm"]
    out = []
    for b, name in enumerate(song["bars"]):
        for off, hand, acc in PATTERNS[name]:
            t = (b * 4 + off) * period
            if hand == "F":
                out.append((t - 0.035, "L", 0))
                out.append((t, "R", 1))
            else:
                out.append((t, hand, acc))
    return out


def duration(song: dict) -> float:
    return len(song["bars"]) * 4 * 60.0 / song["bpm"] + 0.4


def render(song: dict, sample_rate: int = sounds.SAMPLE_RATE) -> np.ndarray:
    """Audio for a song: each hit is a short blip on the hand's pitch, accents louder and a touch longer."""
    n = int(duration(song) * sample_rate)
    out = np.zeros(n, dtype=np.float32)
    for t, hand, acc in hits(song):
        f = song["hi"] if hand == "R" else song["lo"]
        blip = sounds.tone(f * (1.03 if acc else 1.0), 0.075 if acc else 0.055, sample_rate, harmonics=0.35, attack=0.02, release=0.25)
        blip = blip * (0.8 if acc else 0.5)
        i = int(max(0.0, t) * sample_rate)
        j = min(n, i + len(blip))
        out[i:j] += blip[: j - i]
    peak = float(np.max(np.abs(out)))
    if peak > 0.8:
        out *= 0.8 / peak
    return out


def describe(song: dict) -> str:
    return f"{song['name']}: {song['bpm']} bpm, {' '.join(song['bars'])}"
