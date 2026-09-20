"""Songs it makes up. Two styles, both built from the same little dict so they can be saved as JSON.

``drumline`` is rhythm, not melody: two "hands" on two pitches (a high and a low blip), and the
interest is all in the playing — quarters, eighths, triplets, paradiddles, flams, rolls, rests.

``bass`` is the robot's idea of bass music: wubs (a note whose filter opens and shuts), wah-wahs,
lasers, a riser and one drop per song, at a halftime tempo. The speaker cannot reproduce anything
under ~300 Hz, so the bass is implied by a harmonic-rich note up where the speaker works rather
than by an actual low note, and the dubstep stays light.

``compose`` makes one; ``render`` turns one into audio.
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

# Bass moves, one per bar of 4 beats. "drop" and "riser" are the structure; the rest are the voice.
BASS_MOVES = ("wub4", "wub2", "wub_fast", "wahs", "pews", "stabs", "riser", "drop", "rest")
BASS_VOICES = ("wub4", "wub2", "wub_fast", "wahs", "pews", "stabs")
BASS_FORMS = [["A", "A", "B", "A"], ["A", "B", "A", "B"], ["A", "A", "B", "C"], ["A", "B", "A", "C"]]
STYLES = ("drumline", "bass")


def compose(rng: random.Random, style: str | None = None) -> dict:
    """A new song. ``style`` is "drumline", "bass", or None to pick one."""
    style = style if style is not None else rng.choice(STYLES)
    if style not in STYLES:
        raise ValueError(f"unknown song style '{style}'")
    if style == "bass":
        return _compose_bass(rng)
    names = [n for n in PATTERNS if n not in ("rest", "roll_and_stop")]
    form = rng.choice(FORMS)
    parts = {p: rng.choice(names) for p in sorted(set(form))}
    # B (or C) is allowed to be a breath
    if len(parts) > 1 and rng.random() < 0.3:
        parts[sorted(parts)[-1]] = "rest"
    bars = [parts[p] for p in form] + ["roll_and_stop"]
    return {
        "style": "drumline",
        "bpm": rng.choice([88, 96, 104, 112, 120, 128]),
        "bars": bars,
        "hi": rng.choice([1500, 1600, 1700, 1800]),
        "lo": rng.choice([950, 1000, 1100, 1200]),
        "name": rng.choice(["rat-a-tat", "boop cadence", "tenor line", "flam city", "little march", "roll call", "pip pip"]) + f" #{rng.randint(10, 99)}",
    }


def _compose_bass(rng: random.Random) -> dict:
    """Halftime, a riser into one drop, and wubs either side of it."""
    form = rng.choice(BASS_FORMS)
    parts = {p: rng.choice(BASS_VOICES) for p in sorted(set(form))}
    bars = [parts[p] for p in form]
    where = rng.randrange(2, len(bars))  # one drop per song, with a couple of bars of run-up before the riser
    bars[where - 1] = "riser"
    bars[where] = "drop"
    bars = bars + [rng.choice(BASS_VOICES), "rest"]
    return {
        "style": "bass",
        "bpm": rng.choice([70, 74, 78, 82, 86]),  # halftime: the wubs sit on the half-beats
        "bars": bars,
        "hi": rng.choice([1800, 2100, 2400]),  # the lasers
        "lo": rng.choice([310, 330, 350, 370]),  # the "bass": as low as the speaker can still carry
        "name": rng.choice(["wub study", "the drop", "bass face", "wa-wah", "pew pew pew", "filthy (politely)", "sub sandwich"]) + f" #{rng.randint(10, 99)}",
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


def _mix(out: np.ndarray, part: np.ndarray, t: float, sample_rate: int, gain: float = 1.0) -> None:
    i = int(max(0.0, t) * sample_rate)
    j = min(len(out), i + len(part))
    if j > i:
        out[i:j] += part[: j - i] * gain


def _render_bass_bar(out: np.ndarray, move: str, t0: float, period: float, song: dict, sample_rate: int) -> None:
    """One bar of bass music at ``t0``. ``period`` is one beat; a bar is four."""
    lo, hi = float(song["lo"]), float(song["hi"])
    bar = 4 * period
    if move == "wub4":  # one wub a beat, wobbling twice a beat
        for k in range(4):
            _mix(out, sounds.wub(period * 0.95, lo, rate=2.0 / period, sample_rate=sample_rate), t0 + k * period, sample_rate, 0.85)
    elif move == "wub2":  # two long ones, slower wobble: the lazy half-time wub
        for k in range(2):
            _mix(out, sounds.wub(2 * period * 0.95, lo, rate=1.5 / period, sample_rate=sample_rate), t0 + k * 2 * period, sample_rate, 0.85)
    elif move == "wub_fast":  # the busy one: triplet-ish wobble over two long notes
        for k in range(2):
            _mix(out, sounds.wub(2 * period * 0.95, lo * 1.5, rate=3.0 / period, sample_rate=sample_rate), t0 + k * 2 * period, sample_rate, 0.7)
    elif move == "wahs":  # wah, wa-wah, wa-wah
        _mix(out, sounds.wah(period * 0.9, lo * 1.2, sample_rate, up=True), t0, sample_rate, 0.8)
        for k in (1.0, 1.5, 2.5, 3.0):
            _mix(out, sounds.wah(period * 0.45, lo * (1.2 if k % 1 else 1.0), sample_rate, swings=2.0), t0 + k * period, sample_rate, 0.7)
    elif move == "pews":  # lasers down the bar
        for k in range(5):
            _mix(out, sounds.zap(0.11, hi, lo * 1.3, sample_rate), t0 + k * period * 0.75, sample_rate, 0.6)
    elif move == "stabs":  # short wub stabs on the offbeats
        for k in (0.0, 0.75, 1.5, 2.0, 2.75, 3.5):
            _mix(out, sounds.wub(period * 0.4, lo, rate=1.0 / period, sample_rate=sample_rate), t0 + k * period, sample_rate, 0.75)
    elif move == "riser":  # up into the drop, with a laser fill
        _mix(out, sounds.chirp(lo * 1.2, hi, bar * 0.92, sample_rate, curve=2.0, attack=0.1, release=0.05), t0, sample_rate, 0.5)
        for k in range(3):
            _mix(out, sounds.zap(0.08, hi * 1.1, hi * 0.6, sample_rate), t0 + bar * (0.55 + 0.12 * k), sample_rate, 0.4)
    elif move == "drop":  # a beat of air, then the big one (light: one bar, then back to the tune)
        _mix(out, sounds.wub(3 * period, lo * 0.92, rate=1.25 / period, sample_rate=sample_rate, depth=1.3), t0 + period, sample_rate, 1.0)
    elif move != "rest":
        raise KeyError(f"unknown bass move '{move}'")


def render(song: dict, sample_rate: int = sounds.SAMPLE_RATE) -> np.ndarray:
    """Audio for a song. Drumline: a short blip per hit, accents louder and a touch longer.
    Bass: one move per bar, wubs and wahs and lasers."""
    style = song["style"]
    n = int(duration(song) * sample_rate)
    out = np.zeros(n, dtype=np.float32)
    period = 60.0 / song["bpm"]
    if style == "bass":
        for b, move in enumerate(song["bars"]):
            _render_bass_bar(out, move, b * 4 * period, period, song, sample_rate)
    elif style == "drumline":
        for t, hand, acc in hits(song):
            f = song["hi"] if hand == "R" else song["lo"]
            blip = sounds.tone(f * (1.03 if acc else 1.0), 0.075 if acc else 0.055, sample_rate, harmonics=0.35, attack=0.02, release=0.25)
            _mix(out, blip, t, sample_rate, 0.8 if acc else 0.5)
    else:
        raise ValueError(f"unknown song style '{style}'")
    peak = float(np.max(np.abs(out)))
    if peak > 0.8:
        out *= 0.8 / peak
    return out


def describe(song: dict) -> str:
    return f"{song['name']}: {song['style']}, {song['bpm']} bpm, {' '.join(song['bars'])}"
