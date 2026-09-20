"""Songs it makes up. Two styles, both built from the same little dict so they can be saved as JSON.

``drumline`` is rhythm, not melody: two "hands" on two pitches (a high and a low blip), and the
interest is all in the playing — quarters, eighths, triplets, paradiddles, flams, rolls, rests.

``bass`` is the robot's idea of bass music, built the way the genre actually is: 140 bpm, a sparse
halftime kit underneath (kick on the 1, snare on the 3, hats on the offbeats, a two-bar loop) so
there is always something to count against, and a wobbling bass note on top. The arrangement is
fixed rather than random, because that is what makes it followable: two bars of drums to find the
beat, two bars of the A bass, a build bar whose snare roll speeds up into a silent gap, the drop on
the downbeat of bar 5, then B and C sections, a fill, and a last hit left to ring.

The speaker reproduces nothing under ~300 Hz, so the bass is implied by a harmonic-rich note up
where the speaker works, with the filter doing the talking, and the kick leans on its click.

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

# ----------------------------------------------------------------- bass music
# 16 steps to the bar (sixteenths at 140), two bars to a drum loop. "x" is a hit, "." is a rest.
# Sparse on purpose: the whole point of halftime is the room it leaves for the bass.
STEPS_PER_BAR = 16
BASS_BPM = 140  # the genre tempo; the body bobs at half of it (see body_bpm), which is the halftime feel
DRUM_KITS: dict[str, dict[str, str]] = {
    "classic": {  # kick on the 1, snare on the 3, offbeat hats
        "kick":  "x..............." "x.......x.......",
        "snare": "........x......." "........x.......",
        "hat":   "..x...x...x...x." "..x...x...x...x.",
    },
    "rolling": {  # a ghost kick before the snare and a busier second bar
        "kick":  "x..........x...." "x......x...x....",
        "snare": "........x......." "........x.....x.",
        "hat":   "..x...x...x.x.x." "..x...x...x.x.x.",
    },
    "sparse": {  # almost nothing but the 1 and the 3: maximum room
        "kick":  "x..............." "x...............",
        "snare": "........x......." "........x.......",
        "hat":   "....x.......x..." "....x.......x...",
    },
}
# Bass voices: how the note wobbles. The number is wobbles per beat, so they are all locked to the grid.
BASS_VOICES = ("wub1", "wub2", "wub3", "wahs", "pews", "stabs")
# The arrangement, one move per bar. Voices A, B and C are filled in by compose.
# Four-bar phrases throughout, so the drop lands on a downbeat you can feel coming: two bars of kit to
# find the beat, a bar of A, a build; then the drop, two of B, a fill; then two of C, a last B, and out.
BASS_ARRANGEMENT = ("intro", "intro", "A", "build", "drop", "B", "B", "fill", "C", "C", "B", "out")
BASS_STRUCTURE = ("intro", "build", "drop", "fill", "out")
BASS_MOVES = BASS_VOICES + BASS_STRUCTURE
# Where the bass note sits, per bar, as semitones off the root: it moves, so a phrase has somewhere to go.
BASS_DEGREES = (0, 0, 0, 0, 0, 0, 0, 0, 3, 3, -2, 0)
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
    """Halftime at 140 with a fixed arrangement: what varies is the kit, the three bass voices and the pitch."""
    voices = rng.sample(BASS_VOICES, 3)  # A, B, C: three different ones, so the sections are telling apart
    parts = dict(zip("ABC", voices))
    return {
        "style": "bass",
        "bpm": BASS_BPM,
        "bars": [parts.get(m, m) for m in BASS_ARRANGEMENT],
        "kit": rng.choice(sorted(DRUM_KITS)),
        "hi": rng.choice([1800, 2100, 2400]),  # the lasers
        "lo": rng.choice([310, 330, 350, 370]),  # the "bass": as low as the speaker can still carry
        "name": rng.choice(["wub study", "the drop", "bass face", "wa-wah", "pew pew pew", "filthy (politely)", "sub sandwich"]) + f" #{rng.randint(10, 99)}",
    }


def body_bpm(song: dict) -> float:
    """What the robot should bob to. Bass music is counted in halftime: the body moves on the 1 and the 3,
    once per two of the song's beats, which is also as fast as the neck wants to move."""
    return song["bpm"] / 2.0 if song["style"] == "bass" else float(song["bpm"])


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


def _render_drums(out: np.ndarray, song: dict, bar: int, t0: float, step: float, sample_rate: int, rng: random.Random) -> None:
    """The kit for one bar: the two-bar loop, or a roll for a build or a fill.

    This is the part that was missing. Wobbles on their own have nothing to be early or late against;
    with a kick on the 1 and a snare on the 3 underneath, the same wobble suddenly has a place.
    """
    move = song["bars"][bar]
    bar_s = STEPS_PER_BAR * step
    if move == "out":
        sounds_kick = sounds.kick(sample_rate, rng)
        _mix(out, sounds_kick, t0, sample_rate, 1.0)
        _mix(out, sounds.snare(sample_rate, rng, dur=0.5), t0, sample_rate, 0.8)  # last hit, left to ring
        return
    if move in ("build", "fill"):
        # a snare roll that doubles up as it goes, then stops dead for the last quarter: the gap before the
        # drop. The silence is the point — it is what makes the downbeat that follows land.
        n = 8 if move == "fill" else 16
        for k in range(n):
            u = k / n
            if u >= 0.75:
                break  # the gap
            t = t0 + bar_s * u
            _mix(out, sounds.snare(sample_rate, rng, dur=0.06 + 0.03 * u), t, sample_rate, 0.35 + 0.5 * u)
        _mix(out, sounds.kick(sample_rate, rng), t0, sample_rate, 1.0)
        return
    kit = DRUM_KITS[song["kit"]]
    off = (bar % 2) * STEPS_PER_BAR  # the loop is two bars long
    for k in range(STEPS_PER_BAR):
        t = t0 + k * step
        if kit["kick"][off + k] == "x":
            _mix(out, sounds.kick(sample_rate, rng), t, sample_rate, 1.0)
        if kit["snare"][off + k] == "x":
            _mix(out, sounds.snare(sample_rate, rng), t, sample_rate, 1.0)
        if kit["hat"][off + k] == "x":
            _mix(out, sounds.hat(sample_rate, rng, open_=(k == 12 and move == "drop")), t, sample_rate, 0.45)
    if move == "drop":  # the downbeat of the drop gets an open hat over it, like a crash
        _mix(out, sounds.hat(sample_rate, rng, open_=True), t0, sample_rate, 0.5)


def _render_bass(out: np.ndarray, song: dict, bar: int, t0: float, step: float, sample_rate: int) -> None:
    """The bass voice for one bar: one long note (or a few short ones) with the filter moving on the grid."""
    move = song["bars"][bar]
    beat = 4 * step  # a beat is four sixteenths
    bar_s = STEPS_PER_BAR * step
    root = float(song["lo"]) * 2 ** (BASS_DEGREES[bar % len(BASS_DEGREES)] / 12.0)
    hi = float(song["hi"])
    per_beat = 60.0 / song["bpm"]  # seconds per beat: a wobble rate of n per beat is n / per_beat Hz
    if move == "intro":
        return  # just the drums: two bars to find the beat
    if move == "out":
        _mix(out, sounds.wub(bar_s * 0.9, root, rate=0.5 / per_beat, sample_rate=sample_rate), t0, sample_rate, 0.7)
        return
    if move == "build":  # the riser, ending in the gap
        _mix(out, sounds.chirp(root, hi, bar_s * 0.86, sample_rate, curve=2.0, attack=0.15, release=0.04), t0, sample_rate, 0.45)
        return
    if move == "drop":  # the big one: lands on the downbeat, held for the bar, wobbling once a beat
        _mix(out, sounds.wub(bar_s * 0.98, root * 0.92, rate=1.0 / per_beat, sample_rate=sample_rate, depth=1.3), t0, sample_rate, 0.8)
        return
    if move == "fill":
        for k in range(4):
            _mix(out, sounds.zap(0.1, hi, root * 1.4, sample_rate), t0 + k * beat * 0.75, sample_rate, 0.5)
        return
    if move.startswith("wub"):  # wub1 / wub2 / wub3: wobbles per beat, so it is always locked to the kit
        rate = int(move[3]) / per_beat
        for k in range(2):  # two half-bar notes: the second one is where you hear the wobble land on the snare
            _mix(out, sounds.wub(bar_s / 2 * 0.96, root, rate=rate, sample_rate=sample_rate), t0 + k * bar_s / 2, sample_rate, 0.62)
        return
    if move == "wahs":  # wah on the 1, wa-wah into the 3
        _mix(out, sounds.wah(beat * 0.9, root * 1.2, sample_rate, up=True), t0, sample_rate, 0.6)
        for k in (1.5, 2.0, 3.0):
            _mix(out, sounds.wah(beat * 0.45, root * (1.2 if k == 2.0 else 1.0), sample_rate, swings=2.0), t0 + k * beat, sample_rate, 0.55)
        return
    if move == "pews":  # lasers over the beat, landing on the offbeats so the snare still shows through
        for k in (0.0, 0.75, 1.5, 2.5, 3.25):
            _mix(out, sounds.zap(0.11, hi, root * 1.3, sample_rate), t0 + k * beat, sample_rate, 0.45)
        return
    if move == "stabs":  # short wub stabs, off the beat
        for k in (0.0, 0.75, 1.5, 2.0, 2.75, 3.5):
            _mix(out, sounds.wub(beat * 0.4, root, rate=0.5 / per_beat, sample_rate=sample_rate), t0 + k * beat, sample_rate, 0.6)
        return
    raise KeyError(f"unknown bass move '{move}'")


def render(song: dict, sample_rate: int = sounds.SAMPLE_RATE) -> np.ndarray:
    """Audio for a song. Drumline: a short blip per hit, accents louder and a touch longer.
    Bass: the kit on the grid, and a bass voice over it, bar by bar."""
    style = song["style"]
    n = int(duration(song) * sample_rate)
    out = np.zeros(n, dtype=np.float32)
    period = 60.0 / song["bpm"]
    if style == "bass":
        rng = random.Random(song["name"])  # the kit's own jitter, the same every time a saved song plays
        step = period / 4.0
        for b in range(len(song["bars"])):
            t0 = b * 4 * period
            _render_drums(out, song, b, t0, step, sample_rate, rng)
            _render_bass(out, song, b, t0, step, sample_rate)
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
