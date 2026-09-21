"""Songs it makes up, as a little dict so they can be saved as JSON.

``bass`` is the robot's idea of bass music, built the way the genre actually is: 140 bpm, a sparse
halftime kit underneath (kick on the 1, snare on the 3, hats on the offbeats, a two-bar loop) so
there is always something to count against, and a wobbling bass note on top. Arrangements are built
from four-bar phrases by a small grammar, so the shape changes from song to song while staying
followable: it always counts you in, every drop has a build in front of it and lands on a phrase
line, and it always has an ending.

The speaker reproduces nothing under ~300 Hz, so the bass is implied by a harmonic-rich note up
where the speaker works, with the filter doing the talking, and the kick leans on its click.

``compose`` makes one; ``render`` turns one into audio.
"""

from __future__ import annotations

import random

import numpy as np

from festival_pet import sounds

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
BASS_STRUCTURE = ("intro", "build", "drop", "fill", "break", "out")
BASS_MOVES = BASS_VOICES + BASS_STRUCTURE

# Four-bar phrases. The grammar is fixed so a song is always followable — every drop has a build in
# front of it and lands on a phrase line, every song counts you in and has an ending — but which
# phrases it strings together, how many, and where the bass sits are all made up per song. Four days
# of festival is a lot of songs to sit through, so no two should have the same shape.
#   A, B, C  the song's three bass voices      V  any of them, picked per bar
PHRASES: dict[str, tuple[str, ...]] = {
    "count_in": ("intro", "intro", "A", "build"),  # kit alone, then a taste of A, then the build
    "count_in_long": ("intro", "A", "A", "build"),
    "drop": ("drop", "B", "B", "fill"),  # the pay-off phrase
    "drop_long": ("drop", "drop", "B", "fill"),  # two bars of the big one
    "ride": ("V", "V", "V", "fill"),  # a verse of its own
    "ride_easy": ("A", "A", "C", "C"),  # no fill: it just rolls on
    "breakdown": ("break", "break", "C", "build"),  # the floor drops out, then back up into another drop
    "half_break": ("break", "C", "C", "build"),
    "outro": ("B", "B", "C", "out"),
    "outro_quiet": ("break", "C", "A", "out"),
}
MAX_PHRASES = 6  # 24 bars, about 41 s: long enough to have a shape, short enough to watch
MIDDLES = ("ride", "ride_easy", "ride", "ride_easy", "breakdown", "half_break")  # what can sit between the drops
# Where the bass sits per phrase, in semitones off the root: minor-ish, and it moves.
BASS_RIFFS = ((0, 0, 0, 0), (0, 0, 3, 3), (0, 3, 0, -2), (0, 0, -2, -4), (0, 5, 3, 0), (0, -4, 0, 3), (3, 3, 0, 0))
STYLES = ("bass",)  # the drumline beeps are gone: next to the bass songs they were just annoying


def compose(rng: random.Random, style: str | None = None) -> dict:
    """A new song. ``style`` is "bass" or None (the same thing, while bass is the only style)."""
    style = style if style is not None else rng.choice(STYLES)
    if style not in STYLES:
        raise ValueError(f"unknown song style '{style}'")
    return _compose_bass(rng)


def _bass_arrangement(rng: random.Random) -> tuple[list[str], list[int]]:
    """String four-bar phrases into a song, and pick where the bass sits in each. Returns (bars, degrees).

    Always: a count-in phrase, a drop phrase, an outro. In between, one to three middles, and after a
    breakdown there is always another drop (that is what a breakdown is for). 12 to 24 bars, so some
    songs are a quick 20 seconds and others run a minute-ish with two drops.
    """
    plan = [rng.choice(("count_in", "count_in_long")), rng.choice(("drop", "drop", "drop_long"))]
    for _ in range(rng.choices((0, 1, 2, 3), weights=(2, 5, 3, 2))[0]):
        if len(plan) >= MAX_PHRASES - 1:
            break  # room for the outro: nothing here runs past about 40 seconds
        middle = rng.choice(MIDDLES)
        plan.append(middle)
        if middle in ("breakdown", "half_break") and len(plan) < MAX_PHRASES - 1:
            plan.append(rng.choice(("drop", "drop_long")))  # a breakdown builds, so it has to land on something
    plan.append(rng.choice(("outro", "outro_quiet")))

    voices = rng.sample(BASS_VOICES, 3)
    parts = dict(zip("ABC", voices))
    bars: list[str] = []
    degrees: list[int] = []
    for name in plan:
        riff = rng.choice(BASS_RIFFS)
        for k, move in enumerate(PHRASES[name]):
            bars.append(parts[move] if move in parts else rng.choice(voices) if move == "V" else move)
            degrees.append(riff[k])
    return bars, degrees


def _compose_bass(rng: random.Random) -> dict:
    """Halftime at 140. The kit, the three bass voices, the arrangement and the riff are all made up."""
    bars, degrees = _bass_arrangement(rng)
    adjective = rng.choice(["wub", "filthy", "heavy", "sub", "wonky", "grimy", "polite", "tiny"])
    noun = rng.choice(["study", "business", "situation", "o'clock", "sandwich", "weather", "energy", "face"])
    return {
        "style": "bass",
        "bpm": BASS_BPM,
        "bars": bars,
        "degrees": degrees,
        "kit": rng.choice(sorted(DRUM_KITS)),
        "hi": rng.choice([1800, 2100, 2400]),  # the lasers
        "lo": rng.choice([310, 330, 350, 370]),  # the "bass": as low as the speaker can still carry
        "name": f"{adjective} {noun} #{rng.randint(10, 99)}",
    }


def body_bpm(song: dict) -> float:
    """What the robot should bob to. Bass music is counted in halftime: the body moves on the 1 and the 3,
    once per two of the song's beats, which is also as fast as the neck wants to move."""
    return song["bpm"] / 2.0


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
    if move == "break":  # the floor drops out: hats only, so the bar still has a pulse to hold on to
        for k in range(0, STEPS_PER_BAR, 4):
            _mix(out, sounds.hat(sample_rate, rng, open_=(k == 8)), t0 + k * step, sample_rate, 0.4)
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
    root = float(song["lo"]) * 2 ** (song["degrees"][bar] / 12.0)
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
    if move == "break":  # one long note with the filter creeping open: the calm before it comes back
        _mix(out, sounds.wah(bar_s * 0.95, root * 1.15, sample_rate, up=True), t0, sample_rate, 0.45)
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
    """Audio for a song: the kit on the grid, and a bass voice over it, bar by bar."""
    if song["style"] not in STYLES:
        raise ValueError(f"unknown song style '{song['style']}'")
    n = int(duration(song) * sample_rate)
    out = np.zeros(n, dtype=np.float32)
    period = 60.0 / song["bpm"]
    rng = random.Random(song["name"])  # the kit's own jitter, the same every time a saved song plays
    step = period / 4.0
    for b in range(len(song["bars"])):
        t0 = b * 4 * period
        _render_drums(out, song, b, t0, step, sample_rate, rng)
        _render_bass(out, song, b, t0, step, sample_rate)
    peak = float(np.max(np.abs(out)))
    if peak > 0.8:
        out *= 0.8 / peak
    return out


def describe(song: dict) -> str:
    return f"{song['name']}: {song['style']}, {song['bpm']} bpm, {' '.join(song['bars'])}"
