"""Procedural droid vocalisations (beeps, boops, warbles) synthesised with numpy.

Everything here is pure numpy so it can be unit tested off-robot. The robot side
only needs ``render_phrase`` to get a float32 mono buffer at the speaker sample
rate and push it through ``reachy_mini.media.push_audio_sample``.

Design notes
- All primitives return float32 arrays in [-1, 1] at ``sample_rate``.
- A "phrase" is a short sequence of primitives chosen by emotion, with random
  variation so the pet never says exactly the same thing twice.
"""

from __future__ import annotations

import math
import random

import numpy as np

SAMPLE_RATE = 16000  # ReSpeaker output rate on the wireless unit (AudioBase.SAMPLE_RATE)

_TWO_PI = 2.0 * math.pi


def _envelope(n: int, attack: float, release: float) -> np.ndarray:
    """Linear attack / release envelope, attack+release expressed as fraction of n."""
    env = np.ones(n, dtype=np.float32)
    a = max(1, int(n * attack))
    r = max(1, int(n * release))
    env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
    env[n - r :] = np.linspace(1.0, 0.0, r, dtype=np.float32)
    return env


def _time(duration: float, sample_rate: int) -> np.ndarray:
    n = max(1, int(duration * sample_rate))
    return np.arange(n, dtype=np.float32) / sample_rate


def silence(duration: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Return ``duration`` seconds of silence."""
    return np.zeros(max(1, int(duration * sample_rate)), dtype=np.float32)


def tone(
    freq: float,
    duration: float,
    sample_rate: int = SAMPLE_RATE,
    harmonics: float = 0.25,
    attack: float = 0.05,
    release: float = 0.2,
) -> np.ndarray:
    """A steady beep with a little square-ish harmonic for a synthy timbre."""
    t = _time(duration, sample_rate)
    wave = np.sin(_TWO_PI * freq * t) + harmonics * np.sin(_TWO_PI * 3.0 * freq * t)
    wave /= 1.0 + harmonics
    return (wave * _envelope(len(t), attack, release)).astype(np.float32)


def chirp(
    f_start: float,
    f_end: float,
    duration: float,
    sample_rate: int = SAMPLE_RATE,
    curve: float = 1.0,
    attack: float = 0.05,
    release: float = 0.25,
) -> np.ndarray:
    """Frequency sweep from f_start to f_end. curve>1 bends toward the end, <1 toward the start."""
    t = _time(duration, sample_rate)
    frac = (t / t[-1]) ** curve if len(t) > 1 else t
    inst_freq = f_start + (f_end - f_start) * frac
    phase = np.cumsum(inst_freq) * (_TWO_PI / sample_rate)
    wave = np.sin(phase) + 0.2 * np.sin(2.0 * phase)
    wave /= 1.2
    return (wave * _envelope(len(t), attack, release)).astype(np.float32)


def warble(
    freq: float,
    duration: float,
    rate: float = 12.0,
    depth: float = 0.08,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Vibrato-modulated tone: the classic 'happy droid' trill."""
    t = _time(duration, sample_rate)
    inst_freq = freq * (1.0 + depth * np.sin(_TWO_PI * rate * t))
    phase = np.cumsum(inst_freq) * (_TWO_PI / sample_rate)
    wave = np.sin(phase) + 0.3 * np.sin(2.0 * phase)
    wave /= 1.3
    return (wave * _envelope(len(t), 0.05, 0.3)).astype(np.float32)


def purr(
    duration: float,
    base: float = 330.0,
    pulse_rate: float = 24.0,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """A rolling "brrr" purr: a mid tone with harmonics, amplitude-modulated by a fast pulse train.

    The carrier sits above 300 Hz on purpose. A real cat purrs at 25-50 Hz with a 50-60 Hz body,
    but the robot's speaker is tiny and reproduces nothing below ~300 Hz, so a low rumble comes out
    as silence. The roughness of the 24 Hz pulses is what reads as a purr; the harmonics carry it.
    """
    t = _time(duration, sample_rate)
    wobble = 1.0 + 0.015 * np.sin(_TWO_PI * 1.7 * t)  # a slow breath in the pitch
    f = base * wobble
    carrier = np.sin(_TWO_PI * f * t) + 0.6 * np.sin(_TWO_PI * 2.0 * f * t) + 0.35 * np.sin(_TWO_PI * 3.0 * f * t)
    pulses = 0.45 + 0.55 * np.clip(np.sin(_TWO_PI * pulse_rate * t), 0.0, None) ** 0.5  # clipped: distinct "rrr" bumps
    wave = carrier * pulses / 1.95
    return (wave * _envelope(len(t), 0.15, 0.3)).astype(np.float32)


def noise_burst(duration: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Short filtered noise 'pfft' for raspberries and static."""
    n = max(1, int(duration * sample_rate))
    rng = np.random.default_rng()
    white = rng.standard_normal(n).astype(np.float32)
    # crude one-pole low pass so it is not harsh on a small speaker
    out = np.empty_like(white)
    acc = 0.0
    for i in range(n):
        acc = 0.85 * acc + 0.15 * white[i]
        out[i] = acc
    out /= max(1e-6, float(np.max(np.abs(out))))
    return (0.6 * out * _envelope(n, 0.02, 0.5)).astype(np.float32)


def concat(*parts: np.ndarray) -> np.ndarray:
    """Join primitives into one buffer."""
    return np.concatenate(parts).astype(np.float32)


# --------------------------------------------------------------------------- phrases

# Emotions the behavior layer can ask for. Keep this the single source of truth.
EMOTIONS = (
    "hello_new",  # first time meeting someone
    "hello_friend",  # recognised someone it has met before
    "hello_bestie",  # recognised a person with lots of history
    "curious",
    "happy",
    "excited",
    "content",  # gentle, when being held / petted
    "purr",
    "giggle",
    "surprised",
    "confused",
    "sad",
    "lonely",
    "sleepy",
    "yawn",
    "wake",
    "dizzy",
    "annoyed",
    "low_battery",
    "name",  # "huh? me?" when it hears its name
    "ticklish",  # belly scratch
    "shy",
    "sneeze",
    "hiccup",
    "sing",  # one blip on the beat
    "tada",  # finished a trick
    "mirror_start",  # "let's play mirror": two-note rising call
    "mirror_end",  # the same call, falling
    "mime_start",  # "do what I do": a three-note fanfare
    "mime_end",  # the fanfare, closed
    "mime_cue",  # "watch this" blip before each shown move
    "yes",  # "you did it": bright double blip
    "no_no",  # nuh-uh-uh: not in the mood (ears)
    "huff",  # "no? like THIS": a short exasperated puff
    "coo",  # a warm, low coo: being hugged
)

# What each sound means, for the lexicon on the Play tab.
MEANINGS = {
    "hello_new": "oh? hi! (a stranger)", "hello_friend": "hi again! (a friend)", "hello_bestie": "YOU! (a bestie)", "content": "mm, being held or petted", "happy": "pleased", "excited": "very pleased (a bestie, a party)",
    "curious": "hm? what's that?", "giggle": "that tickles / that's funny", "surprised": "oh!", "confused": "where did you go?",
    "sad": "aw... / you left", "lonely": "nobody has visited for a while", "sleepy": "running out of energy", "yawn": "about to nod off",
    "wake": "waking up", "dizzy": "you shook me", "annoyed": "stop that / come ON", "low_battery": "battery low",
    "name": "huh? me?", "ticklish": "belly scratch", "shy": "you are staring at me", "sneeze": "achoo", "hiccup": "hic",
    "sing": "singing along to the beat", "tada": "finished a trick", "purr": "being petted, content",
    "mirror_start": "let's play mirror: I'll copy you", "mirror_end": "mirror game over",
    "mime_start": "Simon says: do what I do", "mime_end": "Simon says is over", "mime_cue": "watch this move",
    "yes": "you did it!", "huff": "no? like THIS. again", "no_no": "nuh-uh-uh: leave my ears alone",
    "coo": "aww... a hug",
}


def render_phrase(emotion: str, rng: random.Random | None = None, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Return a float32 mono buffer for ``emotion``. Raises KeyError for an unknown emotion."""
    if emotion not in EMOTIONS:
        raise KeyError(f"Unknown emotion '{emotion}'. Known: {EMOTIONS}")
    r = rng if rng is not None else random.Random()
    j = lambda lo, hi: r.uniform(lo, hi)  # noqa: E731  jitter helper
    sr = sample_rate

    if emotion == "hello_new":
        # rising two-note question, then a quick happy trill: "oh? hi!"
        out = concat(
            chirp(j(500, 650), j(900, 1100), j(0.12, 0.18), sr, curve=0.8),
            silence(0.05, sr),
            warble(j(1100, 1400), j(0.18, 0.28), rate=14, sample_rate=sr),
        )
    elif emotion == "hello_friend":
        out = concat(
            warble(j(900, 1100), j(0.12, 0.18), rate=10, sample_rate=sr),
            silence(0.04, sr),
            chirp(j(1000, 1200), j(1500, 1800), j(0.10, 0.16), sr),
            silence(0.04, sr),
            tone(j(1300, 1600), j(0.08, 0.14), sr),
        )
    elif emotion == "hello_bestie":
        notes = [chirp(j(800, 1000) * k, j(1400, 1700) * k, 0.09, sr) for k in (1.0, 1.2, 1.5)]
        out = concat(
            *sum(([n, silence(0.03, sr)] for n in notes), []),
            warble(j(1800, 2100), j(0.3, 0.45), rate=16, depth=0.1, sample_rate=sr),
        )
    elif emotion == "curious":
        out = concat(
            tone(j(700, 900), j(0.08, 0.12), sr),
            silence(j(0.03, 0.08), sr),
            chirp(j(800, 950), j(1300, 1600), j(0.14, 0.22), sr, curve=1.6),
        )
    elif emotion == "happy":
        out = concat(
            chirp(j(900, 1100), j(1300, 1500), j(0.08, 0.12), sr),
            silence(0.03, sr),
            chirp(j(1100, 1300), j(1600, 1900), j(0.08, 0.12), sr),
            silence(0.03, sr),
            warble(j(1500, 1800), j(0.15, 0.25), sample_rate=sr),
        )
    elif emotion == "excited":
        parts = []
        for _ in range(r.randint(4, 6)):
            parts.append(chirp(j(1200, 1600), j(1900, 2600), j(0.05, 0.08), sr))
            parts.append(silence(j(0.015, 0.03), sr))
        parts.append(warble(j(2000, 2400), j(0.2, 0.3), rate=18, depth=0.12, sample_rate=sr))
        out = concat(*parts)
    elif emotion == "content":
        out = concat(
            warble(j(500, 650), j(0.3, 0.5), rate=6, depth=0.04, sample_rate=sr),
            silence(0.05, sr),
            chirp(j(600, 700), j(450, 550), j(0.2, 0.3), sr, curve=0.7),
        )
    elif emotion == "purr":
        out = purr(j(1.2, 2.2), base=j(300, 370), pulse_rate=j(20, 27), sample_rate=sr)
    elif emotion == "giggle":
        parts = []
        f = j(1000, 1300)
        for i in range(r.randint(4, 7)):
            parts.append(chirp(f * (1 + 0.06 * i), f * (1 + 0.06 * i) * 1.25, 0.045, sr, release=0.5))
            parts.append(silence(0.035, sr))
        out = concat(*parts)
    elif emotion == "surprised":
        out = concat(
            chirp(j(600, 800), j(1900, 2400), j(0.10, 0.15), sr, curve=2.0),
            silence(0.08, sr),
            tone(j(1800, 2200), j(0.12, 0.2), sr, harmonics=0.4),
        )
    elif emotion == "confused":
        out = concat(
            chirp(j(900, 1000), j(700, 800), j(0.12, 0.18), sr),
            silence(0.06, sr),
            chirp(j(700, 800), j(1000, 1200), j(0.12, 0.18), sr),
            silence(0.06, sr),
            warble(j(850, 950), j(0.2, 0.3), rate=5, depth=0.06, sample_rate=sr),
        )
    elif emotion == "sad":
        out = concat(
            chirp(j(800, 900), j(500, 600), j(0.35, 0.5), sr, curve=0.8),
            silence(0.1, sr),
            chirp(j(550, 650), j(300, 380), j(0.4, 0.6), sr, curve=0.8),
        )
    elif emotion == "lonely":
        out = concat(
            chirp(j(600, 700), j(750, 850), j(0.25, 0.35), sr),
            silence(0.3, sr),
            chirp(j(700, 800), j(450, 520), j(0.5, 0.7), sr, curve=0.6),
        )
    elif emotion == "sleepy":
        out = concat(
            warble(j(350, 450), j(0.5, 0.8), rate=3, depth=0.05, sample_rate=sr),
            silence(0.15, sr),
            chirp(j(400, 450), j(250, 300), j(0.5, 0.8), sr, curve=0.5),
        )
    elif emotion == "yawn":
        out = chirp(j(300, 380), j(700, 850), j(0.5, 0.7), sr, curve=1.3, attack=0.3, release=0.5)
        out = concat(out, chirp(j(700, 850), j(250, 300), j(0.6, 0.9), sr, curve=0.6, attack=0.05, release=0.6))
    elif emotion == "wake":
        out = concat(
            chirp(j(300, 400), j(1200, 1500), j(0.3, 0.45), sr, curve=1.8),
            silence(0.05, sr),
            tone(j(1400, 1700), 0.08, sr),
            silence(0.03, sr),
            tone(j(1700, 2000), 0.1, sr),
        )
    elif emotion == "dizzy":
        t = _time(j(1.0, 1.5), sr)
        inst = j(700, 900) * (1.0 + 0.35 * np.sin(_TWO_PI * j(2.0, 3.5) * t) * np.exp(-t))
        phase = np.cumsum(inst) * (_TWO_PI / sr)
        out = (np.sin(phase) * _envelope(len(t), 0.05, 0.4)).astype(np.float32)
    elif emotion == "annoyed":
        out = concat(
            tone(j(300, 380), j(0.12, 0.18), sr, harmonics=0.6),
            silence(0.05, sr),
            tone(j(280, 340), j(0.15, 0.22), sr, harmonics=0.6),
            noise_burst(j(0.15, 0.25), sr),
        )
    elif emotion == "name":
        out = concat(
            chirp(j(700, 850), j(1100, 1300), j(0.07, 0.10), sr, curve=1.5),
            silence(0.05, sr),
            chirp(j(1000, 1200), j(1700, 2100), j(0.16, 0.24), sr, curve=2.2),
        )
    elif emotion == "ticklish":
        parts = []
        f = j(1300, 1700)
        for i in range(r.randint(6, 9)):
            parts.append(warble(f * (1 + 0.05 * (i % 3)), 0.06, rate=30, depth=0.2, sample_rate=sr))
            parts.append(silence(j(0.02, 0.05), sr))
        parts.append(chirp(f * 1.2, f * 0.7, j(0.15, 0.25), sr, curve=0.7))
        out = concat(*parts)
    elif emotion == "shy":
        out = concat(
            chirp(j(900, 1000), j(650, 750), j(0.2, 0.3), sr, curve=0.8),
            silence(0.12, sr),
            tone(j(600, 700), j(0.08, 0.12), sr, harmonics=0.1),
            silence(0.05, sr),
            tone(j(650, 750), j(0.06, 0.1), sr, harmonics=0.1),
        )
    elif emotion == "sneeze":
        # Timed to motion.g_sneeze (10.6 s): 1.2 s of nothing (look down, shake), three rising inhales 1.2 s apart,
        # a rising wind-up squeak, the choo at 5.4 s, a groggy low note during the recovery.
        inhale = lambda f0, f1, d: chirp(f0, f1, d, sr, curve=2.0, attack=0.5, release=0.3)  # noqa: E731
        out = concat(
            silence(1.2, sr),
            inhale(j(500, 560), j(800, 900), 0.4), silence(0.8, sr),
            inhale(j(650, 720), j(1050, 1150), 0.45), silence(0.75, sr),
            inhale(j(850, 950), j(1500, 1700), 0.5), silence(0.7, sr),
            chirp(j(1200, 1400), j(2200, 2600), 0.5, sr, curve=1.5, attack=0.3, release=0.1),  # wind-up
            silence(0.1, sr),
            noise_burst(j(0.18, 0.24), sr),  # choo
            chirp(j(1400, 1800), j(450, 550), j(0.18, 0.24), sr, curve=0.5),
            silence(0.9, sr),
            tone(j(380, 440), j(0.35, 0.45), sr, harmonics=0.15, attack=0.3, release=0.5),  # ugh
        )
    elif emotion in ("mirror_start", "mirror_end"):
        up = emotion == "mirror_start"
        a, b = (j(700, 760), j(1050, 1150)) if up else (j(1050, 1150), j(700, 760))
        out = concat(tone(a, 0.14, sr, harmonics=0.2), silence(0.04, sr), tone(b, 0.22, sr, harmonics=0.2))
    elif emotion in ("mime_start", "mime_end"):
        up = emotion == "mime_start"
        fs = (j(600, 640), j(800, 840), j(1000, 1060)) if up else (j(1000, 1060), j(800, 840), j(600, 640))
        notes = [tone(f, 0.11, sr, harmonics=0.3) for f in fs]
        out = concat(notes[0], silence(0.03, sr), notes[1], silence(0.03, sr), notes[2], silence(0.05, sr), warble(fs[2], 0.25, rate=10, depth=0.05, sample_rate=sr))
    elif emotion == "mime_cue":
        out = concat(tone(j(1300, 1400), 0.06, sr, harmonics=0.4), silence(0.05, sr), tone(j(1300, 1400), 0.06, sr, harmonics=0.4))
    elif emotion == "yes":
        out = concat(chirp(j(900, 1000), j(1400, 1500), 0.09, sr), silence(0.04, sr), chirp(j(1300, 1400), j(1900, 2000), 0.12, sr))
    elif emotion == "no_no":
        # three short falling "nuh"s, each a step lower, a small wobble on the last: a cute telling-off
        f0 = j(760, 840)
        out = concat(
            chirp(f0, f0 * 0.86, 0.09, sr, release=0.4), silence(0.06, sr),
            chirp(f0 * 0.92, f0 * 0.79, 0.09, sr, release=0.4), silence(0.06, sr),
            warble(f0 * 0.8, 0.16, rate=14, depth=0.06, sample_rate=sr),
        )
    elif emotion == "huff":
        out = concat(tone(j(420, 480), 0.08, sr, harmonics=0.5, attack=0.02, release=0.3), noise_burst(j(0.22, 0.28), sr))
    elif emotion == "hiccup":
        out = concat(tone(j(500, 600), 0.03, sr, attack=0.02, release=0.3), chirp(j(900, 1100), j(1500, 1900), 0.06, sr, curve=1.8))
    elif emotion == "coo":
        # a warm "ooo-oo": two slow, low, gently sliding notes with a soft wobble, then a purr that trails off
        f0 = j(380, 440)
        out = concat(
            chirp(f0, f0 * 1.12, j(0.5, 0.7), sr, curve=0.8, attack=0.2, release=0.4),
            silence(0.05, sr),
            warble(f0 * 0.95, j(0.6, 0.8), rate=5, depth=0.03, sample_rate=sr),
            purr(j(0.8, 1.1), base=f0 * 0.9, pulse_rate=j(18, 22), sample_rate=sr) * np.float32(0.6),
        )
    elif emotion == "sing":
        out = tone(j(1100, 1700), j(0.06, 0.09), sr, harmonics=0.3)
    elif emotion == "tada":
        notes = [chirp(f, f * 1.15, 0.08, sr) for f in (j(900, 1000), j(1150, 1250), j(1400, 1500))]
        out = concat(*sum(([n, silence(0.03, sr)] for n in notes), []), warble(j(1900, 2100), j(0.3, 0.4), rate=12, depth=0.08, sample_rate=sr))
    elif emotion == "low_battery":
        out = concat(
            tone(1000, 0.08, sr), silence(0.08, sr),
            tone(800, 0.08, sr), silence(0.08, sr),
            chirp(700, 350, 0.4, sr, curve=0.7),
        )
    else:  # pragma: no cover - guarded by the membership check above
        raise KeyError(emotion)

    peak = float(np.max(np.abs(out)))
    if peak > 1.0:
        out = out / peak
    return (0.8 * out).astype(np.float32)


def phrase_duration(buffer: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    """Seconds of audio in ``buffer``."""
    return len(buffer) / sample_rate
