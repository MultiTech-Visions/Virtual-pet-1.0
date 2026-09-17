import random

import numpy as np

from festival_pet import sounds


def test_every_emotion_renders_clean_float32():
    rng = random.Random(1)
    for e in sounds.EMOTIONS:
        buf = sounds.render_phrase(e, rng=rng)
        assert buf.dtype == np.float32
        assert buf.ndim == 1
        assert 0.04 < sounds.phrase_duration(buf) < (9.5 if e == "sneeze" else 3.0), e  # the sneeze is a whole bit
        assert np.max(np.abs(buf)) <= 0.8001, e
        assert np.isfinite(buf).all(), e


def test_unknown_emotion_raises():
    try:
        sounds.render_phrase("banana")
    except KeyError:
        return
    raise AssertionError("expected KeyError")


def test_phrases_vary_between_calls():
    a = sounds.render_phrase("happy", rng=random.Random(1))
    b = sounds.render_phrase("happy", rng=random.Random(2))
    assert len(a) != len(b) or not np.allclose(a, b)


def test_sneeze_sound_is_timed_to_the_gesture():
    from festival_pet.motion import SNEEZE_S
    from festival_pet.sounds import phrase_duration, render_phrase

    buf = render_phrase("sneeze", random.Random(1))
    d = phrase_duration(buf)
    assert 6.5 < d < 9.5 and d < SNEEZE_S  # ends during the slow recovery, before the clearing shake
    sr = 16000
    assert abs(buf[: int(1.15 * sr)]).max() < 1e-6  # nothing while it looks down and shakes
    assert abs(buf[int(5.45 * sr) : int(5.65 * sr)]).max() > 0.2  # the choo lands at 5.4-5.7 s


def test_every_sound_has_a_meaning():
    assert set(sounds.MEANINGS) == set(sounds.EMOTIONS)


def test_every_sound_is_audible_on_a_tiny_speaker():
    """The robot's speaker reproduces nothing below ~300 Hz: a sound living down there plays as silence (the old purr)."""
    import numpy as np

    for emotion in sounds.EMOTIONS:
        buf = sounds.render_phrase(emotion)
        spectrum = np.abs(np.fft.rfft(buf))
        freqs = np.fft.rfftfreq(len(buf), 1 / sounds.SAMPLE_RATE)
        low = float(spectrum[freqs < 280].sum() / spectrum.sum())
        assert low < 0.3, f"{emotion}: {low:.0%} of its energy is below 280 Hz"
        assert np.sqrt(np.mean(buf**2)) > 0.15, f"{emotion} is too quiet"


def test_songs_compose_render_and_describe():
    from festival_pet import songs

    rng = random.Random(7)
    seen = set()
    for _ in range(10):
        song = songs.compose(rng)
        seen.add(tuple(song["bars"]))
        assert song["bars"][-1] == "roll_and_stop" and 4 <= len(song["bars"]) <= 9
        buf = songs.render(song)
        assert buf.dtype == np.float32 and np.max(np.abs(buf)) <= 0.8001 and np.isfinite(buf).all()
        assert abs(len(buf) / 16000 - songs.duration(song)) < 0.01
        assert len(songs.hits(song)) >= 4 * len(song["bars"]) - 4
        assert song["name"] in songs.describe(song)
    assert len(seen) > 3  # variety
    # a paradiddle bar has 16 hits with accents on each group of four
    assert [a for _, _, a in songs.PATTERNS["paradiddle"]] == [1, 0, 0, 0] * 4


def test_huff_has_a_real_puff():
    buf = sounds.render_phrase("huff", random.Random(1))
    assert sounds.phrase_duration(buf) > 0.28
