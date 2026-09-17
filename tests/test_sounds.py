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
