import random

import numpy as np

from festival_pet import sounds


def test_every_emotion_renders_clean_float32():
    rng = random.Random(1)
    for e in sounds.EMOTIONS:
        buf = sounds.render_phrase(e, rng=rng)
        assert buf.dtype == np.float32
        assert buf.ndim == 1
        assert 0.2 < sounds.phrase_duration(buf) < 3.0, e
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
