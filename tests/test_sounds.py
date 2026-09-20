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
    seen, styles = set(), set()
    for _ in range(14):
        song = songs.compose(rng)
        seen.add(tuple(song["bars"]))
        styles.add(song["style"])
        buf = songs.render(song)
        assert buf.dtype == np.float32 and np.max(np.abs(buf)) <= 0.8001 and np.isfinite(buf).all()
        assert abs(len(buf) / 16000 - songs.duration(song)) < 0.01
        assert 5.0 < songs.duration(song) < 40.0
        assert song["name"] in songs.describe(song) and song["style"] in songs.describe(song)
        if song["style"] == "drumline":
            assert song["bars"][-1] == "roll_and_stop" and 4 <= len(song["bars"]) <= 9
            assert len(songs.hits(song)) >= 4 * len(song["bars"]) - 4
        else:
            assert song["bars"].count("drop") == 1 and song["bars"].index("riser") == song["bars"].index("drop") - 1
            assert song["lo"] >= 300  # the speaker carries nothing lower: the bass is implied, not played
            assert all(m in songs.BASS_MOVES for m in song["bars"])
    assert len(seen) > 3 and styles == {"drumline", "bass"}  # variety
    # a paradiddle bar has 16 hits with accents on each group of four
    assert [a for _, _, a in songs.PATTERNS["paradiddle"]] == [1, 0, 0, 0] * 4
    # an unknown style is an error, not a quiet fallback
    for bad in (lambda: songs.compose(rng, "polka"), lambda: songs.render({**songs.compose(rng, "bass"), "style": "polka"})):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_jingles_are_short_made_up_melodies_on_a_scale():
    from festival_pet.sounds import JINGLE_BASES, JINGLE_SCALE, jingle_notes, phrase_duration, render_phrase

    pitches, lengths = set(), set()
    for seed in range(12):
        notes = jingle_notes(random.Random(seed))
        assert 3 <= len(notes) <= 14
        starts = [t for t, _, _ in notes]
        assert starts == sorted(starts) and starts[0] == 0.0
        lengths.add(len(notes))
        for _, f, d in notes:
            assert 0.03 < d < 0.2
            # every note is a scale degree of one of the bases: a tune, not a random beep
            assert any(abs(f - base * 2 ** (s / 12.0)) < 0.5 for base in JINGLE_BASES for s in JINGLE_SCALE)
            pitches.add(round(f))
        buf = render_phrase("jingle", random.Random(seed))
        assert 0.3 < phrase_duration(buf) < 2.6
    assert len(pitches) > 8 and len(lengths) > 2  # it does make them up, it is not one tune on repeat


def test_huff_has_a_real_puff():
    buf = sounds.render_phrase("huff", random.Random(1))
    assert sounds.phrase_duration(buf) > 0.28
