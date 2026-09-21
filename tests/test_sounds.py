import random

import numpy as np

from festival_pet import sounds


def test_every_emotion_renders_clean_float32():
    rng = random.Random(1)
    for e in sounds.EMOTIONS:
        buf = sounds.render_phrase(e, rng=rng)
        assert buf.dtype == np.float32
        assert buf.ndim == 1
        # the sneeze is a whole bit, and the jingle is a whole little song with a count-in
        longest = {"sneeze": 9.5, "jingle": 24.0}.get(e, 3.0)
        assert 0.04 < sounds.phrase_duration(buf) < longest, e
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
        assert 5.0 < songs.duration(song) < 45.0
        assert song["name"] in songs.describe(song) and song["style"] in songs.describe(song)
        if True:
            assert song["bpm"] == songs.BASS_BPM and songs.body_bpm(song) == songs.BASS_BPM / 2  # halftime for the body
            assert 1 <= song["bars"].count("drop") <= 4  # at least one, and a breakdown earns another
            assert song["bars"][0] == "intro" and song["bars"][-1] == "out"  # a count-in, and an ending
            assert len(song["degrees"]) == len(song["bars"])
            for k, move in enumerate(song["bars"]):
                if move == "drop":
                    assert k % 4 == 0 or song["bars"][k - 1] == "drop"  # drops land on a phrase line
                    assert "build" in song["bars"][max(0, k - 4):k] or song["bars"][k - 1] == "drop"  # ...with a build in front
            assert song["lo"] >= 300  # the speaker carries nothing lower: the bass is implied, not played
            assert all(m in songs.BASS_MOVES for m in song["bars"]) and song["kit"] in songs.DRUM_KITS
    assert len(seen) > 3 and styles == {"bass"}  # the drumline beeps are gone; the bass songs vary on their own
    # four days of festival: the shape has to change, not just the notes
    shapes = {tuple(songs.compose(random.Random(k), "bass")["bars"]) for k in range(40)}
    assert len(shapes) == 40 and len({len(x) for x in shapes}) >= 3
    # an unknown style is an error, not a quiet fallback
    for bad in (lambda: songs.compose(rng, "polka"), lambda: songs.render({**songs.compose(rng, "bass"), "style": "polka"})):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_bass_songs_have_a_kit_to_count_against():
    """The wobble needs a backbeat or there is nothing to hear it against: kick on the 1, snare on the 3."""
    import numpy as np

    from festival_pet import songs

    sr = 16000
    for kit_name, kit in songs.DRUM_KITS.items():
        for voice, pattern in kit.items():
            assert len(pattern) == 2 * songs.STEPS_PER_BAR and set(pattern) <= {"x", "."}, (kit_name, voice)
        assert kit["kick"][0] == "x" and kit["snare"][8] == "x", kit_name  # the 1 and the 3, in both bars
        assert kit["kick"][songs.STEPS_PER_BAR] == "x" and kit["snare"][songs.STEPS_PER_BAR + 8] == "x", kit_name
        assert kit["snare"].count("x") <= 4, kit_name  # sparse: halftime is mostly room

    song = songs.compose(random.Random(1), "bass")
    period = 60.0 / song["bpm"]
    drums = np.zeros(int(songs.duration(song) * sr), dtype=np.float32)
    rng = random.Random(song["name"])
    for b in range(len(song["bars"])):
        songs._render_drums(drums, song, b, b * 4 * period, period / 4, sr, rng)
    peak_at = lambda t: float(np.abs(drums[int(t * sr):int((t + 0.1) * sr)]).max())  # noqa: E731
    for b, move in enumerate(song["bars"]):
        t0 = b * 4 * period
        assert peak_at(t0) > 0.3, f"bar {b} ({move}) has no downbeat"
        if move not in ("build", "fill", "out"):
            assert peak_at(t0 + 2 * period) > 0.3, f"bar {b} ({move}) has no backbeat on the 3"
    # the build ends in a gap: the last eighth of the bar before the drop is silent
    drop = song["bars"].index("drop")
    gap = drop * 4 * period - period * 0.6  # the last three sixteenths before the drop
    assert float(np.abs(drums[int(gap * sr):int(drop * 4 * period * sr)]).max()) < 0.05
    # and the drums are actually audible against the bass, not buried under it
    full = songs.render(song)
    assert float(np.sqrt((drums**2).mean())) > 0.3 * float(np.sqrt((full**2).mean()))


def test_jingles_are_counted_in_songs_with_a_shape_you_can_follow():
    from festival_pet.sounds import (JINGLE_BASES, JINGLE_COUNT_IN, JINGLE_RHYTHMS, JINGLE_SCALE,
                                     jingle_notes, phrase_duration, render_jingle, render_phrase)

    pitches, bars, tempos = set(), set(), set()
    for seed in range(20):
        notes, bpm = jingle_notes(random.Random(seed))
        beat = 60.0 / bpm
        tempos.add(bpm)
        starts = [t for t, _, _, _ in notes]
        assert starts == sorted(starts) and starts[0] == 0.0
        # four taps, on the beat, before a note of the tune: that is what you count yourself in on
        ticks = [n for n in notes if n[3] == "tick"]
        tune = [n for n in notes if n[3] == "note"]
        assert len(ticks) == JINGLE_COUNT_IN and all(n[3] == "tick" for n in notes[:JINGLE_COUNT_IN])
        for k, (t, _, _, _) in enumerate(ticks):
            assert abs(t - k * beat) < 1e-9
        assert abs(tune[0][0] - JINGLE_COUNT_IN * beat) < 1e-9  # the tune starts on the next "1"
        # a whole number of 4/4 bars of tune, four or eight of them: a song, not a bar of beeps
        span = starts[-1] + notes[-1][2] - tune[0][0]
        n_bars = int((max(starts) - tune[0][0]) / (4 * beat) + 1e-6) + 1
        assert n_bars in (4, 8) and 3.0 * beat <= span <= n_bars * 4 * beat
        bars.add(n_bars)
        # every bar plays the same rhythm — one motif, moved around — which is what makes it hummable
        per_bar = [round((t - tune[0][0]) / (beat / 2)) % 8 for t, _, _, _ in tune]
        assert set(per_bar) in [{i for i, hit in enumerate(rh) if hit} for rh in JINGLE_RHYTHMS]
        assert len(set(per_bar)) == len(tune) // n_bars
        for _, f, d, _ in tune:
            assert 0.05 < d < 2.0
            # every note is a scale degree of one of the bases: a tune, not a random beep
            assert any(abs(f - base * 2 ** (s / 12.0)) < 0.5 for base in JINGLE_BASES for s in JINGLE_SCALE)
            pitches.add(round(f))
        buf, bpm2 = render_jingle(random.Random(seed))
        assert bpm2 == bpm and 4.0 < phrase_duration(buf) < 24.0
        assert phrase_duration(render_phrase("jingle", random.Random(seed))) == phrase_duration(buf)
    assert len(pitches) > 8 and bars == {4, 8} and len(tempos) > 2  # it does make them up, it is not one tune on repeat


def test_huff_has_a_real_puff():
    buf = sounds.render_phrase("huff", random.Random(1))
    assert sounds.phrase_duration(buf) > 0.28
