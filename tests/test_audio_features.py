import numpy as np

from festival_pet.audio_features import HOP, SAMPLE_RATE, BeatTracker, ScratchDetector


def _click_track(bpm, seconds, sr=SAMPLE_RATE, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    x = rng.standard_normal(n).astype(np.float32) * noise
    period = int(sr * 60.0 / bpm)
    t = np.arange(int(0.05 * sr)) / sr
    click = (np.sin(2 * np.pi * 180 * t) * np.exp(-t * 60)).astype(np.float32)  # kick-ish thump
    for start in range(0, n - len(click), period):
        x[start : start + len(click)] += click
    # a sustained pad so it is not just impulses
    tt = np.arange(n) / sr
    x += 0.05 * np.sin(2 * np.pi * 220 * tt).astype(np.float32)
    return x


def _feed(tracker, x, chunk=1600):
    now = 0.0
    for i in range(0, len(x), chunk):
        tracker.push(x[i : i + chunk], now)
        now += chunk / SAMPLE_RATE
    return now


def test_beat_tracker_finds_tempo_and_phase():
    for bpm in (90.0, 124.0):
        bt = BeatTracker()
        x = _click_track(bpm, 12.0)
        end = _feed(bt, x)
        assert bt.music, bt.state
        assert abs(bt.state.bpm - bpm) / bpm < 0.04, bt.state
        # Phase should be near 0 right after a click. Next click time:
        period = 60.0 / bpm
        next_click = np.ceil(end / period) * period
        ph = bt.phase(next_click + 0.01)
        assert min(ph, 1 - ph) < 0.15, (bpm, ph)


def test_beat_tracker_rejects_noise():
    bt = BeatTracker()
    x = np.random.default_rng(1).standard_normal(SAMPLE_RATE * 8).astype(np.float32) * 0.05
    _feed(bt, x)
    assert not bt.music, bt.state


def _music(seconds, sr=SAMPLE_RATE):
    return _click_track(120.0, seconds, noise=0.01, seed=3) * 0.8


def _add_scratch(x, at_s, n_clicks=6, gap=0.11, sr=SAMPLE_RATE, seed=5):
    rng = np.random.default_rng(seed)
    for k in range(n_clicks):
        start = int((at_s + k * gap + rng.uniform(-0.02, 0.02)) * sr)
        dur = int(0.004 * sr)
        burst = rng.standard_normal(dur).astype(np.float32)
        burst = np.diff(np.concatenate([[0.0], burst]))  # emphasise highs
        x[start : start + dur] += 0.9 * burst / np.max(np.abs(burst))
    return x


def test_scratch_detected_on_music_and_not_on_music_alone():
    sr = SAMPLE_RATE
    x = _add_scratch(_music(6.0), 3.0)
    det = ScratchDetector()
    events = []
    now = 0.0
    for i in range(0, len(x), 320):
        if det.push(x[i : i + 320], now):
            events.append(now)
        now += 320 / sr
    assert len(events) == 1 and 3.2 < events[0] < 4.5, events

    det2 = ScratchDetector()
    y = _music(8.0)
    fired = False
    now = 0.0
    for i in range(0, len(y), 320):
        fired |= det2.push(y[i : i + 320], now)
        now += 320 / sr
    assert not fired


def test_scratch_ignores_speech_fixtures():
    import soundfile as sf
    from pathlib import Path

    fix = Path(__file__).parent / "fixtures"
    for name in ("m3_reachy_dance", "hello_there", "m3_what_a_nice_day", "peachy"):
        d, sr = sf.read(fix / f"{name}.wav", dtype="float32")
        det = ScratchDetector()
        fired = False
        now = 0.0
        for gain in (0.7, 1.0):
            for i in range(0, len(d), 320):
                fired |= det.push(gain * d[i : i + 320], now)
                now += 320 / sr
            now += 2.0
        assert not fired, name


def _feed_rub(det, x, chunk=320):
    events, now = [], 0.0
    for i in range(0, len(x), chunk):
        if det.push(x[i : i + chunk], now):
            events.append(round(now, 2))
        now += chunk / SAMPLE_RATE
    return events


def test_rub_detected_only_for_sustained_handling_noise():
    from festival_pet.audio_features import RubDetector

    sr = SAMPLE_RATE
    rng = np.random.default_rng(2)
    quiet = rng.standard_normal(sr * 4).astype(np.float32) * 0.003
    rub = rng.standard_normal(int(sr * 1.5)).astype(np.float32) * 0.5  # loud broadband handling noise
    x = np.concatenate([quiet, rub, quiet])
    events = _feed_rub(RubDetector(), x)
    assert len(events) == 1 and 4.4 < events[0] < 4.9, events

    # music alone (harmonic, not flat) never counts as a pet, even when loud
    music = _click_track(120.0, 8.0, noise=0.01) * 2.0
    assert _feed_rub(RubDetector(), music) == []

    # a scratch burst is too short to be a rub
    y = _add_scratch(np.concatenate([quiet, quiet]), 4.0)
    assert _feed_rub(RubDetector(), y) == []
