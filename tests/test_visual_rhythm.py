import math

import numpy as np

from festival_pet.visual_rhythm import DanceDetector


def _feed(det, fn, t0, t1, hz):
    st = None
    t = t0
    while t < t1:
        cx, cy = fn(t)
        st = det.push(t, cx, cy)
        t += 1.0 / hz
    return st


def test_bobbing_head_is_dancing_and_tempo_is_right():
    det = DanceDetector()
    bpm = 110.0
    bob = lambda t: (0.1 * math.sin(2 * math.pi * 0.2 * t), 0.06 * math.sin(2 * math.pi * bpm / 60 * t))
    st = _feed(det, bob, 0.0, 9.0, 8.0)
    assert st.dancing, st
    assert abs(st.bpm - bpm) / bpm < 0.12, st


def test_still_or_drifting_person_is_not_dancing():
    det = DanceDetector()
    still = lambda t: (0.02 * math.sin(0.1 * t), 0.005 * math.sin(0.3 * t))  # tiny wobble
    assert not _feed(det, still, 0.0, 10.0, 8.0).dancing
    det = DanceDetector()
    walk = lambda t: (0.1 * t - 0.5, 0.0)  # walking across, no rhythm
    assert not _feed(det, walk, 0.0, 10.0, 8.0).dancing


def test_body_rate_sampling_still_works():
    det = DanceDetector()
    bob = lambda t: (0.0, 0.08 * math.sin(2 * math.pi * 1.2 * t))  # 72 bpm at 2.5 Hz body rate
    st = _feed(det, bob, 0.0, 10.0, 2.5)
    assert st.dancing and abs(st.bpm - 72.0) < 12, st


def test_anti_correlated_peaks_do_not_crash():
    """Regression: a slow ~0.45 Hz lean-in with jitter left every autocorrelation peak in the 50-150 BPM
    window negative, so the "within 80 % of the top peak" filter matched nothing and min() raised,
    killing the app the first time it greeted someone. Seed 1323 reproduces it on the old code."""
    rng = np.random.default_rng(1323)
    d = DanceDetector()
    t = 0.0
    for _ in range(60):
        t += 0.125
        d.push(t, 0.5 + 0.1 * np.sin(2 * np.pi * 0.45 * t) + rng.normal(0, 0.01),
               0.5 + 0.06 * np.sin(2 * np.pi * 0.45 * t) + rng.normal(0, 0.01))
    assert not d.state.dancing
