"""Needs the Vosk small English model; skipped when it is not available (set VOSK_MODEL)."""

import os
from pathlib import Path

import numpy as np
import pytest

FIX = Path(__file__).parent / "fixtures"
MODEL = os.environ.get("VOSK_MODEL", "")

pytestmark = pytest.mark.skipif(not MODEL or not Path(MODEL).is_dir(), reason="VOSK_MODEL not set")


def _say(spotter, name):
    import soundfile as sf

    d, sr = sf.read(FIX / f"{name}.wav", dtype="float32")
    assert sr == 16000
    d = np.concatenate([np.zeros(3200, np.float32), d, np.zeros(6400, np.float32)])
    words = []
    for i in range(0, len(d), 320):
        spotter.push(d[i : i + 320])
        words += spotter.poll()
    spotter.flush()
    words += spotter.poll()
    return words


def test_name_and_commands():
    from festival_pet.hearing import NameSpotter

    sp = NameSpotter(Path(MODEL))
    assert _say(sp, "m3_reachy") == ["reachy"]
    assert _say(sp, "m3_hey_reachy") == ["reachy"]
    assert _say(sp, "m3_reachy_dance") == ["reachy", "dance"]
    assert _say(sp, "hello_there") == ["hello"]
    assert "reachy" not in _say(sp, "peachy")
    assert _say(sp, "m3_what_a_nice_day") == []
