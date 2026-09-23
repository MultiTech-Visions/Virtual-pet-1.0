"""Seeing a beat instead of hearing one: rhythmic body/face motion = dancing.

Feed the tracked target's direction in the WORLD frame (yaw, pitch in degrees, as the
gaze code computes it from the pixel position and the head pose at capture) at whatever
rate the detector runs (~8 Hz for faces, ~2.5 Hz for bodies). World angles matter: the
camera sits in the head, so once the pet bobs along, the face bobs in the image by the
pet's own motion, and raw image coordinates would measure the pet, not the person.
A person dancing bobs their head/torso at 50-150 BPM (0.8-2.5 Hz); we look for a
strong autocorrelation peak of the vertical (and horizontal) motion over a rolling window.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

MIN_BPM, MAX_BPM = 50.0, 150.0


@dataclass
class DanceState:
    dancing: bool
    bpm: float
    confidence: float
    amplitude: float  # degrees, peak-to-peak-ish
    since: float  # when dancing started (0 if not)


class DanceDetector:
    def __init__(self, window_s: float = 5.0, min_amp: float = 1.5, min_conf: float = 0.45, hold_s: float = 2.5, release_bars: float = 8.0, min_release_s: float = 16.0) -> None:
        self._win = window_s
        self._min_amp = min_amp
        self._min_conf = min_conf
        self._hold = hold_s
        # Once dancing, keep dancing at the last tempo for this long after the rhythm was last confirmed:
        # a dancer does not stop because the tracker blinked, and music comes in phrases of bars anyway.
        self._release_bars = release_bars
        self._min_release = min_release_s
        self._locked_until = 0.0
        self._pts: deque[tuple[float, float, float]] = deque()
        self._rhythmic_since = 0.0
        self._last_phase_anchor = 0.0
        self._period = 0.0
        self.state = DanceState(False, 0.0, 0.0, 0.0, 0.0)

    def push(self, now: float, yaw_deg: float, pitch_deg: float) -> DanceState:
        self._pts.append((now, yaw_deg, pitch_deg))
        while self._pts and now - self._pts[0][0] > self._win:
            self._pts.popleft()
        if len(self._pts) < 12 or now - self._pts[0][0] < 3.0:
            return self.state
        t = np.array([p[0] for p in self._pts])
        # Resample to a uniform 10 Hz grid so autocorrelation lags mean seconds.
        grid = np.arange(t[0], t[-1], 0.1)
        if len(grid) < 25:
            return self.state
        best = (0.0, 0.0, 0.0)  # conf, bpm, amp
        for axis in (2, 1):
            v = np.interp(grid, t, np.array([p[axis] for p in self._pts]))
            v = v - np.polyval(np.polyfit(grid - grid[0], v, 1), grid - grid[0])  # detrend: walking across is not dancing
            amp = float(np.percentile(v, 95) - np.percentile(v, 5))
            if amp < self._min_amp or float(np.std(v)) < 1e-6:
                continue
            v = v / np.std(v)
            n = len(v)
            ac = np.correlate(v, v, mode="full")[n - 1 :] / np.arange(n, 0, -1)
            ac /= ac[0]
            lo, hi = int(10 * 60.0 / MAX_BPM), int(10 * 60.0 / MIN_BPM)
            seg = ac[lo : hi + 1]
            # Take the SHORTEST lag that is a clear peak near the maximum, so a 110 bpm bob is not read as 55.
            peaks = [i for i in range(1, len(seg) - 1) if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]]
            if not peaks:
                continue
            top = max(seg[i] for i in peaks)
            if top <= 0.0:  # every peak anti-correlated: no rhythm on this axis (and 0.8*top would exceed top)
                continue
            k = min(i for i in peaks if seg[i] >= 0.8 * top)
            conf = float(seg[k])
            lag_s = (lo + k) / 10.0
            if conf > best[0]:
                best = (conf, 60.0 / lag_s, amp)
        conf, bpm, amp = best
        rhythmic = conf >= self._min_conf
        if rhythmic:
            if self._rhythmic_since == 0.0:
                self._rhythmic_since = now
            self._period = 60.0 / bpm
        else:
            self._rhythmic_since = 0.0
        confirmed = rhythmic and now - self._rhythmic_since >= self._hold
        if confirmed:
            self._locked_until = now + max(self._min_release, self._release_bars * 4 * self._period)
        held = self.state.dancing and now < self._locked_until  # sticky: ride out a lost track or a wobble
        dancing = confirmed or held
        if confirmed:
            bpm_out = bpm
        elif held:
            bpm_out = self.state.bpm
        else:
            bpm_out = 0.0
        since = self.state.since if (dancing and self.state.dancing) else (now if dancing else 0.0)
        self.state = DanceState(dancing, bpm_out, conf, amp, since)
        return self.state

    @property
    def locked_for(self) -> float:
        """Seconds of stickiness left from the last confirmation (for the status page)."""
        return self._locked_until

    def phase(self, now: float) -> float:
        """0..1 within the visual beat (anchored to the detector's own clock; good enough to bob along)."""
        if self._period <= 0:
            return 0.0
        return (now / self._period) % 1.0
