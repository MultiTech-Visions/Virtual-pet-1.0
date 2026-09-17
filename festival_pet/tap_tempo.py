"""A hand-tapped beat clock: tap the tempo in, mark the "1", and read beat / bar / phrase phase.

Pure Python, no threads. ``tap(now)`` on every beat; ``tap(now, downbeat=True)`` on the first
beat of a bar. The tempo is the median of the last few tap intervals so one sloppy tap does
not throw it, and a long pause starts a fresh count. Once running, the clock free-wheels from
the last tap, so you can stop tapping and it keeps time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

MIN_TAP_BPM, MAX_TAP_BPM = 40.0, 220.0
RESET_GAP_S = 2.5  # a pause longer than this between taps starts a new count
BEATS_PER_BAR = 4
BARS_PER_PHRASE = 4


@dataclass
class TapTempo:
    bpm: float = 0.0
    _taps: list[float] = field(default_factory=list)
    _anchor: float = 0.0  # a beat falls exactly here
    _bar_anchor: float | None = None  # beat "1" fell here; None until told

    @property
    def active(self) -> bool:
        return self.bpm > 0.0

    @property
    def period(self) -> float:
        return 60.0 / self.bpm

    def tap(self, now: float, downbeat: bool = False) -> float:
        """Register a tap; returns the current bpm (0 until two taps are in)."""
        if self._taps and now - self._taps[-1] > RESET_GAP_S:
            self._taps.clear()
        self._taps.append(now)
        del self._taps[:-8]
        if len(self._taps) >= 2:
            gaps = [b - a for a, b in zip(self._taps, self._taps[1:])]
            bpm = 60.0 / median(gaps)
            if MIN_TAP_BPM <= bpm <= MAX_TAP_BPM:
                self.bpm = bpm
        self._anchor = now  # every tap re-phases the beat, so drifting can be corrected by tapping along
        if downbeat:
            self._bar_anchor = now
        elif self._bar_anchor is not None and self.active:
            # keep the bar anchor on the same beat index it had: snap it to the nearest whole beat behind this tap
            beats = round((now - self._bar_anchor) / self.period)
            self._bar_anchor = now - beats * self.period
        return self.bpm

    def set_bpm(self, bpm: float, beat_at: float | None = None) -> None:
        """Set the tempo directly; ``beat_at`` (a time a beat falls on) also re-phases the clock."""
        if not MIN_TAP_BPM <= bpm <= MAX_TAP_BPM:
            raise ValueError(f"bpm {bpm} outside {MIN_TAP_BPM}-{MAX_TAP_BPM}")
        self.bpm = float(bpm)
        if beat_at is not None:
            self._anchor = beat_at

    def clear(self) -> None:
        self.bpm = 0.0
        self._taps.clear()
        self._bar_anchor = None

    @property
    def downbeat_known(self) -> bool:
        return self._bar_anchor is not None

    def phase(self, now: float) -> float:
        """0..1 within the beat (0 = on the beat)."""
        return ((now - self._anchor) / self.period) % 1.0

    def bar_phase(self, now: float) -> float:
        """0..1 over a 4-beat bar; counted from the last "1" if given, else from the last tap."""
        origin = self._bar_anchor if self._bar_anchor is not None else self._anchor
        return ((now - origin) / (BEATS_PER_BAR * self.period)) % 1.0

    def phrase_phase(self, now: float) -> float:
        """0..1 over a 4-bar (16-beat) phrase, from the last "1"."""
        origin = self._bar_anchor if self._bar_anchor is not None else self._anchor
        return ((now - origin) / (BEATS_PER_BAR * BARS_PER_PHRASE * self.period)) % 1.0

    def beat_in_bar(self, now: float) -> int:
        """1..4"""
        return int(self.bar_phase(now) * BEATS_PER_BAR) + 1

    def bar_in_phrase(self, now: float) -> int:
        """1..4"""
        return int(self.phrase_phase(now) * BARS_PER_PHRASE) + 1
