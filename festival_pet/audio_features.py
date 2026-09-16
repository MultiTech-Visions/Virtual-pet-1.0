"""Microphone features: beat tracking for grooving, and shell-scratch (tickle) detection.

Both consume the same mono 16 kHz float32 stream in small hops. Pure numpy so
they can be tuned against synthetic signals off-robot.

BeatTracker
    spectral-flux onset envelope at HOP samples -> autocorrelation over a
    rolling window -> tempo (60..180 BPM) + beat phase. ``confidence`` tells
    the behavior layer whether there really is music.

ScratchDetector
    Fingernails on the shell reach the mics as structure-borne clicks: very
    short, broadband, strong above 3 kHz, in quick irregular bursts. We flag
    high-band onsets far above the recent median and look for a burst of them.
    Sustained music mostly lives below 3 kHz and rarely makes 4+ clicks in a
    second with 40-250 ms gaps, which is the discriminator.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000
HOP = 512  # 32 ms -> 31.25 onset frames per second
FRAME_RATE = SAMPLE_RATE / HOP


class _Framer:
    """Accumulates arbitrary chunks and yields fixed HOP-sized frames."""

    def __init__(self, hop: int) -> None:
        self._hop = hop
        self._buf = np.zeros(0, dtype=np.float32)

    def push(self, mono: np.ndarray) -> list[np.ndarray]:
        self._buf = np.concatenate([self._buf, mono.astype(np.float32, copy=False)])
        frames = []
        while len(self._buf) >= self._hop:
            frames.append(self._buf[: self._hop])
            self._buf = self._buf[self._hop :]
        return frames


@dataclass
class BeatState:
    bpm: float
    confidence: float  # 0..1-ish; > MUSIC_CONFIDENCE means "there is music"
    phase: float  # 0..1 position within the current beat, 0 = on the beat
    period: float  # seconds per beat


MUSIC_CONFIDENCE = 0.35
MIN_BPM, MAX_BPM = 60.0, 180.0


class BeatTracker:
    """Tempo + phase from spectral flux, cheap enough for the Pi (one 1024-pt FFT per 32 ms)."""

    def __init__(self, window_s: float = 6.0, sample_rate: int = SAMPLE_RATE) -> None:
        self._sr = sample_rate
        self._framer = _Framer(HOP)
        self._win = np.hanning(HOP * 2).astype(np.float32)
        self._prev_frame = np.zeros(HOP, dtype=np.float32)
        self._prev_mag: np.ndarray | None = None
        n = int(window_s * FRAME_RATE)
        self._env = np.zeros(n, dtype=np.float32)
        self._env_t = np.zeros(n, dtype=np.float64)  # wall time of each frame
        self._i = 0
        self._filled = 0
        self.state = BeatState(0.0, 0.0, 0.0, 0.0)
        self._last_beat_time = 0.0
        self._bpm_history: deque[float] = deque(maxlen=int(3 * FRAME_RATE / 8))
        self._frames_since_eval = 0

    def push(self, mono: np.ndarray, now: float) -> None:
        for frame in self._framer.push(mono):
            self._onset(frame, now)
            self._frames_since_eval += 1
            if self._frames_since_eval >= 8:  # re-estimate ~4x per second
                self._frames_since_eval = 0
                self._estimate(now)

    def _onset(self, frame: np.ndarray, now: float) -> None:
        x = np.concatenate([self._prev_frame, frame]) * self._win
        self._prev_frame = frame
        mag = np.abs(np.fft.rfft(x))[: HOP // 2]  # keep < 4 kHz: rhythm lives low
        logmag = np.log1p(10.0 * mag)
        if self._prev_mag is None:
            flux = 0.0
        else:
            flux = float(np.sum(np.maximum(logmag - self._prev_mag, 0.0)))
        self._prev_mag = logmag
        self._env[self._i] = flux
        self._env_t[self._i] = now
        self._i = (self._i + 1) % len(self._env)
        self._filled = min(self._filled + 1, len(self._env))

    def _ordered(self) -> tuple[np.ndarray, np.ndarray]:
        env = np.roll(self._env, -self._i)[-self._filled :]
        t = np.roll(self._env_t, -self._i)[-self._filled :]
        return env, t

    def _estimate(self, now: float) -> None:
        if self._filled < int(3 * FRAME_RATE):
            return
        env, t = self._ordered()
        env = env - np.mean(env)
        if float(np.std(env)) < 1e-6:
            self.state = BeatState(0.0, 0.0, 0.0, 0.0)
            return
        env = env / np.std(env)
        n = len(env)
        ac = np.correlate(env, env, mode="full")[n - 1 :] / n
        min_lag = int(FRAME_RATE * 60.0 / MAX_BPM)
        max_lag = int(FRAME_RATE * 60.0 / MIN_BPM)
        lags = np.arange(min_lag, max_lag + 1)
        scores = ac[min_lag : max_lag + 1].copy()
        # Mild preference for typical dance tempos so 2x/0.5x ambiguities resolve sensibly.
        bpms = 60.0 * FRAME_RATE / lags
        scores *= np.exp(-0.5 * ((np.log(bpms / 118.0)) / 0.45) ** 2) * 0.5 + 0.5
        k = int(np.argmax(scores))
        lag = int(lags[k])
        # Parabolic refinement of the lag for a smoother BPM.
        if 0 < k < len(scores) - 1:
            a, b, c = scores[k - 1], scores[k], scores[k + 1]
            denom = a - 2 * b + c
            frac = 0.5 * (a - c) / denom if abs(denom) > 1e-9 else 0.0
            lag_f = lag + float(np.clip(frac, -0.5, 0.5))
        else:
            lag_f = float(lag)
        period = lag_f / FRAME_RATE
        bpm = 60.0 / period
        confidence = float(np.clip(ac[lag], 0.0, 1.0))

        # Phase: the offset (0..lag-1) where a comb of beats best lines up with recent onsets.
        recent = env[-int(4 * lag) :]
        offsets = np.arange(lag)
        comb_scores = np.array([np.sum(recent[o::lag]) for o in offsets])
        best_off = int(np.argmax(comb_scores))
        # Frame index (within `recent`) of the most recent beat.
        last_beat_idx = best_off + ((len(recent) - 1 - best_off) // lag) * lag
        last_beat_time = float(t[-len(recent) :][last_beat_idx])
        # Keep the beat clock continuous: only snap when we drift by more than a quarter beat.
        if self.state.period > 0 and abs(period - self.state.period) / self.state.period < 0.08:
            predicted = self._last_beat_time + round((last_beat_time - self._last_beat_time) / period) * period
            if abs(predicted - last_beat_time) < 0.25 * period:
                last_beat_time = 0.8 * predicted + 0.2 * last_beat_time
        self._last_beat_time = last_beat_time
        self._bpm_history.append(bpm)
        stable = len(self._bpm_history) >= 4 and (max(self._bpm_history) - min(self._bpm_history)) / bpm < 0.08
        if not stable:
            confidence *= 0.5
        self.state = BeatState(bpm, confidence, self.phase(now), period)

    def phase(self, now: float) -> float:
        if self.state.period <= 0:
            return 0.0
        return ((now - self._last_beat_time) / self.state.period) % 1.0

    @property
    def music(self) -> bool:
        return self.state.confidence >= MUSIC_CONFIDENCE and self.state.bpm > 0


class ScratchDetector:
    """Detects bursts of fingernail clicks on the shell from the mic stream."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        hop: int = 256,
        band_hz: tuple[float, float] = (3000.0, 7600.0),
        onset_ratio: float = 6.0,
        floor: float = 1e-4,
        min_clicks: int = 4,
        burst_window_s: float = 1.2,
        gap_range_s: tuple[float, float] = (0.04, 0.30),
        cooldown_s: float = 3.0,
    ) -> None:
        self._sr = sample_rate
        self._hop = hop
        self._framer = _Framer(hop)
        self._win = np.hanning(hop).astype(np.float32)
        freqs = np.fft.rfftfreq(hop, 1.0 / sample_rate)
        self._band = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
        self._low = freqs < 1500.0
        self.onset_ratio = onset_ratio
        self.floor = floor
        self._min_clicks = min_clicks
        self._burst_window = burst_window_s
        self._gap = gap_range_s
        self._cooldown = cooldown_s
        self._history: deque[float] = deque(maxlen=int(1.0 * sample_rate / hop))  # ~1 s of band energy
        self._clicks: deque[float] = deque()
        self._last_event = -1e9
        self._frame_index = 0
        self._last_click_frame = -10
        self._candidate: tuple[int, float] | None = None  # (frame index, peak energy) awaiting decay check
        self.stats = {"band_energy": 0.0, "median": 0.0, "clicks_in_window": 0}

    def push(self, mono: np.ndarray, now: float) -> bool:
        """Feed audio; returns True once when a scratch burst is recognised."""
        event = False
        for frame in self._framer.push(mono):
            t = now  # chunk granularity is fine for 40 ms+ gaps
            spec = np.abs(np.fft.rfft(frame * self._win))
            hb = float(np.mean(spec[self._band] ** 2))
            lb = float(np.mean(spec[self._low] ** 2)) + 1e-12
            med = float(np.median(self._history)) if len(self._history) >= 8 else hb
            self._history.append(hb)
            self._frame_index += 1
            self.stats["band_energy"], self.stats["median"] = hb, med
            is_click = (
                hb > self.floor
                and hb > self.onset_ratio * (med + 1e-12)
                and hb > 0.5 * lb  # broadband/bright, not a bass thump
                and self._frame_index - self._last_click_frame >= 2  # one click = one event
            )
            # A fingernail click is over in one or two 16 ms frames; a consonant or a hi-hat rings longer.
            if self._candidate is not None:
                idx, peak = self._candidate
                if hb < 0.25 * peak:
                    self._candidate = None
                    self._last_click_frame = self._frame_index
                    self._clicks.append(t)
                elif self._frame_index - idx >= 2:
                    self._candidate = None  # sustained: not a click
            elif is_click:
                self._candidate = (self._frame_index, hb)
            while self._clicks and t - self._clicks[0] > self._burst_window:
                self._clicks.popleft()
            self.stats["clicks_in_window"] = len(self._clicks)
            if len(self._clicks) >= self._min_clicks and t - self._last_event > self._cooldown:
                gaps = np.diff(np.array(self._clicks))
                if len(gaps) and np.all(gaps >= 0.0) and np.mean(gaps) <= self._gap[1]:
                    self._last_event = t
                    self._clicks.clear()
                    event = True
        return event


class RubDetector:
    """Detects a hand rubbing the head: the mics sit in the head, so petting is loud, noisy, sustained.

    Handling noise is broadband and flat (noise-like, spectral flatness high), far
    above the ambient level in the low-mid band, and lasts a good fraction of a
    second. Music is harmonic (low flatness); a scratch is a train of clicks
    (too short); speech is neither flat nor that loud at the mic capsules.
    Emits an edge when a rub starts and keeps ``rubbing`` true while it lasts.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        hop: int = 512,
        band_hz: tuple[float, float] = (80.0, 2500.0),
        level_ratio: float = 12.0,
        flatness_min: float = 0.35,
        floor: float = 3e-3,
        min_duration_s: float = 0.5,
        release_s: float = 0.4,
        cooldown_s: float = 4.0,
    ) -> None:
        self._sr = sample_rate
        self._hop = hop
        self._framer = _Framer(hop)
        self._win = np.hanning(hop).astype(np.float32)
        freqs = np.fft.rfftfreq(hop, 1.0 / sample_rate)
        self._band = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
        self.level_ratio = level_ratio
        self.flatness_min = flatness_min
        self.floor = floor
        self._min_duration = min_duration_s
        self._release = release_s
        self._cooldown = cooldown_s
        self._history: deque[float] = deque(maxlen=int(3.0 * sample_rate / hop))  # ~3 s of band energy
        self._rub_since: float | None = None
        self._last_rub_frame_t = -1e9
        self._last_event = -1e9
        self._frames_in_rub = 0
        self._loud_in_rub = 0
        self.rubbing = False
        self.stats = {"rub_energy": 0.0, "rub_baseline": 0.0, "flatness": 0.0}

    def push(self, mono: np.ndarray, now: float) -> bool:
        """Feed audio; returns True once when a rub is recognised (``rubbing`` stays True while it lasts)."""
        event = False
        for frame in self._framer.push(mono):
            spec = np.abs(np.fft.rfft(frame * self._win)) ** 2
            band = spec[self._band]
            energy = float(np.mean(band))
            # Spectral flatness: geometric mean / arithmetic mean of the band power.
            flat = float(np.exp(np.mean(np.log(band + 1e-12))) / (energy + 1e-12))
            base = float(np.median(self._history)) if len(self._history) >= 10 else energy
            self._history.append(energy)
            self.stats["rub_energy"], self.stats["rub_baseline"], self.stats["flatness"] = energy, base, flat
            loud_flat = energy > self.floor and energy > self.level_ratio * base and flat > self.flatness_min
            if loud_flat:
                self._last_rub_frame_t = now
                if self._rub_since is None:
                    self._rub_since = now
                    self._frames_in_rub = self._loud_in_rub = 0
            if self._rub_since is not None:
                self._frames_in_rub += 1
                self._loud_in_rub += int(loud_flat)
            if self._rub_since is not None and now - self._last_rub_frame_t > self._release:
                self._rub_since = None
                self.rubbing = False
            # A rub is *continuously* loud; a train of clicks is loud only a few frames out of ten.
            dense = self._frames_in_rub > 0 and self._loud_in_rub / self._frames_in_rub >= 0.6
            if self._rub_since is not None and now - self._rub_since >= self._min_duration and dense and not self.rubbing:
                self.rubbing = True
                if now - self._last_event > self._cooldown:
                    self._last_event = now
                    event = True
        return event


class LevelMeter:
    """Cheap per-chunk levels for the tuning page: RMS, peak, and short history."""

    def __init__(self, history_s: float = 20.0, rate_hz: float = 10.0) -> None:
        self._hist: deque[dict] = deque(maxlen=int(history_s * rate_hz))
        self._next_sample = 0.0
        self._period = 1.0 / rate_hz
        self.rms = 0.0
        self.peak = 0.0
        self.max_rms_3s = 0.0
        self._recent_rms: deque[tuple[float, float]] = deque()

    def push(self, mono: np.ndarray, now: float, extra: dict) -> None:
        if len(mono) == 0:
            return
        self.rms = float(np.sqrt(np.mean(mono.astype(np.float32) ** 2)))
        self.peak = float(np.max(np.abs(mono)))
        self._recent_rms.append((now, self.rms))
        while self._recent_rms and now - self._recent_rms[0][0] > 3.0:
            self._recent_rms.popleft()
        self.max_rms_3s = max(r for _, r in self._recent_rms)
        if now >= self._next_sample:
            self._next_sample = now + self._period
            self._hist.append({"t": round(now, 2), "rms": round(self.rms, 5), "peak": round(self.peak, 4), **extra})

    def history(self) -> list[dict]:
        return list(self._hist)


def _tune(cls_name: str, baseline: dict, active: dict) -> dict:
    """Pick thresholds halfway (geometrically) between what quiet and the touch looked like."""
    if cls_name == "rub":
        e_q, e_a = max(baseline["rub_energy"], 1e-9), max(active["rub_energy"], 1e-9)
        ratio = e_a / e_q
        return {
            "level_ratio": max(2.0, ratio ** 0.5),
            "flatness_min": max(0.1, 0.7 * active["flatness"]),
            "floor": (e_q * e_a) ** 0.5,
            "observed_ratio": ratio,
        }
    if cls_name == "scratch":
        e_q, e_a = max(baseline["band_energy"], 1e-9), max(active["band_energy"], 1e-9)
        ratio = e_a / e_q
        return {"onset_ratio": max(2.0, ratio ** 0.5), "floor": (e_q * e_a) ** 0.5, "observed_ratio": ratio}
    raise KeyError(cls_name)
