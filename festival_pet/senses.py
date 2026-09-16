"""Non-visual senses: pickup / shake from the IMU, petting from antenna displacement, loud sounds.

Pure Python so the thresholds can be unit-tested with synthetic streams.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

G = 9.81


@dataclass
class PickupDetector:
    """Classifies the IMU stream into resting / held / shaken.

    The IMU sits in the HEAD, so the pet's own gestures show up in it. The
    caller therefore passes ``self_moving`` (true while the app is commanding
    fast head motion, plus a short hangover); those samples are ignored
    outright. Between gestures the head is nearly still, so what remains is
    the person: lifting shows as sustained accel-magnitude deviation, and a
    shake as spikes far beyond anything a gesture produces.
    """

    window_s: float = 0.8
    lift_accel_dev: float = 2.2  # m/s^2 away from g counts as "moving" (own slow motion stays well under this)
    lift_gyro: float = 1.2  # rad/s (own slow motion ~0.3)
    motion_fraction_to_hold: float = 0.5
    min_samples: int = 10  # need this many un-gated samples in the window before deciding
    settle_s: float = 2.0  # quiet this long -> set down
    shake_gyro: float = 5.0  # rad/s
    shake_accel_dev: float = 8.0

    def __post_init__(self) -> None:
        self._samples: deque[tuple[float, bool]] = deque()
        self.held = False
        self._last_motion = 0.0
        self._last_shake = 0.0
        self.stats = {"accel_dev": 0.0, "gyro": 0.0, "gated": False}

    def update(self, accel: list[float], gyro: list[float], now: float, self_moving: bool = False) -> tuple[bool, bool]:
        """Feed one IMU reading; returns (held, shaken_this_sample)."""
        a_dev = abs(math.sqrt(sum(x * x for x in accel)) - G)
        g_mag = math.sqrt(sum(x * x for x in gyro))
        self.stats["accel_dev"], self.stats["gyro"], self.stats["gated"] = round(a_dev, 2), round(g_mag, 2), self_moving
        if self_moving:
            # Our own motion: learn nothing from this sample. (Being lifted mid-gesture is caught a moment later.)
            while self._samples and now - self._samples[0][0] > self.window_s:
                self._samples.popleft()
            return self.held, False
        moving = a_dev > self.lift_accel_dev or g_mag > self.lift_gyro
        shaken = (g_mag > self.shake_gyro or a_dev > self.shake_accel_dev) and now - self._last_shake > 1.0
        if shaken:
            self._last_shake = now
        if moving:
            self._last_motion = now

        self._samples.append((now, moving))
        while self._samples and now - self._samples[0][0] > self.window_s:
            self._samples.popleft()
        if len(self._samples) < self.min_samples:
            return self.held, shaken
        frac = sum(1 for _, m in self._samples if m) / len(self._samples)

        if not self.held and frac >= self.motion_fraction_to_hold:
            self.held = True
        elif self.held and now - self._last_motion > self.settle_s:
            self.held = False
        return self.held, shaken


class SelfMotionGate:
    """Tracks how fast the app is commanding the head, so head-mounted sensors can be gated."""

    def __init__(self, angular_rate: float = 0.35, z_rate: float = 0.03, hangover_s: float = 0.4) -> None:
        self._angular_rate, self._z_rate, self._hangover = angular_rate, z_rate, hangover_s
        self._prev: tuple[float, "object"] | None = None
        self._busy_until = -1e9
        self.rate = 0.0

    def update(self, head_pose, now: float) -> bool:
        """Feed the commanded 4x4 pose each tick; returns True while the head is (or just was) moving fast."""
        import numpy as np

        if self._prev is not None:
            t0, prev = self._prev
            dt = now - t0
            if dt > 0:
                rel = prev[:3, :3].T @ head_pose[:3, :3]
                angle = math.acos(max(-1.0, min(1.0, (float(np.trace(rel)) - 1.0) / 2.0)))
                self.rate = angle / dt
                dz = abs(float(head_pose[2, 3] - prev[2, 3])) / dt
                if self.rate > self._angular_rate or dz > self._z_rate:
                    self._busy_until = now + self._hangover
        self._prev = (now, head_pose.copy())
        return now < self._busy_until


@dataclass
class TouchDetector:
    """Antenna 'petting': present antenna angle deviates from what we commanded.

    The daemon's own wake gesture uses 0.25 rad press / 0.10 rad release; we
    mirror that. Returns an edge (True only on the tick the press begins).
    """

    press_rad: float = 0.25
    release_rad: float = 0.10
    busy_scale: float = 1.8  # larger threshold while the antennas are being animated
    persist_ticks: int = 4  # deviation must hold this many consecutive updates
    lag_window: int = 12  # the motor may lag up to this many ticks (~240 ms at 50 Hz) behind the command
    baseline_tau: float = 5.0  # seconds; slow EMA of the resting offset (gravity droop, calibration)

    def __post_init__(self) -> None:
        self._pressed = [False, False]
        self._over = [0, 0]
        self._recent: deque[list[float]] = deque(maxlen=self.lag_window)
        self._baseline = [0.0, 0.0]
        self.last_side = 0
        self.stats = {"dev": [0.0, 0.0], "baseline": [0.0, 0.0]}

    def update(self, commanded: list[float], present: list[float], busy: bool = False, dt: float = 0.02) -> bool:
        """Edge: True on the tick a press begins. ``last_side`` then says which antenna (0 right, 1 left)."""
        edge = False
        self._recent.append(list(commanded))
        press = self.press_rad * (self.busy_scale if busy else 1.0)
        for i in range(2):
            # A lagging motor matches *some* recent command; a finger holds it away from all of them.
            dev = min(abs(present[i] - c[i] - self._baseline[i]) for c in self._recent)
            self.stats["dev"][i] = round(dev, 3)
            if dev > press:
                self._over[i] += 1
            else:
                self._over[i] = 0
                if not busy and not self._pressed[i]:
                    raw = present[i] - commanded[i]
                    self._baseline[i] += (raw - self._baseline[i]) * min(1.0, dt / self.baseline_tau)
                    self.stats["baseline"][i] = round(self._baseline[i], 3)
            if not self._pressed[i] and self._over[i] >= self.persist_ticks:
                self._pressed[i] = True
                self.last_side = i
                edge = True
            elif self._pressed[i] and dev < self.release_rad:
                self._pressed[i] = False
        return edge


@dataclass
class LoudSoundDetector:
    """Turns the mic array's speech/DoA flag into rare 'startle' events.

    A festival is loud all the time, so we only react to speech-flagged energy
    that appears after a quiet stretch, and rate-limit hard.
    """

    quiet_needed_s: float = 6.0
    cooldown_s: float = 20.0

    def __post_init__(self) -> None:
        self._last_speech = -1e9
        self._last_event = -1e9

    def update(self, doa: tuple[float, bool] | None, now: float) -> float | None:
        """Returns the world yaw (deg, + = left) to look toward, or None."""
        if doa is None:
            return None
        angle, speech = doa
        if not speech:
            return None
        was_quiet = now - self._last_speech > self.quiet_needed_s
        self._last_speech = now
        if not was_quiet or now - self._last_event < self.cooldown_s:
            return None
        self._last_event = now
        return self.doa_to_yaw(angle)

    @staticmethod
    def doa_to_yaw(angle: float) -> float:
        """DoA convention: 0 = left, pi/2 = front, pi = right. Head yaw: + = left."""
        yaw = math.degrees(math.pi / 2 - angle)
        return max(-60.0, min(60.0, yaw))
