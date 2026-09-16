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

    Held  = sustained motion: accel magnitude keeps deviating from g or the gyro
            keeps moving, over a short window. Robust to the robot's own head
            motion because the IMU sits in the base, not the head.
    Shaken = gyro or jerk spikes well above anything a person does while cuddling.
    """

    window_s: float = 0.6
    lift_accel_dev: float = 1.6  # m/s^2 away from g counts as "moving"
    lift_gyro: float = 0.6  # rad/s
    motion_fraction_to_hold: float = 0.5
    settle_s: float = 2.0  # quiet this long -> set down
    shake_gyro: float = 4.0  # rad/s
    shake_accel_dev: float = 7.0

    def __post_init__(self) -> None:
        self._samples: deque[tuple[float, bool]] = deque()
        self.held = False
        self._last_motion = 0.0
        self._last_shake = 0.0

    def update(self, accel: list[float], gyro: list[float], now: float) -> tuple[bool, bool]:
        """Feed one IMU reading; returns (held, shaken_this_sample)."""
        a_dev = abs(math.sqrt(sum(x * x for x in accel)) - G)
        g_mag = math.sqrt(sum(x * x for x in gyro))
        moving = a_dev > self.lift_accel_dev or g_mag > self.lift_gyro
        shaken = (g_mag > self.shake_gyro or a_dev > self.shake_accel_dev) and now - self._last_shake > 1.0
        if shaken:
            self._last_shake = now
        if moving:
            self._last_motion = now

        self._samples.append((now, moving))
        while self._samples and now - self._samples[0][0] > self.window_s:
            self._samples.popleft()
        frac = sum(1 for _, m in self._samples if m) / max(1, len(self._samples))

        if not self.held and frac >= self.motion_fraction_to_hold:
            self.held = True
        elif self.held and now - self._last_motion > self.settle_s:
            self.held = False
        return self.held, shaken


@dataclass
class TouchDetector:
    """Antenna 'petting': present antenna angle deviates from what we commanded.

    The daemon's own wake gesture uses 0.25 rad press / 0.10 rad release; we
    mirror that. Returns an edge (True only on the tick the press begins).
    """

    press_rad: float = 0.25
    release_rad: float = 0.10
    busy_scale: float = 1.8  # larger threshold while the antennas are being animated (they lag their command)
    persist_ticks: int = 3  # deviation must hold this many consecutive updates

    def __post_init__(self) -> None:
        self._pressed = [False, False]
        self._over = [0, 0]

    def update(self, commanded: list[float], present: list[float], busy: bool = False) -> bool:
        edge = False
        press = self.press_rad * (self.busy_scale if busy else 1.0)
        for i in range(2):
            dev = abs(present[i] - commanded[i])
            if dev > press:
                self._over[i] += 1
            else:
                self._over[i] = 0
            if not self._pressed[i] and self._over[i] >= self.persist_ticks:
                self._pressed[i] = True
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
