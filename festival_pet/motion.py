"""Pose composition: idle life + gaze + short gestures, sampled every control tick.

Pure numpy/scipy. ``MotionComposer.sample(now)`` returns ``(head_4x4, antennas_rad)``.

Layering (all additive in yaw/pitch/roll/z, antenna angles):
    base pose for state  (awake neutral vs. sleeping vs. held)
  + breathing            (slow z bob + tiny roll, antenna sway)
  + gaze                 (smoothed toward the behavior's requested yaw/pitch)
  + gesture overlay      (one active gesture at a time, higher priority preempts)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

# Safe operating envelope (SDK clamps ±40° pitch/roll; we stay well inside for cuteness).
YAW_LIMIT = 55.0
PITCH_LIMIT = 28.0
ROLL_LIMIT = 25.0
Z_LIMIT_M = 0.02

ANTENNA_NEUTRAL = (-0.1745, 0.1745)  # SDK INIT_ANTENNAS_JOINT_POSITIONS (right, left)
ANTENNA_DOWN = (-2.6, 2.6)  # relaxed / asleep (sleep pose is ±3.05)


def head_pose(yaw: float, pitch: float, roll: float, z: float) -> np.ndarray:
    """Build a 4x4 head pose from degrees + z metres (matches create_head_pose)."""
    pose = np.eye(4)
    pose[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_matrix()
    pose[2, 3] = z
    return pose


@dataclass
class Offsets:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    z: float = 0.0
    ant_r: float = 0.0  # radians, added to right antenna
    ant_l: float = 0.0

    def __iadd__(self, other: "Offsets") -> "Offsets":
        self.yaw += other.yaw
        self.pitch += other.pitch
        self.roll += other.roll
        self.z += other.z
        self.ant_r += other.ant_r
        self.ant_l += other.ant_l
        return self


# ----------------------------------------------------------------------------- gestures
# Each gesture is f(u) -> Offsets where u in [0, 1] is normalised progress. They are
# designed as short "cartoon" beats that read well from across a room.


def _ease(u: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * u)


def _pulse(u: float) -> float:
    """0 -> 1 -> 0 smooth bump."""
    return math.sin(math.pi * u)


def g_nod(u: float) -> Offsets:
    return Offsets(pitch=12.0 * math.sin(2 * math.pi * u * 2) * (1 - u), ant_r=-0.15 * _pulse(u), ant_l=0.15 * _pulse(u))


def g_tilt(u: float, side: float) -> Offsets:
    return Offsets(roll=side * 18.0 * _pulse(u), yaw=side * 4.0 * _pulse(u), ant_r=0.3 * _pulse(u) * side, ant_l=0.3 * _pulse(u) * side)


def g_wiggle(u: float) -> Offsets:
    s = math.sin(2 * math.pi * 5 * u) * (1 - u)
    return Offsets(yaw=4.0 * s, ant_r=0.5 * s, ant_l=-0.5 * s)


def g_bounce(u: float) -> Offsets:
    s = abs(math.sin(2 * math.pi * 3 * u)) * (1 - 0.5 * u)
    return Offsets(z=0.012 * s, pitch=-4.0 * s, ant_r=-0.6 * s, ant_l=0.6 * s)


def g_perk(u: float) -> Offsets:
    e = _ease(min(1.0, u * 3)) * (1 - max(0.0, u - 0.7) / 0.3)
    return Offsets(z=0.01 * e, pitch=-8.0 * e, ant_r=-0.7 * e, ant_l=0.7 * e)


def g_droop(u: float) -> Offsets:
    e = _ease(min(1.0, u * 2)) * (1 - max(0.0, u - 0.8) / 0.2)
    return Offsets(z=-0.012 * e, pitch=14.0 * e, ant_r=0.9 * e, ant_l=-0.9 * e)


def g_startle(u: float) -> Offsets:
    e = 1.0 if u < 0.25 else max(0.0, 1 - (u - 0.25) / 0.75)
    return Offsets(z=0.015 * e, pitch=-14.0 * e, ant_r=-1.1 * e, ant_l=1.1 * e)


def g_snuggle(u: float) -> Offsets:
    e = _pulse(u)
    return Offsets(roll=16.0 * e, pitch=6.0 * e, z=-0.006 * e, ant_r=0.4 * e, ant_l=-0.4 * e)


def g_dizzy(u: float) -> Offsets:
    a = 2 * math.pi * 2.2 * u
    decay = 1 - u
    return Offsets(yaw=14.0 * math.sin(a) * decay, pitch=8.0 * math.cos(a) * decay, roll=12.0 * math.sin(a + 1) * decay, ant_r=0.5 * math.sin(a) * decay, ant_l=0.5 * math.sin(a) * decay)


def g_shake_off(u: float) -> Offsets:
    s = math.sin(2 * math.pi * 7 * u) * (1 - u)
    return Offsets(yaw=10.0 * s, roll=5.0 * s, ant_r=0.6 * s, ant_l=-0.6 * s)


def g_search(u: float) -> Offsets:
    return Offsets(yaw=22.0 * math.sin(2 * math.pi * 0.7 * u), pitch=-4.0 * _pulse(u), ant_r=-0.3 * _pulse(u), ant_l=0.3 * _pulse(u))


def g_shy(u: float, side: float) -> Offsets:
    # look away and down, antennas fold, then a peek back at the very end
    e = _ease(min(1.0, u * 2.5))
    peek = max(0.0, (u - 0.75) / 0.25)
    return Offsets(yaw=side * 30.0 * e * (1 - 0.7 * peek), pitch=12.0 * e * (1 - peek), roll=-side * 8.0 * e, z=-0.008 * e, ant_r=0.7 * e, ant_l=-0.7 * e)


def g_nod_off(u: float) -> Offsets:
    # head slowly sinks... then jerks awake
    if u < 0.8:
        e = _ease(u / 0.8)
        return Offsets(pitch=18.0 * e, z=-0.01 * e, ant_r=0.5 * e, ant_l=-0.5 * e)
    j = 1.0 - (u - 0.8) / 0.2
    return Offsets(pitch=-6.0 * j, z=0.006 * j, ant_r=-0.4 * j, ant_l=0.4 * j)


def g_sneeze(u: float) -> Offsets:
    # inhale: head back, antennas up ... achoo: snap forward, antennas splay
    if u < 0.45:
        e = _ease(u / 0.45)
        return Offsets(pitch=-12.0 * e, z=0.008 * e, ant_r=-0.8 * e, ant_l=0.8 * e)
    j = (u - 0.45) / 0.55
    snap = math.exp(-6 * j)
    return Offsets(pitch=20.0 * snap, z=-0.012 * snap, ant_r=0.9 * snap, ant_l=-0.9 * snap, roll=4.0 * math.sin(20 * j) * snap)


def g_hiccup(u: float) -> Offsets:
    j = math.exp(-8 * u)
    return Offsets(z=0.012 * j, pitch=-5.0 * j, ant_r=-0.5 * j, ant_l=0.5 * j)


def g_tada(u: float) -> Offsets:
    e = _pulse(u)
    return Offsets(pitch=-10.0 * e, z=0.012 * e, ant_r=-1.2 * e, ant_l=1.2 * e, yaw=6.0 * math.sin(2 * math.pi * 1.5 * u) * e)


def g_flinch(u: float, side: float) -> Offsets:
    """Ear tickled: yank that antenna away and duck the head to the other side, then settle with a shiver.

    side = +1 -> left antenna (index 1) was touched, -1 -> right antenna (index 0).
    """
    snap = math.exp(-5 * u)
    shiver = math.sin(2 * math.pi * 9 * u) * max(0.0, 1 - u * 1.5) * 0.15
    off = Offsets(yaw=-side * 12.0 * snap, roll=-side * 10.0 * snap, z=-0.006 * snap)
    if side > 0:
        off.ant_l = -1.3 * snap + shiver  # fold the left antenna forward/away
        off.ant_r = 0.2 * snap
    else:
        off.ant_r = 1.3 * snap - shiver
        off.ant_l = -0.2 * snap
    return off


def g_lean(u: float) -> Offsets:
    """Head being petted: lean into the hand, eyes-closed feel, antennas relax slowly."""
    e = _ease(min(1.0, u * 2)) * (1 - max(0.0, u - 0.7) / 0.3)
    return Offsets(pitch=8.0 * e, roll=6.0 * e, z=-0.004 * e, ant_r=0.6 * e, ant_l=-0.6 * e)


def g_glance(u: float, side: float) -> Offsets:
    e = _pulse(u) ** 0.6
    return Offsets(yaw=side * 22.0 * e, pitch=-3.0 * e + 5.0 * math.sin(math.pi * u * 2) * e, roll=side * 4.0 * e)


GESTURES: dict[str, tuple[float, str]] = {
    # name: (duration s, kind)  kind "sided" gestures get a random left/right sign
    "nod": (0.9, "plain"),
    "tilt": (1.1, "sided"),
    "wiggle": (0.8, "plain"),
    "bounce": (1.2, "plain"),
    "perk": (1.0, "plain"),
    "droop": (2.2, "plain"),
    "startle": (0.9, "plain"),
    "snuggle": (2.6, "plain"),
    "dizzy": (2.8, "plain"),
    "shake_off": (0.8, "plain"),
    "search": (2.4, "plain"),
    "glance": (2.0, "sided"),
    "shy": (3.0, "sided"),
    "nod_off": (4.0, "plain"),
    "sneeze": (1.4, "plain"),
    "hiccup": (0.5, "plain"),
    "tada": (1.6, "plain"),
    "flinch": (1.2, "sided"),
    "lean": (2.4, "plain"),
}

_FUNCS = {
    "nod": g_nod, "tilt": g_tilt, "wiggle": g_wiggle, "bounce": g_bounce, "perk": g_perk,
    "droop": g_droop, "startle": g_startle, "snuggle": g_snuggle, "dizzy": g_dizzy,
    "shake_off": g_shake_off, "search": g_search, "glance": g_glance,
    "shy": g_shy, "nod_off": g_nod_off, "sneeze": g_sneeze, "hiccup": g_hiccup, "tada": g_tada,
    "flinch": g_flinch, "lean": g_lean,
}


def groove_offsets(phase: float, intensity: float, bar_phase: float, style: int) -> Offsets:
    """Music bob. ``phase`` 0..1 within the beat (0 = on the beat), ``bar_phase`` 0..1 over 4 beats.

    intensity 0.3 = subtle head nod you notice only if you look; 1.0 = little dance.
    Three styles so it does not look like a metronome.
    """
    a = 2 * math.pi * phase
    # Anticipation: the dip lands slightly *before* the beat, like a real nod.
    dip = math.cos(a + 0.35)
    off = Offsets()
    off.pitch += 5.0 * intensity * dip
    off.z += -0.004 * intensity * dip
    if style == 0:  # head bob + antenna sway on the half beat
        off.ant_r += -0.35 * intensity * math.sin(a)
        off.ant_l += -0.35 * intensity * math.sin(a)
    elif style == 1:  # side-to-side lean over two beats
        b = 2 * math.pi * bar_phase * 2
        off.roll += 7.0 * intensity * math.sin(b)
        off.yaw += 5.0 * intensity * math.sin(b)
        off.ant_r += 0.3 * intensity * math.sin(b)
        off.ant_l += 0.3 * intensity * math.sin(b)
    else:  # slow yaw sway over the bar with antennas flicking on beats 2 and 4
        b = 2 * math.pi * bar_phase
        off.yaw += 8.0 * intensity * math.sin(b)
        flick = max(0.0, math.cos(2 * math.pi * (bar_phase * 4 - 1) / 2))
        off.ant_r += -0.5 * intensity * flick
        off.ant_l += 0.5 * intensity * flick
    return off


@dataclass
class _ActiveGesture:
    name: str
    start: float
    duration: float
    priority: int
    side: float


class MotionComposer:
    """Turns behavior intent into a smooth pose stream."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng if rng is not None else random.Random()
        self.mode = "awake"  # "awake" | "sleeping" | "held"
        self.energy = 0.8
        self._gaze_target: tuple[float, float] | None = None
        self._gaze = np.zeros(2)  # yaw, pitch smoothed
        self._drift_phase = self.rng.uniform(0, 100)
        self._gesture: _ActiveGesture | None = None
        self._sleep_blend = 0.0  # 0 awake .. 1 asleep, eased over time
        self.groove: tuple[float, float, float] | None = None  # (beat phase, bar phase, intensity) or None
        self._groove_level = 0.0  # eased intensity so bobbing fades in/out
        self._groove_style = 0
        self._next_style_change = 0.0
        self.mirror_roll = 0.0  # degrees, follows the person's head tilt
        self._mirror = 0.0

    # ------------------------------------------------------------------ intent
    def set_gaze(self, target: tuple[float, float] | None) -> None:
        self._gaze_target = target

    def request_gesture(self, name: str, now: float, priority: int, side: float | None = None) -> bool:
        """Start a gesture unless a higher-priority one is still running. ``side`` forces ±1 for sided gestures."""
        if name not in GESTURES:
            raise KeyError(f"Unknown gesture '{name}'")
        active = self._gesture
        if active is not None and now - active.start < active.duration and active.priority > priority:
            return False
        duration, kind = GESTURES[name]
        side = (side if side is not None else self.rng.choice((-1.0, 1.0))) if kind == "sided" else 1.0
        self._gesture = _ActiveGesture(name, now, duration, priority, side)
        return True

    def gesture_active(self, now: float) -> bool:
        g = self._gesture
        return g is not None and now - g.start < g.duration

    # ------------------------------------------------------------------ sampling
    def _gesture_offsets(self, now: float) -> Offsets:
        g = self._gesture
        if g is None:
            return Offsets()
        u = (now - g.start) / g.duration
        if u >= 1.0:
            self._gesture = None
            return Offsets()
        fn = _FUNCS[g.name]
        if GESTURES[g.name][1] == "sided":
            return fn(u, g.side)  # type: ignore[call-arg]
        return fn(u)  # type: ignore[call-arg]

    def sample(self, now: float, dt: float) -> tuple[np.ndarray, list[float]]:
        """Return (head 4x4, [right, left] antenna radians) for this instant."""
        # sleep blend eases the body down instead of snapping
        target_sleep = 1.0 if self.mode == "sleeping" else 0.0
        self._sleep_blend += (target_sleep - self._sleep_blend) * min(1.0, dt * 1.2)
        s = self._sleep_blend
        amp = 0.5 + 0.5 * self.energy  # low energy = smaller, slower life

        off = Offsets()

        # breathing / life
        breath = math.sin(2 * math.pi * 0.16 * now + self._drift_phase)
        off.z += 0.004 * amp * breath
        off.roll += 1.2 * amp * math.sin(2 * math.pi * 0.07 * now + 1.3)
        off.pitch += 1.0 * amp * math.sin(2 * math.pi * 0.11 * now + 2.1)
        ant_sway = 0.08 * amp * math.sin(2 * math.pi * 0.35 * now)
        off.ant_r += -ant_sway
        off.ant_l += ant_sway

        # slow idle drift when nobody is being looked at
        if self._gaze_target is None:
            drift_yaw = 9.0 * amp * math.sin(2 * math.pi * 0.045 * now + self._drift_phase)
            drift_pitch = 3.0 * amp * math.sin(2 * math.pi * 0.03 * now + 0.7)
            target = np.array([drift_yaw, drift_pitch])
            rate = 1.5
        else:
            target = np.array(self._gaze_target)
            rate = 6.0  # snappy but not twitchy; vision already filters
        self._gaze += (target - self._gaze) * min(1.0, dt * rate)

        # music groove overlay, faded in and out
        want = self.groove[2] if self.groove is not None else 0.0
        self._groove_level += (want - self._groove_level) * min(1.0, dt * 0.8)
        if self.groove is not None and self._groove_level > 0.02:
            if now >= self._next_style_change:
                self._groove_style = self.rng.randrange(3)
                self._next_style_change = now + self.rng.uniform(12.0, 30.0)
            off += groove_offsets(self.groove[0], self._groove_level, self.groove[1], self._groove_style)

        # mirror the person's head tilt a little (slow, so it reads as empathy not tracking)
        self._mirror += (self.mirror_roll * 0.5 - self._mirror) * min(1.0, dt * 0.8)
        off.roll += self._mirror

        # gesture overlay
        off += self._gesture_offsets(now)

        yaw = self._gaze[0] + off.yaw
        pitch = self._gaze[1] + off.pitch
        roll = off.roll
        z = off.z
        ant_r = ANTENNA_NEUTRAL[0] + off.ant_r
        ant_l = ANTENNA_NEUTRAL[1] + off.ant_l

        if self.mode == "held":
            # relaxed, a touch curled-in
            pitch += 5.0
            ant_r += 0.35
            ant_l -= 0.35

        # blend toward sleep pose
        if s > 0.001:
            yaw = yaw * (1 - s)
            pitch = pitch * (1 - s) + 22.0 * s
            roll = roll * (1 - s)
            z = z * (1 - s) - 0.015 * s
            ant_r = ant_r * (1 - s) + ANTENNA_DOWN[0] * s
            ant_l = ant_l * (1 - s) + ANTENNA_DOWN[1] * s

        yaw = max(-YAW_LIMIT, min(YAW_LIMIT, yaw))
        pitch = max(-PITCH_LIMIT, min(PITCH_LIMIT, pitch))
        roll = max(-ROLL_LIMIT, min(ROLL_LIMIT, roll))
        z = max(-Z_LIMIT_M, min(Z_LIMIT_M, z))
        return head_pose(yaw, pitch, roll, z), [ant_r, ant_l]
