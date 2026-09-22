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
YAW_LIMIT = 150.0  # world yaw; the body follows so the head never needs more than HEAD_YAW_LIMIT from it
HEAD_YAW_LIMIT = 30.0  # head relative to body: past this the head hits the body frame (SDK allows 65)
PITCH_LIMIT = 36.0  # SDK clamps at 40; squatting people are low
BODY_YAW_LIMIT = 150.0
BODY_DEADBAND = 12.0  # head can point this far off-body before the body starts turning
BODY_RATE = 45.0  # deg/s, for the ordinary drift after a gaze
BODY_RATE_FAR = 65.0  # deg/s when the gaze is beyond the head's reach: body for coarse, head for fine
BODY_DEADBAND_GROOVE = 20.0  # while grooving the body only follows a gaze this far off it (a real move, not the bob); well inside the head's reach
GROOVE_HOLD_DEG = 40.0  # while grooving, gaze corrections smaller than this are held against (the bob's own jitter); a 60° turn goes through
ROLL_LIMIT = 25.0
Z_LIMIT_M = 0.02

ANTENNA_NEUTRAL = (-0.1745, 0.1745)  # SDK INIT_ANTENNAS_JOINT_POSITIONS (right, left)
ANTENNA_DOWN = (-2.6, 2.6)  # relaxed / asleep (sleep pose is ±3.05)
ARM_BACK, ARM_FORWARD = 2.3, 1.3  # the antenna as an arm: laid back (hanging down), forward-horizontal (straight out)


def arm_rad(deg: float) -> float:
    """The RIGHT antenna's radians for an arm at ``deg`` from hanging down (0) through out (90) to up (180).

    Literal, so a person can read it: arm out = antenna horizontal in front, arm up = antenna vertical,
    arm down = antenna laid back (the hinge is front-to-back, so "down" is as far back as it goes).
    Piecewise linear, continuous, so the dance-along can copy an arm on its way. Negate for the left.
    """
    d = max(0.0, min(180.0, deg))
    if d <= 90.0:
        return -ARM_BACK + (ARM_FORWARD + ARM_BACK) * d / 90.0
    return ARM_FORWARD * (180.0 - d) / 90.0


ARM_LEVEL_DEG = {"down": 0.0, "out": 90.0, "up": 180.0}  # the three flag positions the arm game shows
EAR_HOLD_RATE = 3.5  # how fast a hold takes an antenna over, and hands it back (eased, per second)
EAR_HOLD_SPEED = 4.0  # ...and the fastest the held antenna itself travels, radians per second.
#                       Tuned until a handshake's fastest antenna move is the GESTURE's own speed and
#                       the hold adds nothing on top: past that it reads as a flick, and it startles people.
LEAN_FADE_S = 2.0  # a keypad groove nudge fades out over about this long (a few beats)
STILL_PITCH = 10.0  # head dipped a little while it holds still: an offered head, and a stable antenna
OFFER_RAD = 0.14  # the offered antenna: just past vertical, tipped toward them so a bracelet slides down it
OFFER_AWAY_RAD = 0.9  # ...and the other one leans out of the way (unless it is wearing one: then it stays up)
OFFER_ROLL = 15.0  # the head tips toward the offered side, so that ear is plainly the one being presented
OFFER_YAW = 22.0  # ...and turns AWAY from it, which is what swings that ear round to face the person
CAREFUL_ANTENNA_RAD = 0.45  # a bracelet rides safely as long as its antenna stays this close to vertical...
CAREFUL_ROLL_DEG = 9.0  # ...but only once the head is tilted: level, a bracelet sits at the base of a lowered
CAREFUL_ROLL_FULL = 19.0  # antenna quite happily, which is why shedding one takes a tilt AND a lowering. So the
#                           gate comes in with the tilt, which is what dancing has and what a peace sign does not.
CAREFUL_SCALE = 0.4  # for a little while after a trade it also moves this much of normal, to settle
TILT_Z_FROM = 0.55  # past this fraction of the roll limit the head stops dropping: tilted right over, a
#                     lowered head puts the side of it on the body frame
LEAN_BODY_DEG = 5.0  # how far the body leans into a full nudge (a lean, not a turn: the gaze stays on the person)


def head_pose(yaw: float, pitch: float, roll: float, z: float, x: float = 0.0) -> np.ndarray:
    """Build a 4x4 head pose from degrees + z (and forward x) metres (matches create_head_pose).

    ``x`` is forward along the head's own heading: the pose is in the base frame, so the shift is
    turned with the yaw, or a head looking 90 degrees round would be pushed sideways instead.
    """
    pose = np.eye(4)
    pose[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_matrix()
    # Base frame is x forward, y left, z up (the SDK's sleep pose nests the head at x = -0.021: backwards).
    # An earlier "+y is forward" reading was made with the body turned toward the person, which is exactly
    # the mistake this rotation fixes.
    heading = math.radians(yaw)
    pose[0, 3] = x * math.cos(heading)
    pose[1, 3] = x * math.sin(heading)
    pose[2, 3] = z
    return pose


def turn_pose(pose: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Rotate a head pose about the base's vertical axis.

    Recorded moves are authored with the body facing forward and the SDK takes head poses in the
    base frame, so a move played while the body is turned must be turned with it, or the head is
    asked to twist back across the body.
    """
    turn = np.eye(4)
    turn[:3, :3] = R.from_euler("z", yaw_deg, degrees=True).as_matrix()
    return turn @ pose


@dataclass
class Offsets:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    z: float = 0.0
    ant_r: float = 0.0  # radians, added to right antenna
    ant_l: float = 0.0
    body: float = 0.0  # degrees, added to the body yaw (the head keeps its world heading)
    x: float = 0.0  # metres along the head's own heading (+ forward, - back)

    def __iadd__(self, other: "Offsets") -> "Offsets":
        self.yaw += other.yaw
        self.pitch += other.pitch
        self.roll += other.roll
        self.z += other.z
        self.ant_r += other.ant_r
        self.ant_l += other.ant_l
        self.body += other.body
        self.x += other.x
        return self


# ----------------------------------------------------------------------------- gestures
# Each gesture is f(u) -> Offsets where u in [0, 1] is normalised progress. They are
# designed as short "cartoon" beats that read well from across a room.


def _ease(u: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * u)


def _pulse(u: float) -> float:
    """0 -> 1 -> 0 smooth bump."""
    return math.sin(math.pi * u)


def g_nod(u: float, reps: float = 2.0) -> Offsets:
    """``reps`` full nods over the gesture; the last one fades out."""
    return Offsets(pitch=12.0 * math.sin(2 * math.pi * u * reps) * (1 - 0.6 * u), ant_r=-0.15 * _pulse(u), ant_l=0.15 * _pulse(u))


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
    # Widening sweep left/right, dipping down (kids, squatters) then up (tall people).
    sweep = 28.0 * math.sin(2 * math.pi * 0.55 * u) * min(1.0, 0.4 + u)
    dip = 18.0 * math.sin(2 * math.pi * 0.3 * u)  # + = down first
    return Offsets(yaw=sweep, pitch=dip, ant_r=-0.3 * _pulse(u), ant_l=0.3 * _pulse(u))


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


SNEEZE_S = 10.6  # the whole bit, timed to sounds.render_phrase("sneeze")
ANT_FULL_DOWN = 2.3  # offset from neutral that lays the antennas right back (ANTENNA_DOWN)


def g_sneeze(u: float) -> Offsets:
    """A proper sneeze, in seconds along ``SNEEZE_S`` (the gaze is released for it: it stops looking at you):

    0.0-1.2   uh oh: looks down, antennas drop all the way, a small head shake trying to clear it
    1.2-4.8   build-up: three lifts of the head, each further back, antennas rising a step each time
    4.8-5.4   wind-up: head right back, antennas whip up and cross on top
    5.4-5.7   CHOO: head snaps down hard, antennas swing out wide
    5.7-8.5   slow recovery to neutral, antennas drooped a little
    8.5-10.6  clearing shake, antennas perk back to normal
    """
    t = u * SNEEZE_S
    o = Offsets()
    if t < 1.2:
        e = _ease(min(1.0, t / 0.4))
        o.pitch = 14.0 * e
        o.ant_r, o.ant_l = ANT_FULL_DOWN * e, -ANT_FULL_DOWN * e
        o.yaw = 5.0 * math.sin(2 * math.pi * 2 * (t - 0.4) / 0.8) * (1.0 if 0.4 <= t < 1.2 else 0.0)
    elif t < 4.8:
        k = (t - 1.2) / 3.6
        lift_n = min(2, int(k * 3))
        w = (k * 3) % 1.0
        lift = 6.0 * (lift_n + 1) * _pulse(min(1.0, w / 0.7))
        o.pitch = 14.0 * (1 - k) - lift
        o.z = 0.004 * (lift_n + 1) * _pulse(min(1.0, w / 0.7))
        # antennas: from laid back (2.3) up to near vertical (-0.2), one step per lift
        top = -0.2
        start = ANT_FULL_DOWN + (top - ANT_FULL_DOWN) * lift_n / 3
        end = ANT_FULL_DOWN + (top - ANT_FULL_DOWN) * (lift_n + 1) / 3
        a = start + (end - start) * _ease(min(1.0, w / 0.5))
        o.ant_r, o.ant_l = a, -a
    elif t < 5.4:
        j = _ease((t - 4.8) / 0.6)
        o.pitch = -18.0 - 6.0 * j
        o.z = 0.012
        o.ant_r, o.ant_l = -0.2 + 0.9 * j, 0.2 - 0.9 * j  # past vertical, crossing on top
    elif t < 5.7:
        j = _ease((t - 5.4) / 0.3)
        o.pitch = -24.0 + 48.0 * j
        o.z = 0.012 - 0.024 * j
        o.ant_r, o.ant_l = 0.7 - 2.1 * j, -0.7 + 2.1 * j  # whip out wide (-1.4 / +1.4)
        o.roll = 3.0 * math.sin(30 * j)
    elif t < 8.5:
        r = 1 - _ease((t - 5.7) / 2.8)
        o.pitch = 24.0 * r
        o.z = -0.012 * r
        d = 0.6  # a little droop while it recovers
        o.ant_r, o.ant_l = -1.4 * r + d * (1 - r), 1.4 * r - d * (1 - r)
    else:
        j = (t - 8.5) / 2.1
        sw = math.sin(2 * math.pi * 2.5 * min(1.0, j * 1.6)) * (1 - j)
        o.yaw = 6.0 * sw
        o.ant_r, o.ant_l = 0.6 * (1 - j) - 0.3 * _pulse(j), -0.6 * (1 - j) + 0.3 * _pulse(j)  # droop fades, a small perk
    return o


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


def g_shake(u: float, reps: float = 2.0) -> Offsets:
    """A clear shake-back: ``reps`` yaw swings, antennas along for the ride."""
    sw = math.sin(2 * math.pi * reps * u) * (1 - 0.3 * u)
    return Offsets(yaw=14.0 * sw, ant_r=0.25 * sw, ant_l=0.25 * sw)


def g_lean(u: float) -> Offsets:
    """Head being petted: lean into the hand, eyes-closed feel, antennas relax slowly."""
    e = _ease(min(1.0, u * 2)) * (1 - max(0.0, u - 0.7) / 0.3)
    return Offsets(pitch=8.0 * e, roll=6.0 * e, z=-0.004 * e, ant_r=0.6 * e, ant_l=-0.6 * e)


def g_point(u: float, side: float) -> Offsets:
    """"Turn me that way!": a couple of head bounces, both antennas wiggle, then the antenna on that side
    drops forward and stays pointing while the head strains that way. side +1 = its left."""
    bounce = abs(math.sin(2 * math.pi * 2.5 * min(u, 0.4) / 0.4)) if u < 0.4 else 0.0
    wig = math.sin(2 * math.pi * 6 * u) * (1 - u) * 0.4 if u < 0.45 else 0.0
    pt = _ease(min(1.0, max(0.0, (u - 0.4) / 0.2))) * (1 - max(0.0, u - 0.85) / 0.15)  # the point, held, released at the end
    # pointing = that antenna forward-down (right: +, left: -), the other perked
    ant_r = wig + (1.1 * pt if side < 0 else -0.5 * pt)
    ant_l = -wig + (-1.1 * pt if side > 0 else 0.5 * pt)
    return Offsets(yaw=side * 20.0 * pt, z=0.012 * bounce, pitch=-6.0 * bounce, roll=side * 6.0 * pt, ant_r=ant_r, ant_l=ant_l)


BOW_S = 6.0
BOW_YAWS = (-30.0, 30.0, 0.0)  # to its right, to its left, then the centre: the whole house


def g_bow(u: float) -> Offsets:
    """A performer's bow, three times: to the right, to the left, to the centre.

    The BODY turns between bows and the head goes with it (a head turned on the body slammed the face
    into the body frame on the dip). Each bow: turn while up, dip well forward and hold, come back up.
    The right antenna sweeps across in front on the first, the left on the second, both on the last.
    A solo gesture: the gaze pitch lets go of the person (so the dip is not eaten by a head already
    looking up) but the yaw stays where it was looking, the centre of the house, so the bows are
    30 degrees either side of whoever it was performing for, not of the robot's own front.
    """
    seg = min(2, int(u * 3))
    v = u * 3 - seg
    prev = BOW_YAWS[seg - 1] if seg else 0.0
    yaw = prev + (BOW_YAWS[seg] - prev) * _ease(min(1.0, v / 0.2))
    dip = _ease(min(1.0, max(0.0, (v - 0.15) / 0.25))) * (1 - _ease(min(1.0, max(0.0, (v - 0.72) / 0.28))))
    arm = _pulse(min(1.0, max(0.0, (v - 0.15) / 0.7)))
    ant_r = 1.4 * arm if seg in (0, 2) else 0.0
    ant_l = -1.4 * arm if seg in (1, 2) else 0.0
    # The head slides back as it dips: bowing from the neutral spot, the face hits the front lip of the body.
    # yaw == body: the head keeps facing straight out of the body, so the dip is over the body's own front.
    return Offsets(yaw=yaw, body=yaw, pitch=30.0 * dip, z=-0.01 * dip, x=-0.02 * dip, ant_r=ant_r, ant_l=ant_l)


SWAT_RAD = 1.65  # how far through the swat the antenna sweeps: half as far again as it first reached,
#                  which is what it takes to actually arrive at the hand rather than gesture at it


def g_swat(u: float, side: float) -> Offsets:
    """Not in the mood: the free antenna bats at the hand, three quick taps (no-no-no), with a little turn toward it.

    side = +1 -> the LEFT antenna swats, -1 -> the right one. The other antenna is left to the ear hold.
    """
    sweep = math.sin(2 * math.pi * 3 * min(1.0, u / 0.85)) * (1 - max(0.0, u - 0.85) / 0.15)
    lean = _pulse(u)
    off = Offsets(yaw=side * 8.0 * lean, roll=side * 5.0 * lean, pitch=4.0 * lean)
    if side > 0:
        off.ant_l = -SWAT_RAD * abs(sweep) - 0.45 * lean  # forward, over the head, and back
    else:
        off.ant_r = SWAT_RAD * abs(sweep) + 0.45 * lean
    return off


NUZZLE_S = 4.2
NUZZLE_CIRCLES = 2.0  # how many times round the circle
NUZZLE_DIP = 11.0  # degrees the gaze drops for the whole of it: a lowered, affectionate head
NUZZLE_RISE = 7.0  # ...and how far it comes back up at the top of each circle
NUZZLE_PUSH_M = 0.016  # how far forward it pushes at the widest point


def g_nuzzle(u: float) -> Offsets:
    """A dog pushing its head into you: the gaze drops, then the face goes slowly round a circle — up and
    forward into the hand, over the top, down and back — a couple of times, and eases out.

    The circle is in the (forward, up) plane: ``x`` leads the pitch by a quarter turn, which is what turns
    two sine waves into one slow rolling push rather than a nod.
    """
    e = _ease(min(1.0, u / 0.22)) * (1 - _ease(min(1.0, max(0.0, (u - 0.78) / 0.22))))
    a = 2 * math.pi * NUZZLE_CIRCLES * u
    return Offsets(pitch=(NUZZLE_DIP - NUZZLE_RISE * (1 - math.cos(a))) * e,
                   x=NUZZLE_PUSH_M * math.sin(a) * e,
                   z=-0.004 * e + 0.004 * (1 - math.cos(a)) * e,
                   roll=2.5 * math.sin(a) * e,
                   ant_r=0.5 * e, ant_l=-0.5 * e)


def g_boop(u: float) -> Offsets:
    """Nose booped: the head pops back a hair, the antennas cross inward over the face, then it shakes it off."""
    snap = math.exp(-6 * u)
    cross = _pulse(min(1.0, u / 0.6))
    shake = math.sin(2 * math.pi * 6 * u) * max(0.0, u - 0.5) * 2 * 3.0
    return Offsets(pitch=-6.0 * snap, z=0.006 * snap, x=-0.012 * snap, yaw=shake, ant_r=0.7 * cross, ant_l=-0.7 * cross)


WAVE_S = 3.0
WAVE_SWEEPS = 3.0  # full side-to-side sweeps over the wave: slow enough to read as a wave, not a shake
WAVE_ARC = 0.85  # radians either side of upright the waving antenna swings through
WAVE_ROLL, WAVE_YAW = 14.0, 12.0  # the head lifts that side, and turns the other way to present it
WAVE_TUCK = 0.35  # the other antenna leans back out of the picture
HUG_S = 4.0


def g_wave(u: float, side: float) -> Offsets:
    """A proper parade wave — HEY, OVER HERE — with one antenna. ``side`` +1 = the left, -1 = the right.

    The head lifts that side and turns the other way, which swings that antenna round to the front where
    it can be seen; the antenna itself stands up and sweeps slowly side to side through a big arc. The
    other one leans back out of the way so there is only one thing moving.

    (Antenna angles here are absolute, not offsets from neutral: upright is 0, forward is + on the right
    and - on the left, so the sweep is symmetric about standing straight up.)
    """
    up = _ease(min(1.0, u / 0.18)) * (1 - _ease(min(1.0, max(0.0, (u - 0.82) / 0.18))))
    swing = WAVE_ARC * math.sin(2 * math.pi * WAVE_SWEEPS * u)
    off = Offsets(roll=side * WAVE_ROLL * up, yaw=-side * WAVE_YAW * up, pitch=-6.0 * up, z=0.008 * up)
    if side > 0:  # the left antenna waves: its upright is 0, so the offset is -neutral, plus the sweep
        off.ant_l = (-ANTENNA_NEUTRAL[1] - swing) * up
        off.ant_r = -WAVE_TUCK * up
    else:
        off.ant_r = (-ANTENNA_NEUTRAL[0] + swing) * up
        off.ant_l = WAVE_TUCK * up
    return off


def g_hug(u: float) -> Offsets:
    """A hug: both antennas open wide (back, like a perk), the head lowers and turns aside to nuzzle in,
    and the body rocks gently side to side (about five degrees) for the whole of it."""
    e = _ease(min(1.0, u / 0.25)) * (1 - _ease(min(1.0, max(0.0, (u - 0.8) / 0.2))))
    rock = math.sin(2 * math.pi * 0.7 * u * HUG_S)
    return Offsets(pitch=12.0 * e, roll=14.0 * e, yaw=10.0 * e, z=-0.008 * e, x=0.012 * e,
                   ant_r=-1.0 * e, ant_l=1.0 * e, body=5.0 * rock * e)


PEACE_DIP_U, PEACE_RISE_U = 0.3, 0.3  # fractions of the gesture spent going down, then coming up
PEACE_DOWN_RAD = -ARM_BACK  # laid right back, as low as the hinge goes: the wind-up
PEACE_UP_RAD = 0.12  # a hair off vertical, so the two of them make a Y rather than a post


def g_peace(u: float) -> Offsets:
    """Peace, as a little routine rather than a pose: both antennas sink all the way down, then rise
    together into the Y and hold there with a double bounce.

    Standing them up on their own is barely different from how it stands about all day, which is why the
    trip to the bottom is the gesture: you see them go, and then you see them arrive.
    (Angles are absolute for the right antenna — down is negative, upright is 0 — mirrored for the left.)
    """
    fade = 1 - _ease(min(1.0, max(0.0, (u - 0.88) / 0.12)))
    if u < PEACE_DIP_U:
        k = _ease(u / PEACE_DIP_U)
        a = ANTENNA_NEUTRAL[0] + (PEACE_DOWN_RAD - ANTENNA_NEUTRAL[0]) * k
        pitch, z = 8.0 * k, -0.006 * k
    else:
        k = _ease(min(1.0, (u - PEACE_DIP_U) / PEACE_RISE_U))
        a = PEACE_DOWN_RAD + (PEACE_UP_RAD - PEACE_DOWN_RAD) * k
        a += 0.12 * math.sin(2 * math.pi * 2.0 * (u - PEACE_DIP_U)) * k  # the bounce, once they are up
        pitch, z = 8.0 - 16.0 * k, -0.006 + 0.018 * k
    return Offsets(pitch=pitch * fade, z=z * fade,
                   ant_r=(a - ANTENNA_NEUTRAL[0]) * fade, ant_l=(-a - ANTENNA_NEUTRAL[1]) * fade)


CROSS_RAD = 0.95  # how far past vertical the antennas lean back to cross over behind the head


def g_heart(u: float) -> Offsets:
    """Love: both antennas swing back past vertical and cross over behind its head, and hold there.

    The hinge only runs front-to-back, so a "cross" is the two of them leaning the same way past upright
    until the tips converge over the back of the head — the same trick the sneeze's wind-up uses, going
    the other way. The head tips back a little to show it off.
    """
    e = _ease(min(1.0, u / 0.35)) * (1 - _ease(min(1.0, max(0.0, (u - 0.75) / 0.25))))
    squeeze = 0.08 * math.sin(2 * math.pi * 1.2 * u) * e
    a = -(CROSS_RAD + squeeze)  # absolute, for the right antenna: negative is back
    return Offsets(pitch=-7.0 * e, z=0.006 * e, roll=2.5 * math.sin(2 * math.pi * 0.5 * u) * e,
                   ant_r=(a - ANTENNA_NEUTRAL[0]) * e, ant_l=(-a - ANTENNA_NEUTRAL[1]) * e)


def g_glance(u: float, side: float) -> Offsets:
    e = _pulse(u) ** 0.6
    return Offsets(yaw=side * 22.0 * e, pitch=-3.0 * e + 5.0 * math.sin(math.pi * u * 2) * e, roll=side * 4.0 * e)


GESTURES: dict[str, tuple[float, str]] = {
    # name: (duration s, kind)  kind "sided" gestures get a random left/right sign
    "nod": (0.42, "repeat"),  # duration is per repetition: a quick "yes", not a slow bow
    "tilt": (1.1, "sided"),
    "wiggle": (0.8, "plain"),
    "bounce": (1.2, "plain"),
    "perk": (1.0, "plain"),
    "droop": (2.2, "plain"),
    "startle": (0.9, "plain"),
    "snuggle": (2.6, "plain"),
    "dizzy": (2.8, "plain"),
    "shake_off": (0.8, "plain"),
    "search": (4.5, "plain"),
    "glance": (2.0, "sided"),
    "shy": (3.0, "sided"),
    "nod_off": (4.0, "plain"),
    "sneeze": (SNEEZE_S, "plain"),
    "hiccup": (0.5, "plain"),
    "tada": (1.6, "plain"),
    "flinch": (1.2, "sided"),
    "lean": (2.4, "plain"),
    "shake": (0.55, "repeat"),
    "point": (2.6, "sided"),
    "bow": (BOW_S, "plain"),
    "swat": (1.3, "sided"),
    "nuzzle": (NUZZLE_S, "plain"),
    "boop": (0.7, "plain"),
    "wave": (WAVE_S, "sided"),
    "peace": (2.6, "plain"),
    "heart": (2.6, "plain"),
    "hug": (HUG_S, "plain"),
}

SOLO_GESTURES = frozenset({"sneeze", "bow"})  # the whole body is the gesture: no groove, mimic, mirror or beep sway on top

_FUNCS = {
    "nod": g_nod, "tilt": g_tilt, "wiggle": g_wiggle, "bounce": g_bounce, "perk": g_perk,
    "droop": g_droop, "startle": g_startle, "snuggle": g_snuggle, "dizzy": g_dizzy,
    "shake_off": g_shake_off, "search": g_search, "glance": g_glance,
    "shy": g_shy, "nod_off": g_nod_off, "sneeze": g_sneeze, "hiccup": g_hiccup, "tada": g_tada,
    "flinch": g_flinch, "lean": g_lean, "shake": g_shake, "point": g_point, "bow": g_bow,
    "swat": g_swat, "nuzzle": g_nuzzle, "boop": g_boop, "wave": g_wave, "hug": g_hug, "peace": g_peace, "heart": g_heart,
}


@dataclass
class GrooveMix:
    """How much of each body part joins the groove (0 = none, 1 = normal, 2 = lots)."""

    bob: float = 1.0  # head pitch/z on the beat
    sway: float = 1.0  # head roll/yaw over the bar
    body: float = 0.0  # body yaw over the bar (off by default: it swings the camera too)
    ears: float = 1.0  # antennas

    def as_dict(self) -> dict[str, float]:
        return {"bob": self.bob, "sway": self.sway, "body": self.body, "ears": self.ears}


def groove_offsets(phase: float, intensity: float, bar_phase: float, style: int, mix: GrooveMix | None = None, accent_downbeat: bool = False) -> Offsets:
    """Music bob. ``phase`` 0..1 within the beat (0 = on the beat), ``bar_phase`` 0..1 over 4 beats.

    intensity 0.3 = subtle head nod you notice only if you look; 1.0 = little dance.
    Three styles so it does not look like a metronome. ``mix`` scales each body part;
    ``accent_downbeat`` (when the "1" is known) makes the first beat of the bar land harder.
    """
    mix = mix if mix is not None else GrooveMix()
    a = 2 * math.pi * phase
    # Anticipation: the dip lands slightly *before* the beat, like a real nod.
    dip = math.cos(a + 0.35)
    beat1 = 1.0 + 0.5 * max(0.0, math.cos(2 * math.pi * bar_phase)) ** 4 if accent_downbeat else 1.0
    off = Offsets()
    off.pitch += 5.0 * intensity * mix.bob * dip * beat1
    off.z += -0.004 * intensity * mix.bob * dip * beat1
    if style == 0:  # head bob + antenna sway on the half beat
        off.ant_r += -0.35 * intensity * mix.ears * math.sin(a)
        off.ant_l += -0.35 * intensity * mix.ears * math.sin(a)
    elif style == 1:  # side-to-side lean over two beats
        b = 2 * math.pi * bar_phase * 2
        off.roll += 7.0 * intensity * mix.sway * math.sin(b)
        off.yaw += 5.0 * intensity * mix.sway * math.sin(b)
        off.ant_r += 0.3 * intensity * mix.ears * math.sin(b)
        off.ant_l += 0.3 * intensity * mix.ears * math.sin(b)
    else:  # slow yaw sway over the bar with antennas flicking on beats 2 and 4
        b = 2 * math.pi * bar_phase
        off.yaw += 8.0 * intensity * mix.sway * math.sin(b)
        flick = max(0.0, math.cos(2 * math.pi * (bar_phase * 4 - 1) / 2))
        off.ant_r += -0.5 * intensity * mix.ears * flick
        off.ant_l += 0.5 * intensity * mix.ears * flick
    # body sway: one slow swing per bar, the head holds its heading so it reads as the body moving under it
    off.body += 10.0 * intensity * mix.body * math.sin(2 * math.pi * bar_phase + math.pi / 2)
    return off


@dataclass
class _ActiveGesture:
    name: str
    start: float
    duration: float
    priority: int
    side: float
    reps: int = 1


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
        self.groove_mix = GrooveMix()
        self.groove_phrase: float | None = None  # 0..1 over a 16-beat phrase when the "1" is known (manual clock), else None
        self._groove_level = 0.0  # eased intensity so bobbing fades in/out
        self._groove_style = 0
        self._next_style_change = 0.0
        self._last_phrase = 0.0
        self.mirror_roll = 0.0  # degrees, follows the person's head tilt
        self._mirror = 0.0
        self.still_until = 0.0  # holding dead still (someone is putting something on it)
        self._still = 0.0
        self._still_yaw = 0.0  # where it was looking when it froze: held there, rather than tracking a moving target
        self.offer_side: int | None = None  # 0 right, 1 left: the antenna held out for a bracelet while still
        self.loaded = [False, False]  # antennas wearing a bracelet (right, left): held near vertical so it cannot slide off
        self.gentle_until = 0.0  # ...and for a moment after a trade, everything moves smaller
        self._gentle = 0.0
        self.kandi_damp = 0.5  # 0..1: how much of its head movement to take out for as long as it wears one.
        #                        Upright antennas hold a bracelet through anything, but a bracelet hanging
        #                        off one still swings, and a big move throws it about.
        self._gate = [0.0, 0.0]  # how closed each antenna's upright gate is: it shuts and opens smoothly
        self.body_yaw = 0.0  # degrees, follows the gaze slowly so the head can recenter
        self.body_follow = True
        self._reaiming = False  # mid-way through a big re-aim while grooving: the gaze hold is off until it lands
        self.held = False  # in someone's hands: the body never turns (see Pet.control("pickup"))
        self.petted = False  # a hand on the head: the antennas ease down like a dog's ears
        self._pet_level = 0.0
        self.yaw_short = 0.0  # degrees the head wanted to turn beyond what it can, + = to its left (for asking to be turned)
        self.mimic: tuple[float, float, float] | None = None  # (yaw, pitch, roll) of the person's head to copy, or None
        self.mimic_flip = True  # mirror-image (True) or same-direction copy (False) for yaw and roll
        self.mimic_gain = 0.9
        self._mimic_pose = np.zeros(3)
        self.voice_level = 0.0  # 0..1 loudness of the pet's own beeps; drives a little "talking" sway
        self.forward_shift_m = 0.020  # slide the head forward by up to this when looking up, so it clears the body
        self.hold: tuple[float, float, float, float] | None = None  # (yaw, pitch, roll, until): a shown pose (Simon says)
        # Ear keep-away: an antenna parked somewhere on purpose, overriding every overlay on it.
        # [absolute radians or None] x2, and when each hold ends.
        self.ear_hold: list[float | None] = [None, None]
        self.ear_hold_until = [0.0, 0.0]
        self._ear_away_k = [0, 0]  # keep-away alternates positions per antenna
        self._ear_level = [0.0, 0.0]  # how much a hold has taken each antenna over, eased
        self._ear_rad = [ANTENNA_NEUTRAL[0], ANTENNA_NEUTRAL[1]]  # ...and where it is on the way there
        # The antennas as arms (arm_rad): (robot-left deg, robot-right deg) to show, until when; smoothed fast
        # enough to follow a dancer at 120 bpm. An ear hold (the clock hand) still wins on its antenna.
        self.arms: tuple[float, float] | None = None
        self.arms_until = 0.0
        self._arms_rad = [ANTENNA_NEUTRAL[1], ANTENNA_NEUTRAL[0]]  # [left, right], where they are now
        self._arms_level = 0.0  # how much the arms override the rest (eased in and out)
        # Groove nudge from the keypad: a lean that way (+ = its left) on top of the groove, fading over a few beats
        self.groove_lean = 0.0
        self._lean = 0.0
        self._hold_roll = 0.0
        self._solo_yaw = 0.0  # where it was looking when a solo gesture started: the bow's centre of the house
        self._voice = 0.0
        self._voice_phases = [self.rng.uniform(0, 2 * math.pi) for _ in range(4)]

    # ------------------------------------------------------------------ ears
    # Antenna sign convention: right neutral -0.17, left +0.17; the bigger the magnitude, the further BACK
    # (sleep is +-3.05, flat back). The opposite sign leans the antenna forward over the face.
    EAR_AWAY = (2.3, -0.9)  # keep-away spots, alternating: far back, then forward
    EAR_TUCK = 1.5  # over the head, out of reach

    def ears_away(self, i: int, now: float, hold_s: float = 5.0) -> None:
        """Keep-away: move antenna ``i`` (0 right, 1 left) somewhere else and keep it there a while."""
        sign = -1.0 if i == 0 else 1.0
        mag = self.EAR_AWAY[self._ear_away_k[i] % len(self.EAR_AWAY)]
        self._ear_away_k[i] += 1
        self.ear_hold[i], self.ear_hold_until[i] = sign * mag, now + hold_s

    EAR_TUCK_S = 26.0  # how long "not in the mood" lasts, left alone

    def ears_tuck(self, i: int, now: float, hold_s: float = EAR_TUCK_S) -> None:
        """Not in the mood: antenna ``i`` goes forward over the head and stays there."""
        sign = -1.0 if i == 0 else 1.0
        self.ear_hold[i], self.ear_hold_until[i] = -sign * self.EAR_TUCK, now + hold_s

    EAR_CLOCK_DOWN = 1.5  # horizontal, pointing forward at the person (the hinge runs front-to-back)

    def ear_clock(self, i: int, frac: float, now: float) -> None:
        """A clock hand for a countdown: antenna ``i`` from horizontal-forward (frac 0) up to vertical (frac 1)."""
        sign = -1.0 if i == 0 else 1.0
        self.ear_hold[i] = -sign * self.EAR_CLOCK_DOWN * (1.0 - max(0.0, min(1.0, frac)))
        self.ear_hold_until[i] = now + 0.3

    def ears_clear(self) -> None:
        self.ear_hold = [None, None]
        self._ear_away_k = [0, 0]

    def show_arms(self, left_deg: float | None, right_deg: float | None = None, now: float = 0.0, hold_s: float = float("inf")) -> None:
        """Antennas as arms: ``left_deg`` / ``right_deg`` are the ROBOT's left and right (0 down, 90 out, 180 up)
        for ``hold_s``; ``show_arms(None)`` lets them go."""
        if left_deg is None:
            self.arms = None
            return
        self.arms, self.arms_until = (float(left_deg), float(right_deg)), now + hold_s  # type: ignore[arg-type]

    # ------------------------------------------------------------------ intent
    def hold_still(self, now: float, seconds: float, offer: int | None = None) -> None:
        """Stop moving for ``seconds``: breathing, groove, gestures and all, head bowed a little and the
        antennas parked. For someone threading a kandi bracelet over an antenna, a pet that keeps breathing
        is a pet that drops it.

        ``offer`` (0 right, 1 left) holds that antenna up just past vertical and tipped toward them, with
        the other one leaned away and the head turned a little that way: a post to slide a bracelet onto,
        angled so gravity takes it down to the head rather than off the end.
        """
        self.still_until = max(self.still_until, now + seconds)
        self.offer_side = offer

    def release_still(self) -> None:
        """Come out of it early (the bracelet is on, or someone cancelled). The blend eases back, it does not snap."""
        self.still_until = 0.0
        self.offer_side = None

    def set_gaze(self, target: tuple[float, float] | None) -> None:
        self._gaze_target = target

    def request_gesture(self, name: str, now: float, priority: int, side: float | None = None, reps: int | None = None) -> bool:
        """Start a gesture unless a higher-priority one is still running.

        ``side`` forces ±1 for sided gestures; ``reps`` sets repetitions for repeatable ones (random 2-4 if None).
        """
        if name not in GESTURES:
            raise KeyError(f"Unknown gesture '{name}'")
        active = self._gesture
        if active is not None and now - active.start < active.duration and active.priority > priority:
            return False
        duration, kind = GESTURES[name]
        side = (side if side is not None else self.rng.choice((-1.0, 1.0))) if kind == "sided" else 1.0
        n = 1
        if kind == "repeat":
            n = reps if reps is not None else self.rng.randint(2, 4)
            duration *= n
        self._gesture = _ActiveGesture(name, now, duration, priority, side, n)
        if name in SOLO_GESTURES:
            self._solo_yaw = float(self._gaze[0])
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
        kind = GESTURES[g.name][1]
        if kind == "sided":
            return fn(u, g.side)  # type: ignore[call-arg]
        if kind == "repeat":
            return fn(u, float(g.reps))  # type: ignore[call-arg]
        return fn(u)  # type: ignore[call-arg]

    def sample(self, now: float, dt: float) -> tuple[np.ndarray, list[float], float]:
        """Return (head 4x4 in world frame, [right, left] antenna radians, body yaw degrees) for this instant."""
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

        solo = self._gesture is not None and self._gesture.name in SOLO_GESTURES and now - self._gesture.start < self._gesture.duration
        holding = self.hold is not None and now < self.hold[3]

        # slow idle drift when nobody is being looked at
        if holding:
            target = np.array(self.hold[:2])  # showing a pose: look there, not at them
            rate = 5.0
        elif solo:
            target = np.array([self._solo_yaw, 0.0])  # a solo gesture plays level, facing where it was: it stops looking UP at you
            rate = 5.0
        elif self._gaze_target is None:
            drift_yaw = 9.0 * amp * math.sin(2 * math.pi * 0.045 * now + self._drift_phase)
            drift_pitch = 3.0 * amp * math.sin(2 * math.pi * 0.03 * now + 0.7)
            target = np.array([drift_yaw, drift_pitch])
            rate = 1.5
        else:
            target = np.array(self._gaze_target)
            rate = 6.0  # snappy but not twitchy; vision already filters
        # While grooving, the face reading carries whatever of our own bob the camera timing did not cancel;
        # chasing it would feed that error back into the head and the swing grows. Hold the gaze against
        # those small corrections; a real re-aim (a keypad turn, a new person, a search) still goes through.
        grooving = self._groove_level > 0.05
        away = float(np.max(np.abs(target - self._gaze)))
        if away > GROOVE_HOLD_DEG:
            self._reaiming = True  # a real move: follow it all the way in, hold or no hold
        elif away < 2.0:
            self._reaiming = False
        if grooving and not (solo or holding or self._reaiming):
            rate *= 0.02
        self._gaze += (target - self._gaze) * min(1.0, dt * rate)
        self._hold_roll += ((self.hold[2] if holding else 0.0) - self._hold_roll) * min(1.0, dt * 5.0)
        off.roll += self._hold_roll

        # music groove overlay, faded in and out
        want = self.groove[2] if self.groove is not None and not solo else 0.0
        self._groove_level += (want - self._groove_level) * min(1.0, dt * 0.8)
        if self.groove is not None and self._groove_level > 0.02:
            if self.groove_phrase is not None:
                # The "1" is known: like a dancer counting, change style only where a phrase turns over.
                if self.groove_phrase < self._last_phrase:
                    self._groove_style = (self._groove_style + self.rng.choice((1, 2))) % 3
                self._last_phrase = self.groove_phrase
            elif now >= self._next_style_change:
                self._groove_style = self.rng.randrange(3)
                self._next_style_change = now + self.rng.uniform(12.0, 30.0)
            off += groove_offsets(self.groove[0], self._groove_level, self.groove[1], self._groove_style, self.groove_mix, self.groove_phrase is not None)

        # beep sway: several slow sines gated by the loudness envelope, so the head "talks" with the sound
        self._voice += (self.voice_level - self._voice) * min(1.0, dt * 25.0)
        if self._voice > 0.01 and not solo:
            v = self._voice
            ph = self._voice_phases
            off.pitch += 3.0 * v * math.sin(2 * math.pi * 2.2 * now + ph[0])
            off.yaw += 2.5 * v * math.sin(2 * math.pi * 0.9 * now + ph[1])
            off.roll += 2.0 * v * math.sin(2 * math.pi * 1.4 * now + ph[2])
            off.z += 0.003 * v * math.sin(2 * math.pi * 0.5 * now + ph[3])
            off.ant_r += -0.15 * v
            off.ant_l += 0.15 * v

        # mimic game: copy the person's head pose (mirror-image by default), smoothly, on top of looking at them
        if self.mimic is not None and not solo:
            sign = -1.0 if self.mimic_flip else 1.0
            target = np.array([sign * self.mimic[0], self.mimic[1], sign * self.mimic[2]]) * self.mimic_gain
        else:
            target = np.zeros(3)
        self._mimic_pose += (target - self._mimic_pose) * min(1.0, dt * 3.0)
        off.yaw += float(self._mimic_pose[0])
        off.pitch += float(self._mimic_pose[1])
        off.roll += float(self._mimic_pose[2])

        # otherwise mirror the person's head tilt a little (slow, so it reads as empathy not tracking)
        if self.mimic is None and not solo:
            self._mirror += (max(-20.0, min(20.0, self.mirror_roll * 0.8)) - self._mirror) * min(1.0, dt * 1.5)
        else:
            self._mirror += (0.0 - self._mirror) * min(1.0, dt * 1.5)
        off.roll += self._mirror

        # the keypad's groove nudge: lean that way with the head, the antennas (one forward, one back) and a
        # few degrees of body, fading. The body only leans: it never turns away from whoever it is dancing with.
        self.groove_lean *= math.exp(-dt / LEAN_FADE_S)
        self._lean += (self.groove_lean - self._lean) * min(1.0, dt * 6.0)
        if abs(self._lean) > 0.01 and not solo:
            off.roll += 12.0 * self._lean
            off.yaw += 8.0 * self._lean
            off.body += LEAN_BODY_DEG * self._lean
            off.ant_r -= 0.5 * self._lean
            off.ant_l -= 0.5 * self._lean

        # being petted: the antennas fold back into an X over the head and STAY there, so the hand can
        # settle on the crossing and massage them; the head lifts a little into the hand. Once it has
        # settled in, the tiny push: the antennas close and open by a hair, like ears pressed into a palm.
        self._pet_level += ((1.0 if self.petted else 0.0) - self._pet_level) * min(1.0, dt * (0.6 if self.petted else 0.4))
        p = self._pet_level
        off.z += 0.006 * p

        # gesture overlay
        off += self._gesture_offsets(now)

        yaw = self._gaze[0] + off.yaw
        pitch = self._gaze[1] + off.pitch
        roll = off.roll
        z = off.z
        ant_r = ANTENNA_NEUTRAL[0] + off.ant_r
        ant_l = ANTENNA_NEUTRAL[1] + off.ant_l

        if p > 0.02:
            push = 0.04 * math.sin(2 * math.pi * 0.5 * now) * max(0.0, (p - 0.8) / 0.2)
            pet_r, pet_l = ANTENNA_NEUTRAL[0] + 0.9 + push, ANTENNA_NEUTRAL[1] - 0.9 - push
            ant_r = ant_r * (1 - p) + pet_r * p  # every other antenna overlay fades out as the hand settles
            ant_l = ant_l * (1 - p) + pet_l * p

        # antennas as arms: a shown flag position (Simon says) or a dancer's arms being copied
        arms_on = self.arms is not None and now < self.arms_until and s < 0.5
        self._arms_level += ((1.0 if arms_on else 0.0) - self._arms_level) * min(1.0, dt * 6.0)
        if arms_on:
            want = [-arm_rad(self.arms[0]), arm_rad(self.arms[1])]  # type: ignore[index]
            for i in range(2):
                self._arms_rad[i] += (want[i] - self._arms_rad[i]) * min(1.0, dt * 12.0)
        elif self._arms_level < 0.01:
            self._arms_rad = [ant_l, ant_r]
        if self._arms_level > 0.01:
            ant_l = ant_l * (1 - self._arms_level) + self._arms_rad[0] * self._arms_level
            ant_r = ant_r * (1 - self._arms_level) + self._arms_rad[1] * self._arms_level

        # An ear hold takes an antenna over completely. It used to do so INSTANTLY, and to let go just as
        # instantly when it expired: with a hold going on and off — the handshake's countdown standing
        # aside for each answering gesture and coming back after — the antenna slammed between the top of
        # the head and the bottom of it, over and over, hard enough to startle somebody. So the takeover
        # is eased in and out, and the antenna travels there at a sane rate instead of teleporting.
        for i in (0, 1):
            held, on = self.ear_hold[i], self.ear_hold[i] is not None and now < self.ear_hold_until[i]
            if held is not None and not on:
                self.ear_hold[i] = None
            self._ear_level[i] += ((1.0 if on else 0.0) - self._ear_level[i]) * min(1.0, dt * EAR_HOLD_RATE)
            if on:
                step = EAR_HOLD_SPEED * dt
                self._ear_rad[i] += max(-step, min(step, held - self._ear_rad[i]))
            elif self._ear_level[i] < 0.01:
                self._ear_rad[i] = ant_r if i == 0 else ant_l  # not holding: keep it where the pose is
            k = self._ear_level[i]
            if k > 0.001:
                if i == 0:
                    ant_r = ant_r * (1 - k) + self._ear_rad[0] * k
                else:
                    ant_l = ant_l * (1 - k) + self._ear_rad[1] * k

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

        # holding still for a bracelet: everything fades toward one parked pose and stays there
        if now < self.still_until and self._still < 0.05:
            self._still_yaw = float(self._gaze[0])  # latch it at the moment of freezing: still means still, even
        self._still += ((1.0 if now < self.still_until else 0.0) - self._still) * min(1.0, dt * 2.5)
        if self._still > 0.001:                     # if whoever it is looking at wanders off
            k = self._still
            want_r, want_l = ANTENNA_NEUTRAL
            want_roll = want_yaw = 0.0
            if self.offer_side is not None:
                # The offered antenna stands up just past vertical, tipped toward them so a bracelet slides
                # down it. The head tips TOWARD that side and turns AWAY from it: that is what swings the ear
                # round to face whoever is standing in front, instead of leaving it off to one side.
                other = 1 - self.offer_side
                # ...and the other one leans back out of the way, unless it is wearing one of its own, in
                # which case it stays up where a bracelet is safe and the tilt does the telling instead.
                away = 0.0 if self.loaded[other] else OFFER_AWAY_RAD
                sign = 1.0 if self.offer_side == 0 else -1.0  # right antenna: forward is +; left: forward is -
                want_r = sign * (OFFER_RAD if self.offer_side == 0 else away)
                want_l = sign * (away if self.offer_side == 0 else OFFER_RAD)
                want_roll = -sign * OFFER_ROLL
                want_yaw = sign * OFFER_YAW
            yaw = yaw * (1 - k) + (self._still_yaw + want_yaw) * k
            pitch = pitch * (1 - k) + STILL_PITCH * k
            roll = roll * (1 - k) + want_roll * k
            z *= 1 - k
            ant_r = ant_r * (1 - k) + want_r * k
            ant_l = ant_l * (1 - k) + want_l * k

        # Bracelets ride on the antennas. Upright, they stay put through anything — dancing included —
        # so the gate is simply always on for a loaded antenna rather than a special careful mode. Only
        # the settling right after a trade takes the size out of everything else.
        self._gentle += ((1.0 if now < self.gentle_until else 0.0) - self._gentle) * min(1.0, dt * 1.5)
        worn = max(self._gate)  # how loaded it is, eased: the damping comes and goes with the gate
        damp = max(self._gentle * (1.0 - CAREFUL_SCALE), worn * max(0.0, min(1.0, self.kandi_damp)))
        if holding or self._still > 0.001:
            damp = 0.0  # a pose it is deliberately holding — the trade, a Simon says move — is the point: never shrink it
        if damp > 0.001:
            # Shrink the MOVEMENT, not the looking: everything is pulled back toward where the gaze is
            # pointed, so it still follows the person, it just stops throwing the bracelets around.
            yaw = float(self._gaze[0]) + (yaw - float(self._gaze[0])) * (1 - damp)
            pitch = float(self._gaze[1]) + (pitch - float(self._gaze[1])) * (1 - damp)
            roll *= 1 - damp
            z *= 1 - damp
        for i in (0, 1):
            self._gate[i] += ((1.0 if self.loaded[i] else 0.0) - self._gate[i]) * min(1.0, dt * 1.5)
            if self._gate[i] > 0.995:
                self._gate[i] = 1.0  # snap: once shut, the limit is exactly CAREFUL_ANTENNA_RAD, not almost
            elif self._gate[i] < 0.005:
                self._gate[i] = 0.0
            tilted = max(0.0, min(1.0, (abs(roll) - CAREFUL_ROLL_DEG) / (CAREFUL_ROLL_FULL - CAREFUL_ROLL_DEG)))
            shut = self._gate[i] * tilted
            if shut > 0.001:
                lim = CAREFUL_ANTENNA_RAD + (1.0 - shut) * math.pi  # the clamp closes (and opens) smoothly
                if i == 0:
                    ant_r = max(-lim, min(lim, ant_r))
                else:
                    ant_l = max(-lim, min(lim, ant_l))

        yaw = max(-YAW_LIMIT, min(YAW_LIMIT, yaw))
        pitch = max(-PITCH_LIMIT, min(PITCH_LIMIT, pitch))
        roll = max(-ROLL_LIMIT, min(ROLL_LIMIT, roll))
        z = max(-Z_LIMIT_M, min(Z_LIMIT_M, z))
        # Tilted right over, the head is nearly sitting on the side of the body frame already: dropping it as
        # well is what knocks them together. So the further it is rolled, the less of a drop it is allowed —
        # both limits are fine on their own, it is only the corner where they meet that hits.
        if z < 0.0:
            tilt = abs(roll) / ROLL_LIMIT
            z *= 1.0 - max(0.0, min(1.0, (tilt - TILT_Z_FROM) / (1.0 - TILT_Z_FROM)))

        # Body follows the gaze (not the gesture wobble) when the head is far off-centre, slowly and with a deadband,
        # so the whole robot ends up facing the person and the head has room to move both ways. While grooving the
        # deadband is wide (the body sway must not turn into a slow chase of the bobbing face reading), unless a
        # turn is asked for: then the body goes there, at the far rate, groove or not.
        if self.held:
            self.body_yaw = 0.0
        elif self.body_follow and s < 0.5 and self._still < 0.5:  # frozen for a bracelet: the body does not creep either
            band = BODY_DEADBAND if not grooving else BODY_DEADBAND_GROOVE
            off_body = float(self._gaze[0]) - self.body_yaw
            if abs(off_body) > band:
                rate = BODY_RATE_FAR if abs(off_body) > HEAD_YAW_LIMIT * 0.8 else BODY_RATE
                step = min(abs(off_body) - band * 0.5, rate * dt)
                self.body_yaw += math.copysign(step, off_body)
        self.body_yaw = max(-BODY_YAW_LIMIT, min(BODY_YAW_LIMIT, self.body_yaw))
        body = 0.0 if self.held else max(-BODY_YAW_LIMIT, min(BODY_YAW_LIMIT, self.body_yaw + off.body * (1 - s)))
        # Never ask the head for more than it can do relative to the body.
        wanted = yaw
        yaw = max(body - HEAD_YAW_LIMIT, min(body + HEAD_YAW_LIMIT, yaw))
        self.yaw_short = wanted - yaw
        x = self.forward_shift_m * max(0.0, -pitch) / PITCH_LIMIT + off.x  # looking up: slide forward so the back of the head clears the body
        return head_pose(yaw, pitch, roll, z, x), [ant_r, ant_l], body
