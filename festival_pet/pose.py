"""Body pose: where the arms are, from the MediaPipe BlazePose landmark model (OpenCV Zoo ONNX).

The person detector we already run (vision.BodyFinder) is the front half of MediaPipe's pose pipeline:
besides the box it regresses four keypoints, the hip centre and a "full body" point above it, which
give the landmark model its region of interest and its rotation. This is the back half: one 256 px
crop in, 33 body landmarks out (plus six auxiliary ones; 33 and 34 are the same hip / full-body pair,
so a confident pose is next frame's region of interest and the detector need not run again).

Only the arms are read: shoulder, elbow, wrist on each side, as an angle from hanging straight down
(0) through straight out (90) to straight up (180), and a coarse level for the games. Legs and hands
are not looked at.

Pure OpenCV + numpy; ``PoseReader.infer`` is called by the vision thread.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

POSE_INPUT = 256
POSE_CONF_MIN = 0.5  # the model's own "is there a person in this crop" score
PRESENCE_MIN = 0.5  # a landmark counts when the model puts it in the frame with at least this probability
ARM_OUT_DEG, ARM_UP_DEG = 50.0, 130.0  # level boundaries: down below 50, up above 130, out between
# BlazePose landmark indices (its "left" is the PERSON's left, which is image-right for someone facing us)
NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP = 0, 11, 12, 13, 14, 15, 16, 23, 24
AUX_HIP, AUX_FULL = 33, 34  # auxiliary points: hip centre and the full-body point, the next region of interest
ROI_TRACK_SCALE = 1.25  # the region from its own landmarks is grown by this, as MediaPipe does: at 1.0 it shrinks frame by frame


def arm_level(deg: float) -> str:
    return "down" if deg < ARM_OUT_DEG else "up" if deg > ARM_UP_DEG else "out"


def arm_angle(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray, wrist_ok: bool) -> float:
    """Degrees from hanging down (image +y) of the arm's direction: the whole arm when the wrist is in
    the frame, else the upper arm (a raised arm often puts the hand out of the top of the picture)."""
    tip = wrist if wrist_ok else elbow
    v = tip - shoulder
    n = float(np.hypot(v[0], v[1]))
    if n < 1e-6:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, float(v[1]) / n))))


@dataclass
class Arms:
    """What the games and the dance layer read. ``left`` is the PERSON's left arm."""

    ts: float
    left_deg: float
    right_deg: float
    left: str  # "down" | "out" | "up"
    right: str
    conf: float
    shoulder_px: float  # shoulder width in the detection frame: how big (close) they are
    points: dict  # name -> (x, y) in the detection frame, for the preview overlay


WAVE_MIN_DEG = 100.0  # an arm at least this far up can be waving
WAVE_SWINGS = 3  # hand direction reversals...
WAVE_WINDOW_S = 2.0  # ...within this long
WAVE_MIN_SWING = 0.12  # each swing at least this fraction of the shoulder width (side to side, relative to the shoulder)
# The PLUR handshake, done with arms instead of fingers: peace (both arms up in a V, hands apart),
# love (hands together at chest or higher, making a heart), unity (hands clasped low in front), and
# respect (one arm held out to it, offering the bracelet). Finger poses are beyond what the pose
# model gives us at this distance, but these four read clearly from arm angles and where the wrists
# are, and the sequence matters more than any one of them: each step only counts after the one before.
PLUR_STEPS = ("peace", "love", "unity", "respect")
PLUR_HOLD_S = 0.5  # hold a pose this long for it to count
PLUR_STEP_WINDOW_S = 12.0  # ...and get to the next one within this, or the handshake lapses
PEACE_MIN_DEG = 118.0  # arms halfway between out and up (a hug is arms straight out, ~90)
PEACE_MIN_SEP = 1.0  # wrists at least this far apart, in shoulder widths
PEACE_MIN_ABOVE = 0.0  # ...and above the shoulder line, which a hug's arms are not
TOGETHER_SEP = 0.7  # wrists this close count as hands together
HEART_MIN_ABOVE = -0.3  # hands together at chest height or higher (in shoulder widths, + is above the shoulders)
CLASP_MAX_ABOVE = -0.7  # ...and right down in front of them, arms in a V, for the clasp
RESPECT_FOREARM_DEG = 35.0  # the raised fist: forearm within this of straight up
RESPECT_MIN_ABOVE = -0.25  # ...with the fist up around head level
RESPECT_OTHER_DEG = 40.0  # ...and the other arm hanging down
HUG_HOLD_S = 2.5  # both arms held out this long: a hug
HUG_REARM_S = 1.0  # the arms have to leave "out" for this long before another hug counts
SIGN_WATCH_S = 1.5  # a raised or open arm seen this recently keeps the pose model reading every frame


class ArmSigns:
    """What someone is saying with their arms, read from a run of arm readings. Feed every reading; get
    ("wave", side) once per wave (side is the PERSON's hand), ("hug",) once per hug, ("plur", step) as each
    step of the handshake lands, or None.

    A wave is the hand swinging side to side relative to its shoulder, WAVE_SWINGS reversals in WAVE_WINDOW_S,
    with the arm up. A hug is both arms out for HUG_HOLD_S. The PLUR handshake is PLUR_STEPS in order, each
    held PLUR_HOLD_S, the next one within PLUR_STEP_WINDOW_S; while one is under way the wave and hug
    detectors stand down, because a peace sign and a hug are nearly the same shape to a pair of arm angles.
    ``watching`` says the pose model should run every frame right now (a raised or open arm was just seen);
    at the idle rate a wave cannot be told from a stretch.
    """

    def __init__(self) -> None:
        self._swings: dict[str, list[float]] = {"left": [], "right": []}  # times of direction reversals per hand
        self._last_dx: dict[str, float | None] = {"left": None, "right": None}  # last hand offset from its shoulder (px)
        self._turn_dx: dict[str, float | None] = {"left": None, "right": None}  # hand offset at the last reversal
        self._dir: dict[str, int] = {"left": 0, "right": 0}
        self._out_since = 0.0  # both arms out since (0 = not out)
        self._hugged = False  # this hold has already been a hug
        self._out_last = 0.0
        self.watching_until = 0.0
        self.plur_step = 0  # how many of PLUR_STEPS have landed in this handshake
        self.plur_at = 0.0  # when the last one landed
        self._pose: str | None = None  # the pose being held right now, and since when
        self._pose_since = 0.0

    @property
    def watching(self) -> bool:
        return self.watching_until > 0.0

    def feed(self, a: Arms, now: float) -> tuple | None:
        self.watching_until = 0.0
        found = self._plur(a, now)
        if self.plur_step:  # mid-handshake: hold the pose model open, and let the handshake have the arms
            self.watching_until = now + SIGN_WATCH_S
            self._out_since, self._hugged = 0.0, False
            return found
        for hand, deg, ok in (("left", a.left_deg, a.points.get("l_wrist_ok", False)), ("right", a.right_deg, a.points.get("r_wrist_ok", False))):
            if deg < WAVE_MIN_DEG or not ok:
                self._swings[hand].clear()
                self._last_dx[hand], self._turn_dx[hand], self._dir[hand] = None, None, 0
                continue
            self.watching_until = now + SIGN_WATCH_S
            dx = a.points[hand[0] + "_wrist"][0] - a.points[hand[0] + "_shoulder"][0]
            last = self._last_dx[hand]
            self._last_dx[hand] = dx
            if last is None:
                self._turn_dx[hand] = dx
                continue
            step = 1 if dx > last else -1 if dx < last else 0
            if step and step != self._dir[hand]:
                # moving the other way now: the swing that just ended counts if it was big enough
                ref = self._turn_dx[hand]
                if self._dir[hand] and ref is not None and abs(last - ref) >= WAVE_MIN_SWING * a.shoulder_px:
                    self._swings[hand] = [t for t in self._swings[hand] if now - t <= WAVE_WINDOW_S] + [now]
                self._turn_dx[hand] = last
                self._dir[hand] = step
            if len(self._swings[hand]) >= WAVE_SWINGS and found is None:
                self._swings[hand].clear()
                found = ("wave", hand)
        if a.left == "out" and a.right == "out":
            self.watching_until = now + SIGN_WATCH_S
            if self._out_since == 0.0 or now - self._out_last >= HUG_REARM_S:
                self._out_since, self._hugged = now, False  # a fresh hold (a short gap in the readings is the same hold)
            self._out_last = now
            if not self._hugged and now - self._out_since >= HUG_HOLD_S:
                self._hugged = True
                found = ("hug",)
        return found

    def _plur(self, a: Arms, now: float) -> tuple | None:
        """Walk the handshake. Returns ("plur", step) as each one lands, else None."""
        if self.plur_step and now - self.plur_at > PLUR_STEP_WINDOW_S:
            self.plur_step = 0  # they wandered off mid-handshake
        pose = plur_pose(a)
        if pose != self._pose:
            self._pose, self._pose_since = pose, now
            return None
        if pose is None or now - self._pose_since < PLUR_HOLD_S:
            return None
        if self.plur_step:
            self.watching_until = now + SIGN_WATCH_S
        if pose != PLUR_STEPS[self.plur_step]:
            return None  # a pose, but not the one that comes next
        self._pose, self._pose_since = None, now  # this one has landed: wait for a change before the next
        self.plur_step += 1
        self.plur_at = now
        if self.plur_step >= len(PLUR_STEPS):
            self.plur_step = 0
        return ("plur", pose)


def _mid(a, b) -> tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def plur_pose(a: Arms) -> str | None:
    """Which PLUR pose the arms are making, if any. See PLUR_STEPS.

    All four are read from arm angles and where the wrists are, which is what this model gives us
    reliably at across-the-tent distance. Peace is arms up at 45 with the hands apart (a hug is the
    same hands-apart shape but with the arms straight out, so the height of the wrists separates
    them); love is the hands together up at the chest; unity is the same hands, lowered right down
    in front; respect is one arm raised and bent so the forearm stands straight up, fist at head
    height, with the other arm down.
    """
    p = a.points
    scale = max(a.shoulder_px, 1e-6)
    l_deg, r_deg = a.left_deg, a.right_deg
    sh_y = _mid(p["l_shoulder"], p["r_shoulder"])[1]
    if bool(p.get("l_wrist_ok")) and bool(p.get("r_wrist_ok")):
        sep = math.dist(p["l_wrist"], p["r_wrist"]) / scale
        above = (sh_y - _mid(p["l_wrist"], p["r_wrist"])[1]) / scale  # + = hands above the shoulders
        if min(l_deg, r_deg) >= PEACE_MIN_DEG and sep >= PEACE_MIN_SEP and above >= PEACE_MIN_ABOVE:
            return "peace"
        if sep <= TOGETHER_SEP:
            if above >= HEART_MIN_ABOVE:
                return "love"
            if above <= CLASP_MAX_ABOVE:
                return "unity"
    # the raised fist: one forearm standing straight up with the hand about head high, the other arm down
    for side, deg, other in (("l", l_deg, r_deg), ("r", r_deg, l_deg)):
        if other >= RESPECT_OTHER_DEG or not p.get(side + "_wrist_ok"):
            continue
        elbow, wrist = p[side + "_elbow"], p[side + "_wrist"]
        rise, run = elbow[1] - wrist[1], wrist[0] - elbow[0]  # image y grows downward, so rise > 0 is upward
        if rise <= 0:
            continue
        if math.degrees(math.atan2(abs(run), rise)) <= RESPECT_FOREARM_DEG and (sh_y - wrist[1]) / scale >= RESPECT_MIN_ABOVE:
            return "respect"
    return None


class PoseReader:
    """MediaPipe pose landmarker (OpenCV Zoo, Apache-2), fed a region of interest from the person
    detector's keypoints or from its own last confident pose."""

    def __init__(self, model_path: Path) -> None:
        self._net = cv2.dnn.readNet(str(model_path))
        self.roi: tuple[np.ndarray, np.ndarray] | None = None  # (hip centre, full-body point) of the last confident pose
        self.roi_at = 0.0

    def infer(self, frame_bgr: np.ndarray, hip: np.ndarray, full: np.ndarray) -> tuple[np.ndarray, float]:
        """Run the landmarker on the body around ``hip`` (its region reaches ``full``, the point the
        detector puts above the head). Returns (39 x 5 landmarks in frame pixels: x, y, z, visibility,
        presence; confidence). The crop is turned so the hip-to-head line is vertical, as the model
        expects, and the landmarks are turned back."""
        hip = np.asarray(hip, dtype=np.float64)
        full = np.asarray(full, dtype=np.float64)
        dist = float(np.linalg.norm(full - hip))
        if dist < 4.0:
            raise ValueError("person keypoints too close together for a pose crop")
        # the rotation that stands the body up (OpenCV Zoo mp_pose._preprocess), as a single crop-rotate-scale affine
        radians = math.pi / 2 - math.atan2(-(full[1] - hip[1]), full[0] - hip[0])
        radians -= 2 * math.pi * math.floor((radians + math.pi) / (2 * math.pi))
        M = cv2.getRotationMatrix2D((float(hip[0]), float(hip[1])), math.degrees(radians), POSE_INPUT / (2.0 * dist))
        M[:, 2] += POSE_INPUT / 2.0 - hip  # the hip lands in the middle of the crop
        crop = cv2.warpAffine(frame_bgr, M, (POSE_INPUT, POSE_INPUT), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        blob = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        self._net.setInput(blob[np.newaxis])  # NHWC, as the converted model wants it
        out = self._net.forward(self._net.getUnconnectedOutLayersNames())
        by_size = {o.size: o for o in out}
        landmarks = by_size[195].reshape(39, 5).astype(np.float64)
        conf = float(by_size[1].ravel()[0])
        landmarks[:, 3:] = 1.0 / (1.0 + np.exp(-landmarks[:, 3:]))
        inv = cv2.invertAffineTransform(M)
        landmarks[:, :2] = landmarks[:, :2] @ inv[:, :2].T + inv[:, 2]
        return landmarks, conf

    def read_arms(self, frame_bgr: np.ndarray, hip: np.ndarray, full: np.ndarray, now: float) -> Arms | None:
        """One pose step: infer, remember the region for next time if confident, and read the arms.
        None when there is no confident person or the shoulders / elbows are not in the picture."""
        lm, conf = self.infer(frame_bgr, hip, full)
        if conf < POSE_CONF_MIN:
            self.roi = None
            return None
        hip_next = lm[AUX_HIP, :2].copy()
        self.roi, self.roi_at = (hip_next, hip_next + (lm[AUX_FULL, :2] - hip_next) * ROI_TRACK_SCALE), now
        present = lm[:, 4]
        if min(present[L_SHOULDER], present[R_SHOULDER], present[L_ELBOW], present[R_ELBOW]) < PRESENCE_MIN:
            return None
        pts = {name: lm[i, :2] for name, i in (("nose", NOSE), ("l_shoulder", L_SHOULDER), ("r_shoulder", R_SHOULDER), ("l_elbow", L_ELBOW),
                                              ("r_elbow", R_ELBOW), ("l_wrist", L_WRIST), ("r_wrist", R_WRIST))}
        left = arm_angle(pts["l_shoulder"], pts["l_elbow"], pts["l_wrist"], present[L_WRIST] >= PRESENCE_MIN)
        right = arm_angle(pts["r_shoulder"], pts["r_elbow"], pts["r_wrist"], present[R_WRIST] >= PRESENCE_MIN)
        shoulder_px = float(np.linalg.norm(pts["l_shoulder"] - pts["r_shoulder"]))
        points = {k: (float(v[0]), float(v[1])) for k, v in pts.items()}
        points["l_wrist_ok"] = bool(present[L_WRIST] >= PRESENCE_MIN)  # type: ignore[assignment]
        points["r_wrist_ok"] = bool(present[R_WRIST] >= PRESENCE_MIN)  # type: ignore[assignment]
        return Arms(now, left, right, arm_level(left), arm_level(right), conf, shoulder_px, points)
