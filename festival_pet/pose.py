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
HUG_HOLD_S = 2.5  # both arms held out this long: a hug
HUG_REARM_S = 1.0  # the arms have to leave "out" for this long before another hug counts
SIGN_WATCH_S = 1.5  # a raised or open arm seen this recently keeps the pose model reading every frame


class ArmSigns:
    """Waves and hugs, read from a run of arm readings. Feed every reading; get ("wave", side) once per wave
    (side is the PERSON's hand, "left" or "right"), ("hug",) once per hug, or None.

    A wave is the hand swinging side to side relative to its shoulder, WAVE_SWINGS reversals in WAVE_WINDOW_S,
    with the arm up. A hug is both arms out for HUG_HOLD_S. ``watching`` says the pose model should run every
    frame right now (a raised or open arm was just seen); at the idle rate a wave cannot be told from a stretch.
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

    @property
    def watching(self) -> bool:
        return self.watching_until > 0.0

    def feed(self, a: Arms, now: float) -> tuple | None:
        self.watching_until = 0.0
        found: tuple | None = None
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
