"""Arms from the pose landmarker: the geometry around the model, with the model faked."""

import math

import numpy as np

from festival_pet.pose import (ARM_OUT_DEG, ARM_UP_DEG, AUX_FULL, AUX_HIP, L_ELBOW, L_SHOULDER, L_WRIST, POSE_INPUT, R_ELBOW, R_SHOULDER,
                               R_WRIST, PoseReader, arm_angle, arm_level)


def test_arm_angle_and_levels():
    s = np.array([100.0, 100.0])
    assert arm_angle(s, s + [0, 20], s + [0, 40], True) == 0.0  # hanging down
    assert abs(arm_angle(s, s + [20, 0], s + [40, 0], True) - 90.0) < 1e-9  # straight out
    assert abs(arm_angle(s, s + [0, -20], s + [0, -40], True) - 180.0) < 1e-9  # straight up
    assert abs(arm_angle(s, s + [20, -20], s + [40, 40], False) - 135.0) < 1e-9  # wrist out of frame: the upper arm decides
    assert arm_level(10) == "down" and arm_level(ARM_OUT_DEG) == "out" and arm_level(ARM_UP_DEG) == "out" and arm_level(170) == "up"
    assert arm_angle(s, s, s, True) == 0.0  # degenerate: no crash


def _arms_at(ts, l_deg, r_deg, l_wrist_dx=0.0, r_wrist_dx=0.0, shoulder=50.0):
    from festival_pet.pose import Arms

    pts = {"l_shoulder": (100.0, 100.0), "r_shoulder": (100.0 + shoulder, 100.0), "l_wrist": (100.0 + l_wrist_dx, 40.0), "r_wrist": (100.0 + shoulder + r_wrist_dx, 40.0),
           "l_wrist_ok": True, "r_wrist_ok": True}
    return Arms(ts, l_deg, r_deg, arm_level(l_deg), arm_level(r_deg), 0.9, shoulder, pts)


def test_a_wave_is_three_swings_of_a_raised_hand_and_a_hug_is_arms_out_held():
    from festival_pet.pose import HUG_HOLD_S, ArmSigns

    s = ArmSigns()
    assert not s.watching
    # arm down and swinging: nothing (a walk); arm up and still: nothing but it watches
    assert s.feed(_arms_at(0.0, 20.0, 20.0, l_wrist_dx=10.0), 0.0) is None and not s.watching
    assert s.feed(_arms_at(0.1, 160.0, 20.0, l_wrist_dx=0.0), 0.1) is None and s.watching
    # the left hand swings +-15 px about its shoulder (shoulder width 50): a reversal every 0.2 s
    found = []
    t = 0.2
    for k in range(12):
        dx = 15.0 if k % 2 == 0 else -15.0
        found.append(s.feed(_arms_at(t, 160.0, 20.0, l_wrist_dx=dx), t))
        t += 0.2
    waves = [f for f in found if f is not None]
    assert waves and waves[0] == ("wave", "left") and found.index(waves[0]) >= 3  # three full swings before the first (the brain rate-limits the rest)
    # tiny jitters do not count as swings
    s2 = ArmSigns()
    for k in range(12):
        assert s2.feed(_arms_at(k * 0.1, 160.0, 20.0, l_wrist_dx=2.0 if k % 2 else -2.0), k * 0.1) is None
    # a hug: both arms out for HUG_HOLD_S, once; arms must leave "out" for a moment before another counts
    s3 = ArmSigns()
    t = 10.0
    got = []
    while t < 10.0 + HUG_HOLD_S + 1.0:
        got.append(s3.feed(_arms_at(t, 90.0, 90.0), t))
        t += 0.1
    assert got.count(("hug",)) == 1 and got[0] is None and s3.watching
    assert s3.feed(_arms_at(t, 90.0, 90.0), t) is None  # still held: not again
    assert s3.feed(_arms_at(t + 0.1, 10.0, 10.0), t + 0.1) is None  # arms dropped...
    t2 = t + 0.1 + 1.5
    got2 = []
    while t2 < t + 0.1 + 1.5 + HUG_HOLD_S + 0.3:
        got2.append(s3.feed(_arms_at(t2, 90.0, 90.0), t2))
        t2 += 0.1
    assert got2.count(("hug",)) == 1  # ...and back out long enough: a second hug


class FakeNet:
    """Answers like the OpenCV Zoo pose model: landmarks (1, 195) in crop pixels, conf (1, 1), and the rest."""

    def __init__(self, crop_landmarks: np.ndarray, conf: float, presence_logit: float = 5.0):
        self.lm, self.conf, self.pres = crop_landmarks, conf, presence_logit
        self.inputs = []

    def setInput(self, blob):
        self.inputs.append(blob)

    def getUnconnectedOutLayersNames(self):
        return ("a", "b", "c", "d", "e")

    def forward(self, names):
        lm = np.zeros((39, 5), np.float32)
        lm[:, :2] = self.lm
        lm[:, 3:] = self.pres
        return [lm.reshape(1, 195), np.array([[self.conf]], np.float32), np.zeros((1, 256, 256, 1), np.float32), np.zeros((1, 64, 64, 39), np.float32), np.zeros((1, 117), np.float32)]


def _reader(net):
    r = object.__new__(PoseReader)
    r._net, r.roi, r.roi_at = net, None, 0.0
    return r


def test_landmarks_come_back_in_frame_pixels_and_the_next_roi_is_remembered():
    # a person standing upright: hip at (200, 300), full-body point 100 px above -> a 200 px crop around the hip
    crop = np.full((39, 2), POSE_INPUT / 2, np.float64)
    crop[AUX_HIP] = (POSE_INPUT / 2, POSE_INPUT / 2)
    crop[AUX_FULL] = (POSE_INPUT / 2, 0.0)  # top of the crop = 100 px above the hip
    crop[L_SHOULDER], crop[R_SHOULDER] = (POSE_INPUT / 2 + 32, 64.0), (POSE_INPUT / 2 - 32, 64.0)  # 25 px either side, 50 px up
    crop[L_ELBOW], crop[R_ELBOW] = (POSE_INPUT / 2 + 64, 64.0), (POSE_INPUT / 2 - 32, 128.0)  # left out, right down
    crop[L_WRIST], crop[R_WRIST] = (POSE_INPUT / 2 + 96, 64.0), (POSE_INPUT / 2 - 32, 160.0)
    net = FakeNet(crop, 0.9)
    r = _reader(net)
    frame = np.zeros((480, 640, 3), np.uint8)
    lm, conf = r.infer(frame, np.array([200.0, 300.0]), np.array([200.0, 200.0]))
    assert abs(conf - 0.9) < 1e-6
    assert net.inputs[0].shape == (1, POSE_INPUT, POSE_INPUT, 3) and net.inputs[0].dtype == np.float32
    assert np.allclose(lm[AUX_HIP, :2], [200.0, 300.0], atol=1e-6)  # the hip is where we said it was
    assert np.allclose(lm[AUX_FULL, :2], [200.0, 200.0], atol=1e-6)
    assert np.allclose(lm[L_SHOULDER, :2], [225.0, 250.0], atol=1e-6)  # crop pixel = 100/128 frame px
    assert np.allclose(lm[L_WRIST, :2], [275.0, 250.0], atol=1e-6)
    assert abs(lm[L_SHOULDER, 4] - 1 / (1 + math.exp(-5.0))) < 1e-6  # presence went through the sigmoid
    arms = r.read_arms(frame, np.array([200.0, 300.0]), np.array([200.0, 200.0]), 5.0)
    assert arms is not None and arms.left == "out" and arms.right == "down" and abs(arms.left_deg - 90.0) < 1e-6 and arms.right_deg == 0.0
    assert abs(arms.shoulder_px - 50.0) < 1e-6 and abs(arms.conf - 0.9) < 1e-6
    assert r.roi is not None and np.allclose(r.roi[0], [200.0, 300.0]) and r.roi_at == 5.0  # tracks itself next time
    assert np.allclose(r.roi[1], [200.0, 175.0])  # the next region is grown by a quarter (it shrinks otherwise)
    assert arms.points["l_wrist_ok"] and arms.points["r_wrist_ok"]


def test_a_leaning_body_is_stood_up_for_the_model_and_turned_back():
    # hip at (200, 300), full-body point up and to the right: the crop is rotated so that line is vertical
    crop = np.full((39, 2), POSE_INPUT / 2, np.float64)
    crop[AUX_FULL] = (POSE_INPUT / 2, 0.0)
    net = FakeNet(crop, 0.8)
    r = _reader(net)
    hip, full = np.array([200.0, 300.0]), np.array([200.0 + 100 * math.sin(math.radians(30)), 300.0 - 100 * math.cos(math.radians(30))])
    lm, _ = r.infer(np.zeros((480, 640, 3), np.uint8), hip, full)
    assert np.allclose(lm[AUX_HIP, :2], hip, atol=1e-6)
    assert np.allclose(lm[AUX_FULL, :2], full, atol=1e-6)  # the crop's top-centre lands on the full-body point


def test_no_person_or_arms_out_of_the_picture_reads_as_nothing():
    crop = np.full((39, 2), POSE_INPUT / 2, np.float64)
    r = _reader(FakeNet(crop, 0.2))
    frame = np.zeros((480, 640, 3), np.uint8)
    r.roi = (np.array([1.0, 1.0]), np.array([1.0, 50.0]))
    assert r.read_arms(frame, np.array([200.0, 300.0]), np.array([200.0, 200.0]), 1.0) is None
    assert r.roi is None  # a lost person: the detector is asked next time
    r = _reader(FakeNet(crop, 0.9, presence_logit=-3.0))  # confident there is a person, but the shoulders are out of frame
    assert r.read_arms(frame, np.array([200.0, 300.0]), np.array([200.0, 200.0]), 1.0) is None
    assert r.roi is not None  # still tracking them, though
