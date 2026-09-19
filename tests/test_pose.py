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
