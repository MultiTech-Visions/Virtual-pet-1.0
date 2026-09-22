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

    pts = {"l_shoulder": (100.0, 100.0), "r_shoulder": (100.0 + shoulder, 100.0),
           "l_elbow": (100.0, 125.0), "r_elbow": (100.0 + shoulder, 125.0),
           "l_wrist": (100.0 + l_wrist_dx, 40.0), "r_wrist": (100.0 + shoulder + r_wrist_dx, 40.0),
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


def _plur_arms(ts, l_deg, r_deg, l_wrist, r_wrist, l_elbow=None, r_elbow=None, shoulder=50.0, shoulder_y=100.0):
    from festival_pet.pose import Arms

    pts = {"l_shoulder": (100.0, shoulder_y), "r_shoulder": (100.0 + shoulder, shoulder_y),
           "l_elbow": l_elbow or (100.0, shoulder_y + 20), "r_elbow": r_elbow or (100.0 + shoulder, shoulder_y + 20),
           "l_wrist": l_wrist, "r_wrist": r_wrist, "l_wrist_ok": True, "r_wrist_ok": True}
    return Arms(ts, l_deg, r_deg, arm_level(l_deg), arm_level(r_deg), 0.9, shoulder, pts)


PLUR_POSES = {
    # arms up at 45 with the hands apart and above the shoulders
    "peace": lambda t: _plur_arms(t, 135.0, 135.0, (55.0, 60.0), (195.0, 60.0)),
    # hands together up at the chest
    "love": lambda t: _plur_arms(t, 100.0, 100.0, (120.0, 95.0), (133.0, 95.0)),
    # arms folded across the chest: each wrist past the midline, onto the other side of the body
    # (the person's left shoulder is at image-RIGHT of the midline for someone facing the robot)
    "unity": lambda t: _plur_arms(t, 55.0, 55.0, (145.0, 118.0), (105.0, 118.0)),
    # right arm up and bent, forearm straight up, fist at head height; left arm down
    "respect": lambda t: _plur_arms(t, 10.0, 120.0, (95.0, 180.0), (165.0, 80.0),
                                    l_elbow=(100.0, 140.0), r_elbow=(163.0, 125.0)),
}
# arms straight out to the sides at shoulder height: a hug, and none of the four
HUG_ARMS = lambda t: _plur_arms(t, 90.0, 90.0, (30.0, 100.0), (220.0, 100.0))  # noqa: E731


def test_plur_poses_are_told_apart_and_only_count_in_order():
    from festival_pet.pose import PLUR_STEPS, ArmSigns, plur_pose

    for name, make in PLUR_POSES.items():
        assert plur_pose(make(0.0)) == name, name
    # arms hanging down, or straight out to the sides for a hug, are none of them
    assert plur_pose(_plur_arms(0.0, 10.0, 10.0, (95.0, 180.0), (155.0, 180.0))) is None
    assert plur_pose(HUG_ARMS(0.0)) is None

    def run(signs, order, t0=0.0, hold=1.2):
        got, t = [], t0
        for name in order:
            end = t + hold
            while t < end:
                r = signs.feed(PLUR_POSES[name](t), t)
                if r is not None:
                    got.append(r)
                t += 0.1
            t += 0.3
        return got, t

    signs = ArmSigns()
    got, t = run(signs, PLUR_STEPS)
    assert got == [("plur", s) for s in PLUR_STEPS]
    assert signs.plur_step == 0  # the handshake is complete, ready for the next one

    # out of order: only the poses that come next count, and a hug is not started mid-handshake
    signs = ArmSigns()
    got, t = run(signs, ("love", "unity", "peace", "love"))
    assert got == [("plur", "peace"), ("plur", "love")]
    # ...and wandering off mid-handshake lets it lapse
    from festival_pet.pose import PLUR_STEP_WINDOW_S

    signs = ArmSigns()
    run(signs, ("peace",))
    assert signs.plur_step == 1
    signs.feed(PLUR_POSES["unity"](100.0 + PLUR_STEP_WINDOW_S), 100.0 + PLUR_STEP_WINDOW_S)
    assert signs.plur_step == 0


def test_a_peace_sign_does_not_read_as_a_hug():
    from festival_pet.pose import HUG_HOLD_S, ArmSigns

    signs = ArmSigns()
    t, got = 0.0, []
    while t < HUG_HOLD_S * 2:  # holding the peace pose for much longer than a hug needs
        r = signs.feed(PLUR_POSES["peace"](t), t)
        if r is not None:
            got.append(r)
        t += 0.1
    assert ("hug",) not in got and ("plur", "peace") in got


def test_a_pose_can_be_shown_to_it_and_is_then_recognised():
    """The built-in rules are a guess at where somebody holds a double peace sign. When the guess is
    wrong for a particular person — arms lower, hands toward the middle — they can show it the pose
    instead, and what it measures is added to what it already knows."""
    from festival_pet.pose import PLUR_TOL, ArmSigns, plur_features, plur_pose

    # arms well below where the rule looks for peace, hands in toward the middle: a shrug, as far as the
    # built-in rule is concerned, and near enough a hug to the hug detector
    mine = lambda t: _plur_arms(t, 58.0, 56.0, (75.0, 118.0), (175.0, 116.0))  # noqa: E731
    assert plur_pose(mine(0.0)) is None

    proto = plur_features(mine(0.0))
    trained = {"peace": proto}
    assert plur_pose(mine(0.0), trained) == "peace"
    # near enough still counts; a different pose does not get swept up in it
    nearly = _plur_arms(0.0, 58.0 + PLUR_TOL["hi_deg"] * 0.5, 56.0, (78.0, 120.0), (172.0, 118.0))
    assert plur_pose(nearly, trained) == "peace"
    assert plur_pose(PLUR_POSES["unity"](0.0), trained) == "unity"  # its own rule, untouched
    assert plur_pose(_plur_arms(0.0, 10.0, 10.0, (95.0, 180.0), (155.0, 180.0)), trained) is None
    # ...and what it already knew still works: training only ever widens
    for name, make in PLUR_POSES.items():
        assert plur_pose(make(0.0), trained) == name, name

    # once trained, it walks the handshake from that pose, and the hug detector stands down for it
    signs = ArmSigns()
    signs.trained = trained
    got, t = [], 0.0
    while t < 1.2:
        r = signs.feed(mine(t), t)
        if r is not None:
            got.append(r)
        t += 0.1
    assert got == [("plur", "peace")] and signs.plur_step == 1


def _shaped(ts, l_deg, r_deg, l_wrist, r_wrist):
    from festival_pet.pose import arm_level
    return _plur_arms(ts, l_deg, r_deg, l_wrist, r_wrist)


def test_teaching_it_a_hug_is_what_tells_a_hug_from_a_peace_sign():
    """Arms out wide and a double peace sign held at 45 are nearly the same shape to a pair of arm
    angles, which is why a handshake kept being read as a cuddle. Teaching it both turns the question
    from 'is this a peace sign?' into 'which of these is it most like?'."""
    from festival_pet.pose import ArmSigns, plur_features, plur_pose, trained_pose

    peace = lambda t=0.0: _shaped(t, 58.0, 56.0, (112.0, 78.0), (140.0, 76.0))  # noqa: E731  this person's peace
    hug = lambda t=0.0: _shaped(t, 92.0, 90.0, (35.0, 100.0), (215.0, 101.0))  # noqa: E731

    assert plur_pose(peace()) != "peace"  # the built-in guess does not fit how they hold it
    signs = ArmSigns()
    for i in range(3):  # a few goes each, with the wobble a real person has
        signs.add_training("peace", plur_features(_shaped(0.0, 58.0 + i * 3, 56.0 - i * 2, (112.0 + i, 78.0 - i), (140.0 - i, 76.0 + i))))
        signs.add_training("hug", plur_features(_shaped(0.0, 92.0 - i * 2, 90.0 + i * 3, (35.0 + i, 100.0), (215.0 - i, 101.0))))
    assert trained_pose(peace(), signs.trained) == "peace"
    assert trained_pose(hug(), signs.trained) == "hug"

    # holding the peace sign is the handshake starting, NOT a hug, however long it is held
    got = [signs.feed(peace(t * 0.1), t * 0.1) for t in range(40)]
    assert [g for g in got if g] == [("plur", "peace")]
    # ...and a hug is still a hug
    other = ArmSigns()
    other.trained = signs.trained
    got = [other.feed(hug(t * 0.1), t * 0.1) for t in range(40)]
    assert [g for g in got if g] == [("hug",)]


def test_a_pose_is_averaged_over_its_last_few_goes_and_one_bad_go_cannot_spoil_it():
    from festival_pet.pose import PLUR_TRAIN_KEEP, ArmSigns, plur_features, prototype, runs_of, trained_pose

    good = lambda i: plur_features(_shaped(0.0, 58.0 + i, 56.0 - i, (112.0, 78.0), (140.0, 76.0)))  # noqa: E731
    signs = ArmSigns()
    for i in range(3):
        assert signs.add_training("peace", good(i)) == i + 1
    clean = prototype(runs_of(signs.trained["peace"]))
    # one go caught mid-move, with the arms nowhere near the pose
    signs.add_training("peace", plur_features(_shaped(0.0, 150.0, 20.0, (10.0, 10.0), (250.0, 190.0))))
    spoilt = prototype(runs_of(signs.trained["peace"]))
    assert abs(spoilt["hi_deg"] - clean["hi_deg"]) < 8.0, "one bad go dragged the whole pose"
    assert trained_pose(_shaped(0.0, 59.0, 55.0, (112.0, 78.0), (140.0, 76.0)), signs.trained) == "peace"

    for i in range(10):  # it only ever keeps the last few
        signs.add_training("peace", good(i))
    assert len(signs.trained["peace"]) == PLUR_TRAIN_KEEP

    # a pose stored the old way, as a single prototype, still loads and still matches
    signs2 = ArmSigns()
    signs2.trained = {"peace": good(0)}
    assert trained_pose(_shaped(0.0, 58.0, 56.0, (112.0, 78.0), (140.0, 76.0)), signs2.trained) == "peace"
    assert signs2.add_training("peace", good(1)) == 2


def test_unity_is_folded_arms_read_off_the_shoulders_not_the_hands():
    """Hands clasped low in front was the worst pose this model could be asked for: wrists are the
    landmarks it is least sure about, low hands sit near the edge of the frame, and 'both arms down with
    the hands near each other' is barely distinguishable from standing still. Folded arms are measured
    against the SHOULDERS, which it is surest about, and nothing else looks like them."""
    from festival_pet.pose import CROSS_MIN, ArmSigns, crossed, plur_features, plur_pose

    folded = lambda t=0.0: _plur_arms(t, 55.0, 55.0, (145.0, 118.0), (105.0, 118.0))  # noqa: E731
    assert plur_pose(folded()) == "unity"

    # the crossing number separates it from everything else by a mile, and only it is positive
    everything = {name: plur_features(make(0.0))["cross"] for name, make in PLUR_POSES.items()}
    everything["hug"] = plur_features(HUG_ARMS(0.0))["cross"]
    everything["resting"] = plur_features(_plur_arms(0.0, 10.0, 10.0, (95.0, 180.0), (155.0, 180.0)))["cross"]
    assert everything["unity"] >= CROSS_MIN
    assert all(v < 0.0 for k, v in everything.items() if k != "unity"), everything
    assert min(everything["unity"] - v for k, v in everything.items() if k != "unity") > 0.4

    # it does not care which way round somebody is standing: the sign comes from their own shoulders
    mirrored = _plur_arms(0.0, 55.0, 55.0, (105.0, 118.0), (145.0, 118.0), shoulder=-50.0)
    assert plur_features(mirrored)["cross"] >= CROSS_MIN

    # and folded arms are not a hug, however long they are held: nobody hugs with their arms folded
    assert crossed(folded()) and not crossed(HUG_ARMS(0.0))
    signs = ArmSigns()
    assert [g for g in (signs.feed(folded(t * 0.1), t * 0.1) for t in range(60)) if g] == []
