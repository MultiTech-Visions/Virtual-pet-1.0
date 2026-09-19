"""The vision worker must outlive a camera or daemon hiccup, and say what went wrong."""

import threading
import time

from festival_pet.vision import Vision


def _bare_vision(get_frame, process):
    """A Vision without models: only the thread machinery, which is what is under test."""
    v = object.__new__(Vision)
    v._get_frame = get_frame
    v._get_head_pose = lambda: None
    v.process_frame = process
    v._lock = threading.Lock()
    v._sighting = None
    v._active = threading.Event(); v._active.set()
    v._stop = threading.Event()
    v._thread = None
    v.refined = False
    v.pose, v.pose_live, v._arms, v._last_pose_check, v._last_someone = None, False, None, 0.0, 0.0
    v.stats = {"detect_ms": 0.0, "embed_ms": 0.0, "body_ms": 0.0, "pose_ms": 0.0, "frames": 0, "faces": 0, "bodies": 0,
               "last_frame_at": 0.0, "no_frame": 0, "errors": 0, "last_error": ""}
    return v


def test_vision_thread_survives_a_camera_error_and_reports_it():
    calls = {"n": 0}

    def get_frame():
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("camera went away")
        return "frame"

    v = _bare_vision(get_frame, lambda frame, pose, now: "sighting")
    v.start()
    deadline = time.time() + 3.0
    while time.time() < deadline and calls["n"] < 4:
        time.sleep(0.02)
    v.stop()
    assert calls["n"] >= 4  # kept reading after the error
    st = v.status(time.time())
    assert st["errors"] == 1 and st["last_error"] == "RuntimeError: camera went away"
    assert st["frame_age_s"] is not None and st["frame_age_s"] < 2.0
    assert v.latest() == "sighting"


def test_vision_status_says_when_it_is_paused_or_dead():
    v = _bare_vision(lambda: None, lambda frame, pose, now: None)
    st = v.status(time.time())
    assert not st["alive"] and st["active"] and st["frame_age_s"] is None
    v.set_active(False)
    assert not v.status(time.time())["active"]


def test_preview_draws_faces_landmarks_and_the_body_box():
    import cv2
    import numpy as np

    v = _bare_vision(lambda: None, lambda frame, pose, now: None)
    v.preview = True
    v._track = None
    v._last_body_box = (10.0, 40.0, 120.0, 170.0, 0.51)
    small = np.zeros((180, 320, 3), dtype=np.uint8)
    faces = np.array([[40, 30, 60, 70, 55, 50, 85, 50, 70, 65, 58, 80, 82, 80, 0.9]], dtype=np.float32)
    v._preview(small, faces, faces[0])
    assert v.last_jpeg is not None
    img = cv2.imdecode(np.frombuffer(v.last_jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (180, 320, 3)
    assert img[30, 70].sum() > 100  # the face box's top edge was drawn
    assert img[40, 60].sum() > 100  # the body box's top edge too
    from festival_pet.pose import Arms

    v._arms = Arms(0.0, 170.0, 20.0, "up", "down", 0.9, 60.0, {"nose": (200.0, 100.0), "l_shoulder": (230.0, 130.0), "r_shoulder": (170.0, 130.0),
                                                            "l_elbow": (240.0, 100.0), "r_elbow": (160.0, 160.0), "l_wrist": (245.0, 60.0), "r_wrist": (150.0, 175.0),
                                                            "l_wrist_ok": True, "r_wrist_ok": False})
    v._preview(small, faces, faces[0])
    img = cv2.imdecode(np.frombuffer(v.last_jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert img[130, 200].sum() > 100  # the shoulder line
    assert img[80, 242].sum() > 100  # the left forearm (wrist in the frame)
    assert img[168, 155].sum() < 60  # no right forearm drawn: the wrist was out of the frame
    v.preview = False
    v._preview(small, faces, None)
    assert v.last_jpeg is None


def test_arms_are_read_every_frame_when_live_and_twice_a_second_otherwise():
    """The scheduler around the pose model, with the model and the person detector faked."""
    import numpy as np

    from festival_pet.pose import Arms

    class FakePose:
        roi = None
        roi_at = 0.0
        calls: list = []

        def read_arms(self, frame, hip, full, now):
            self.calls.append(now)
            self.roi, self.roi_at = (np.array([10.0, 20.0]), np.array([10.0, 2.0])), now
            return Arms(now, 30.0, 100.0, "down", "out", 0.9, 40.0, {})

    class FakeBody:
        calls = 0

        def detect(self, frame):
            FakeBody.calls += 1
            return [(0.0, 0.0, 50.0, 90.0, 0.6, np.array([[25.0, 60.0], [25.0, 5.0], [25.0, 30.0], [25.0, 10.0]]))]

    v = _bare_vision(lambda: None, lambda frame, pose, now: None)
    v.pose, v.body = FakePose(), FakeBody()
    v._last_body_box = None
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    v._last_someone = 100.0  # someone was just seen
    for i in range(12):  # 8 frames/s idle (1.3 s): only every 0.5 s
        v._read_arms(frame, 100.0 + i * 0.12)
    assert len(v.pose.calls) == 3 and v.latest_arms() is not None and v.latest_arms().right == "out"
    assert FakeBody.calls == 1  # the detector gave the first region; the pose tracked itself after that
    v.pose_live = True
    v.pose.calls.clear()
    for i in range(10):
        v._read_arms(frame, 102.0 + i * 0.12)
    assert len(v.pose.calls) == 10  # every frame
    v.pose_live = False
    v._last_someone = 0.0  # nobody about for ages
    v._read_arms(frame, 110.0)
    assert v.latest_arms() is None and len(v.pose.calls) == 10  # nothing run, the stale read dropped


def test_refine_landmarks_maps_a_close_up_back_into_the_frame():
    import numpy as np

    from festival_pet.vision import refine_landmarks

    class FakeDet:
        def __init__(self):
            self.seen = None
        def detect(self, big):
            self.seen = big.shape
            h, w = big.shape[:2]
            # one face in the middle of the crop, landmarks at known crop pixels, upright already: every rotation scores the same
            return np.array([[w * 0.3, h * 0.3, w * 0.4, h * 0.4, w * 0.4, h * 0.45, w * 0.6, h * 0.45, w * 0.5, h * 0.55, w * 0.42, h * 0.65, w * 0.58, h * 0.65, 0.9]], dtype=np.float32)
    small = np.zeros((240, 320, 3), dtype=np.uint8)
    row = np.array([100, 80, 40, 40] + [0] * 10 + [0.9], dtype=np.float32)  # a 40 px face at (100, 80)
    det = FakeDet()
    out, roll, ok = refine_landmarks(det, small, row)
    assert ok and det.seen[0] > 150  # the crop was blown up
    assert np.array_equal(out[:4], row[:4])  # box untouched
    assert abs(roll) < 1e-6  # flat scores: the parabola sits on the centre sample, no roll
    # the crop is a 72 px square centred on the face (120, 100): crop-centre landmarks land near there
    assert abs(out[8] - 120) < 3 and abs(out[9] - 100 - 0.05 * 72) < 3


def test_rotation_search_finds_the_roll_and_rotates_the_landmarks_back(monkeypatch):
    import numpy as np

    from festival_pet import vision
    from festival_pet.vision import refine_landmarks

    class ScoreByAngle:
        """A detector whose score peaks when the crop has been rotated by +20 (the face was rolled -20)."""
        def detect(self, big):
            return np.zeros((0, 15), dtype=np.float32)
    seen_angles = []

    def fake_detect_rotated(detector, big, angle):
        import cv2
        seen_angles.append(angle)
        h, w = big.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        score = max(0.0, 0.95 - 0.012 * abs(angle - 20.0))
        # the eyes in the ROTATED crop are level; rotated back they should come out tilted
        face = np.array([w * 0.3, h * 0.3, w * 0.4, h * 0.4, w * 0.4, h * 0.45, w * 0.6, h * 0.45, w * 0.5, h * 0.55, w * 0.42, h * 0.65, w * 0.58, h * 0.65, score], dtype=np.float32)
        return face, M
    monkeypatch.setattr(vision, "_detect_rotated", fake_detect_rotated)
    small = np.zeros((240, 320, 3), dtype=np.uint8)
    row = np.array([100, 80, 40, 40] + [0] * 10 + [0.9], dtype=np.float32)
    out, roll, ok = refine_landmarks(ScoreByAngle(), small, row, roll_prev=0.0)
    assert ok and seen_angles == [-15.0, 0.0, 15.0]
    assert roll > 12.0  # peak at +15 with the parabola pulling toward +20: the head is rolled about +17 (clockwise)
    rex, rey, lex, ley = out[4], out[5], out[6], out[7]
    assert ley - rey > 2.0  # the eye line, rotated back, tilts the way the head does (clockwise: image-right eye lower)
    # next frame the search is centred on the last roll and lands on the peak
    seen_angles.clear()
    out2, roll2, ok2 = refine_landmarks(ScoreByAngle(), small, row, roll_prev=roll)
    assert ok2 and abs(roll2 - 20.0) < 2.0
    # nothing found at any angle: the coarse row stands and roll is 0
    monkeypatch.setattr(vision, "_detect_rotated", lambda d, b, a: (None, np.eye(2, 3, dtype=np.float32)))
    out3, roll3, ok3 = refine_landmarks(ScoreByAngle(), small, row)
    assert not ok3 and roll3 == 0.0 and np.array_equal(out3, row)
