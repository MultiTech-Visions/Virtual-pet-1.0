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
    v.stats = {"detect_ms": 0.0, "embed_ms": 0.0, "body_ms": 0.0, "frames": 0, "faces": 0, "bodies": 0,
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
    v.preview = False
    v._preview(small, faces, None)
    assert v.last_jpeg is None


def test_refine_landmarks_maps_a_close_up_back_into_the_frame():
    import numpy as np

    from festival_pet.vision import refine_landmarks

    class FakeDet:
        def __init__(self):
            self.seen = None
        def detect(self, big):
            self.seen = big.shape
            h, w = big.shape[:2]
            # one face in the middle of the crop, landmarks at known crop pixels
            return np.array([[w * 0.3, h * 0.3, w * 0.4, h * 0.4, w * 0.4, h * 0.45, w * 0.6, h * 0.45, w * 0.5, h * 0.55, w * 0.42, h * 0.65, w * 0.58, h * 0.65, 0.9]], dtype=np.float32)
    small = np.zeros((240, 320, 3), dtype=np.uint8)
    row = np.array([100, 80, 40, 40] + [0] * 10 + [0.9], dtype=np.float32)  # a 40 px face at (100, 80)
    det = FakeDet()
    out = refine_landmarks(det, small, row)
    assert det.seen[0] > 150  # the crop was blown up
    assert np.array_equal(out[:4], row[:4])  # box untouched
    # the crop is a 72 px square centred on the face (120, 100): crop-centre landmarks land near there
    assert abs(out[8] - 120) < 3 and abs(out[9] - 100 - 0.05 * 72) < 3
