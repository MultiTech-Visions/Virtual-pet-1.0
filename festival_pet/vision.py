"""Camera perception: detect faces (YuNet), track one target, recognise it (SFace).

Runs in its own thread so the 50 Hz control loop never waits on inference.

CPU budget on the CM4 inside the wireless unit (numbers reported by other
projects on the same hardware): YuNet at 320 px ~25-30 ms, one SFace
embedding ~250-350 ms. So detection runs continuously at a modest rate and
embedding only happens when a new track appears or when a periodic re-check
is due. Both models are the OpenCV Zoo ONNX files; download them once with
scripts/setup_offline.sh.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from festival_pet.memory import FaceMemory, Person

logger = logging.getLogger(__name__)

DETECT_WIDTH = 320
DETECT_INTERVAL = 0.12  # ~8 detections per second
EMBED_RECHECK = 4.0  # seconds between re-identifications of a locked track
EMBED_MIN_FACE_PX = 44  # skip embedding tiny faces (in detect-resolution pixels)
LOST_AFTER = 0.7  # seconds without a detection before we stop reporting the track
FORGET_AFTER = 4.0  # seconds before a returning face nearby counts as a new track (peekaboo keeps identity)
ASSOC_MAX_NORM = 0.35  # max normalised centre jump to keep associating a track
UNKNOWN_VOTES_TO_ENROLL = 3  # consecutive "no match" embeddings before we create a new person
BODY_INTERVAL = 0.4  # person detection only when no face is visible, ~2.5x per second


@dataclass
class Track:
    track_id: int
    cx: float  # normalised [-1, 1]
    cy: float
    area_frac: float
    last_seen: float
    person: Person | None = None
    similarity: float = 0.0
    last_embed: float = 0.0
    unknown_votes: int = 0
    pending_embeddings: list[np.ndarray] = field(default_factory=list)
    pending_crops: list[np.ndarray] = field(default_factory=list)


def _blazepose_anchors() -> np.ndarray:
    """Anchor centres (normalised x, y) for the MediaPipe person detector at 224x224.

    Two anchors per cell on the 28x28 and 14x14 grids, six per cell on the 7x7
    grid: 1568 + 392 + 294 = 2254 rows, in model output order (verified against
    the OpenCV Zoo reference list).
    """
    rows = []
    for cells, per in ((28, 2), (14, 2), (7, 6)):
        for y in range(cells):
            for x in range(cells):
                rows += [((x + 0.5) / cells, (y + 0.5) / cells)] * per
    return np.array(rows, dtype=np.float32)


class BodyFinder:
    """MediaPipe person detector (OpenCV Zoo, Apache-2): finds a torso so the head can look up for the face."""

    INPUT = 224

    def __init__(self, model_path: Path, score_threshold: float = 0.35) -> None:  # this model's scores peak ~0.5
        self._net = cv2.dnn.readNet(str(model_path))
        self._anchors = _blazepose_anchors()
        self._score = score_threshold

    def detect(self, frame_bgr: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        """Return person boxes (x1, y1, x2, y2, score) in the frame's pixel coordinates."""
        h, w = frame_bgr.shape[:2]
        scale = max(h, w)
        ratio = self.INPUT / scale
        rw, rh = max(1, int(w * ratio)), max(1, int(h * ratio))
        img = cv2.cvtColor(cv2.resize(frame_bgr, (rw, rh)), cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        pad_l, pad_t = (self.INPUT - rw) // 2, (self.INPUT - rh) // 2
        canvas = np.zeros((self.INPUT, self.INPUT, 3), dtype=np.float32)
        canvas[pad_t : pad_t + rh, pad_l : pad_l + rw] = img
        self._net.setInput(canvas.transpose(2, 0, 1)[np.newaxis])
        out = self._net.forward(self._net.getUnconnectedOutLayersNames())
        regs, logits = (out[0], out[1]) if out[0].shape[-1] > out[1].shape[-1] else (out[1], out[0])
        score = 1.0 / (1.0 + np.exp(-np.clip(logits[0, :, 0].astype(np.float64), -100, 100)))
        keep = np.nonzero(score >= self._score)[0]
        if keep.size == 0:
            return []
        box = regs[0, keep, :4] / self.INPUT
        cxy = box[:, :2] + self._anchors[keep]
        wh = box[:, 2:]
        xy1 = (cxy - wh / 2) * scale - [pad_l / ratio, pad_t / ratio]
        xy2 = (cxy + wh / 2) * scale - [pad_l / ratio, pad_t / ratio]
        boxes = np.concatenate([xy1, xy2], axis=1)
        idx = cv2.dnn.NMSBoxes([(float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])) for b in boxes], score[keep].astype(np.float32), self._score, 0.3)
        return [(*map(float, boxes[i]), float(score[keep][i])) for i in np.asarray(idx).ravel()]


@dataclass
class Sighting:
    """What the behavior layer consumes each tick."""

    track_id: int
    u: float  # pixel coords in the FULL frame
    v: float
    area_frac: float
    person: Person | None
    similarity: float
    head_pose_at_capture: np.ndarray  # 4x4, so gaze math is not skewed by stale poses
    ts: float  # wall-clock time of the frame
    roll_deg: float  # head tilt of the person (from the eye line), + = their head tilts to their left
    kind: str = "face"  # "face", or "body" when only a torso was found and (u, v) is where the head should be
    head_yaw_deg: float = 0.0  # where the person's head is turned, + = toward image-left (rough, from landmarks)
    head_pitch_deg: float = 0.0  # + = looking down (rough)
    cx: float = 0.0  # normalised centre in [-1, 1], for rhythm detection
    cy: float = 0.0
    smile: float = 0.0  # mouth width / eye distance (see smile_from_landmarks)


def head_pose_from_landmarks(row: np.ndarray) -> tuple[float, float, float]:
    """Rough (yaw, pitch, roll) in degrees of the person's head from YuNet's 5 landmarks.

    Row: x, y, w, h, right_eye(x,y), left_eye(x,y), nose(x,y), mouth_right(x,y), mouth_left(x,y), score.
    yaw   + = their nose moved toward image-left of the eye midpoint (they turned toward image-left).
    pitch + = looking down (nose sits low between eyes and mouth).
    roll  + = their head top leans toward image-left.
    Good enough for mirroring games; not a measurement.
    """
    rex, rey, lex, ley, nx, ny, mrx, mry, mlx, mly = (float(v) for v in row[4:14])
    eye_mid = ((rex + lex) / 2, (rey + ley) / 2)
    mouth_mid = ((mrx + mlx) / 2, (mry + mly) / 2)
    eye_dist = max(1.0, math.hypot(lex - rex, ley - rey))
    face_h = max(1.0, mouth_mid[1] - eye_mid[1])
    roll = math.degrees(math.atan2(ley - rey, lex - rex))
    yaw = -60.0 * (nx - eye_mid[0]) / eye_dist  # nose left of the eye midpoint = turned toward image-left
    ratio = (ny - eye_mid[1]) / face_h  # ~0.55 frontal
    pitch = 120.0 * (ratio - 0.55)
    return max(-45.0, min(45.0, yaw)), max(-30.0, min(30.0, pitch)), roll


def smile_from_landmarks(row: np.ndarray) -> float:
    """Mouth width over eye distance: ~0.6-0.7 neutral, 0.8+ a real smile (rough; landmarks, not a model)."""
    rex, rey, lex, ley, nx, ny, mrx, mry, mlx, mly = (float(v) for v in row[4:14])
    eye_dist = max(1.0, math.hypot(lex - rex, ley - rey))
    return math.hypot(mlx - mrx, mly - mry) / eye_dist


SMILE_RATIO = 0.8


def _jpeg(crop_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return bytes(buf)


class FaceRecognizer:
    """Thin wrapper around cv2.FaceRecognizerSF + FaceMemory."""

    def __init__(self, sface_model: Path, memory: FaceMemory) -> None:
        self.memory = memory
        self._sf = cv2.FaceRecognizerSF.create(str(sface_model), "")

    def embed(self, frame_bgr: np.ndarray, face_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (128-d embedding, the 112x112 aligned BGR crop it came from)."""
        aligned = self._sf.alignCrop(frame_bgr, face_row)
        feat = self._sf.feature(aligned)
        return np.asarray(feat, dtype=np.float32).ravel(), aligned


class FaceDetector:
    """cv2.FaceDetectorYN with a fixed input size."""

    def __init__(self, yunet_model: Path, score_threshold: float = 0.6) -> None:  # SDK uses 0.6; 0.7 dropped out too often
        self._det = cv2.FaceDetectorYN.create(
            str(yunet_model), "", (DETECT_WIDTH, DETECT_WIDTH), score_threshold=score_threshold, nms_threshold=0.3, top_k=50
        )
        self._size = (0, 0)

    def detect(self, small_bgr: np.ndarray) -> np.ndarray:
        h, w = small_bgr.shape[:2]
        if (w, h) != self._size:
            self._det.setInputSize((w, h))
            self._size = (w, h)
        result = self._det.detect(small_bgr)
        faces = result[1] if isinstance(result, tuple) else result
        if faces is None:
            return np.zeros((0, 15), dtype=np.float32)
        return np.asarray(faces, dtype=np.float32)


class Vision:
    """Background perception thread. ``latest()`` returns the current sighting or None."""

    def __init__(
        self,
        yunet_model: Path,
        sface_model: Path,
        memory: FaceMemory,
        get_frame: Callable[[], np.ndarray | None],
        get_head_pose: Callable[[], np.ndarray],
        recognise: bool = True,
        person_model: Path | None = None,
    ) -> None:
        self.detector = FaceDetector(yunet_model)
        self.recognizer = FaceRecognizer(sface_model, memory) if recognise else None
        self.body = BodyFinder(person_model) if person_model is not None else None
        self.body_enabled = True
        self._active = threading.Event()
        self._active.set()
        self._last_body_check = 0.0
        self.memory = memory
        self._get_frame = get_frame
        self._get_head_pose = get_head_pose
        self._lock = threading.Lock()
        self._sighting: Sighting | None = None
        self._track: Track | None = None
        self._next_track_id = 1
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.stats = {"detect_ms": 0.0, "embed_ms": 0.0, "body_ms": 0.0, "frames": 0, "faces": 0, "bodies": 0}

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="festival-pet-vision", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def latest(self) -> Sighting | None:
        with self._lock:
            return self._sighting

    preview: bool = False  # keep a JPEG of each processed (320 px) frame for the page
    last_jpeg: bytes | None = None
    capture_now: bool = False  # page asked for one more embedding of the current person

    def set_active(self, active: bool) -> None:
        """Pause (camera idle, no CPU) or resume detection."""
        if active:
            self._active.set()
        else:
            self._active.clear()
            with self._lock:
                self._sighting = None

    # ------------------------------------------------------------------ core
    def process_frame(self, frame_bgr: np.ndarray, head_pose: np.ndarray, now: float) -> Sighting | None:
        """One perception step; public so tests can drive it with synthetic frames."""
        H, W = frame_bgr.shape[:2]
        scale = DETECT_WIDTH / W
        small = cv2.resize(frame_bgr, (DETECT_WIDTH, max(2, int(round(H * scale)) // 2 * 2)), interpolation=cv2.INTER_AREA)
        sh, sw = small.shape[:2]
        if self.preview:
            self.last_jpeg = _jpeg(small)
        else:
            self.last_jpeg = None

        t0 = time.perf_counter()
        faces = self.detector.detect(small)
        self.stats["detect_ms"] = (time.perf_counter() - t0) * 1000.0
        self.stats["frames"] += 1
        self.stats["faces"] = int(len(faces))

        chosen = self._select(faces, sw, sh, now)
        if chosen is None:
            if self._track is not None and now - self._track.last_seen > FORGET_AFTER:
                self._track = None
            return self._body_fallback(small, scale, head_pose, now)

        row, track = chosen
        if self.recognizer is not None and self._embedding_due(track, row, now):
            t0 = time.perf_counter()
            self._identify(small, row, track, now)
            self.stats["embed_ms"] = (time.perf_counter() - t0) * 1000.0

        x, y, w, h = row[:4]
        u = (x + w / 2) / scale
        v = (y + h * 0.45) / scale  # aim a little above bbox centre: between the eyes
        yaw_p, pitch_p, roll = head_pose_from_landmarks(row)
        return Sighting(track.track_id, float(u), float(v), track.area_frac, track.person, track.similarity, head_pose, now, roll,
                        "face", yaw_p, pitch_p, track.cx, track.cy, smile_from_landmarks(row))

    def _body_fallback(self, small: np.ndarray, scale: float, head_pose: np.ndarray, now: float) -> Sighting | None:
        """No face: look for a torso (cheaper rate) and report where the head should be, above it."""
        if self.body is None or not self.body_enabled or now - self._last_body_check < BODY_INTERVAL:
            return None
        self._last_body_check = now
        t0 = time.perf_counter()
        boxes = self.body.detect(small)
        self.stats["body_ms"] = (time.perf_counter() - t0) * 1000.0
        self.stats["bodies"] = len(boxes)
        if not boxes:
            return None
        sh, sw = small.shape[:2]
        x1, y1, x2, y2, score = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        # The head sits above the torso box: aim a bit above its top edge, clamped to the frame.
        u = (x1 + x2) / 2 / scale
        v = max(1.0, (y1 - 0.15 * (y2 - y1))) / scale
        area = (x2 - x1) * (y2 - y1) / float(sw * sh)
        return Sighting(-1, float(u), float(v), float(area), None, 0.0, head_pose, now, 0.0, "body",
                        0.0, 0.0, (x1 + x2) / sw - 1.0, (y1 + y2) / sh - 1.0)

    def _select(self, faces: np.ndarray, sw: int, sh: int, now: float) -> tuple[np.ndarray, Track] | None:
        if len(faces) == 0:
            return None
        centres = np.stack([(faces[:, 0] + faces[:, 2] / 2) / sw * 2 - 1, (faces[:, 1] + faces[:, 3] / 2) / sh * 2 - 1], axis=1)
        areas = faces[:, 2] * faces[:, 3] / float(sw * sh)
        track = self._track
        if track is not None and now - track.last_seen <= FORGET_AFTER:
            d = np.linalg.norm(centres - np.array([track.cx, track.cy]), axis=1)
            i = int(np.argmin(d))
            recently = now - track.last_seen <= LOST_AFTER
            if d[i] > (ASSOC_MAX_NORM if recently else 1.6 * ASSOC_MAX_NORM):
                if recently:
                    return None  # someone else; keep waiting for our target briefly
                track = None  # they left and someone else showed up elsewhere
        else:
            track = None  # no track, or one too stale to associate with: lock afresh below
        if track is None:
            i = int(np.argmax(areas))  # new lock: the biggest (closest) face wins
            track = Track(self._next_track_id, 0.0, 0.0, 0.0, now)
            self._next_track_id += 1
            self._track = track
        track.cx, track.cy = float(centres[i, 0]), float(centres[i, 1])
        track.area_frac = float(areas[i])
        track.last_seen = now
        return faces[i], track

    def _embedding_due(self, track: Track, row: np.ndarray, now: float) -> bool:
        if min(row[2], row[3]) < EMBED_MIN_FACE_PX:
            return False
        if self.capture_now:
            return True
        if track.person is None and track.unknown_votes < UNKNOWN_VOTES_TO_ENROLL:
            return now - track.last_embed > 0.5
        return now - track.last_embed > EMBED_RECHECK

    def _identify(self, small: np.ndarray, row: np.ndarray, track: Track, now: float) -> None:
        assert self.recognizer is not None
        track.last_embed = now
        emb, crop = self.recognizer.embed(small, row)
        forced, self.capture_now = self.capture_now, False
        person, sim = self.memory.match(emb)
        if person is None and forced and track.person is not None:
            person = track.person  # asked for another view of who we are already with: take it even if it matched badly
        if person is not None:
            track.person, track.similarity = person, sim
            track.unknown_votes = 0
            track.pending_embeddings.clear()
            track.pending_crops.clear()
            self.memory.reinforce(person, emb, 0.0 if forced else sim)  # forced: keep it whatever the similarity
            if forced:
                logger.info("captured another view of person #%d (similarity %.2f)", person.person_id, sim)
            if not self.memory.thumbnail_path(person.person_id).exists():
                self.memory.set_thumbnail(person, _jpeg(crop))
            return
        track.similarity = sim
        if track.person is not None:
            # A known track failed a re-check once: keep the identity, it is probably a bad angle.
            return
        track.unknown_votes += 1
        track.pending_embeddings.append(emb)
        track.pending_crops.append(crop)
        if track.unknown_votes >= UNKNOWN_VOTES_TO_ENROLL:
            person = self.memory.enroll(track.pending_embeddings[0], now)
            for extra in track.pending_embeddings[1:]:
                self.memory.reinforce(person, extra, 0.0)
            self.memory.set_thumbnail(person, _jpeg(track.pending_crops[-1]))
            track.person = person
            track.pending_embeddings.clear()
            track.pending_crops.clear()
            logger.info("Enrolled new person #%d", person.person_id)

    def _run(self) -> None:
        next_at = 0.0
        while not self._stop.is_set():
            if not self._active.is_set():
                self._active.wait(0.2)
                continue
            now = time.monotonic()
            if now < next_at:
                time.sleep(min(0.02, next_at - now))
                continue
            next_at = now + DETECT_INTERVAL
            frame = self._get_frame()
            if frame is None:
                continue
            head_pose = self._get_head_pose()
            try:
                sighting = self.process_frame(frame, head_pose, time.time())
            except Exception:
                logger.exception("vision step failed")
                continue
            with self._lock:
                self._sighting = sighting
