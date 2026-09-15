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
LOST_AFTER = 0.7  # seconds without a detection before the track is dropped
ASSOC_MAX_NORM = 0.35  # max normalised centre jump to keep associating a track
UNKNOWN_VOTES_TO_ENROLL = 3  # consecutive "no match" embeddings before we create a new person


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


class FaceRecognizer:
    """Thin wrapper around cv2.FaceRecognizerSF + FaceMemory."""

    def __init__(self, sface_model: Path, memory: FaceMemory) -> None:
        self.memory = memory
        self._sf = cv2.FaceRecognizerSF.create(str(sface_model), "")

    def embed(self, frame_bgr: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        aligned = self._sf.alignCrop(frame_bgr, face_row)
        feat = self._sf.feature(aligned)
        return np.asarray(feat, dtype=np.float32).ravel()


class FaceDetector:
    """cv2.FaceDetectorYN with a fixed input size."""

    def __init__(self, yunet_model: Path, score_threshold: float = 0.7) -> None:
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
    ) -> None:
        self.detector = FaceDetector(yunet_model)
        self.recognizer = FaceRecognizer(sface_model, memory) if recognise else None
        self.memory = memory
        self._get_frame = get_frame
        self._get_head_pose = get_head_pose
        self._lock = threading.Lock()
        self._sighting: Sighting | None = None
        self._track: Track | None = None
        self._next_track_id = 1
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.stats = {"detect_ms": 0.0, "embed_ms": 0.0, "frames": 0, "faces": 0}

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

    # ------------------------------------------------------------------ core
    def process_frame(self, frame_bgr: np.ndarray, head_pose: np.ndarray, now: float) -> Sighting | None:
        """One perception step; public so tests can drive it with synthetic frames."""
        H, W = frame_bgr.shape[:2]
        scale = DETECT_WIDTH / W
        small = cv2.resize(frame_bgr, (DETECT_WIDTH, max(2, int(round(H * scale)) // 2 * 2)), interpolation=cv2.INTER_AREA)
        sh, sw = small.shape[:2]

        t0 = time.perf_counter()
        faces = self.detector.detect(small)
        self.stats["detect_ms"] = (time.perf_counter() - t0) * 1000.0
        self.stats["frames"] += 1
        self.stats["faces"] = int(len(faces))

        chosen = self._select(faces, sw, sh, now)
        if chosen is None:
            if self._track is not None and now - self._track.last_seen > LOST_AFTER:
                self._track = None
            return None

        row, track = chosen
        if self.recognizer is not None and self._embedding_due(track, row, now):
            t0 = time.perf_counter()
            self._identify(small, row, track, now)
            self.stats["embed_ms"] = (time.perf_counter() - t0) * 1000.0

        x, y, w, h = row[:4]
        u = (x + w / 2) / scale
        v = (y + h * 0.45) / scale  # aim a little above bbox centre: between the eyes
        return Sighting(track.track_id, float(u), float(v), track.area_frac, track.person, track.similarity, head_pose, now)

    def _select(self, faces: np.ndarray, sw: int, sh: int, now: float) -> tuple[np.ndarray, Track] | None:
        if len(faces) == 0:
            return None
        centres = np.stack([(faces[:, 0] + faces[:, 2] / 2) / sw * 2 - 1, (faces[:, 1] + faces[:, 3] / 2) / sh * 2 - 1], axis=1)
        areas = faces[:, 2] * faces[:, 3] / float(sw * sh)
        track = self._track
        if track is not None and now - track.last_seen <= LOST_AFTER:
            d = np.linalg.norm(centres - np.array([track.cx, track.cy]), axis=1)
            i = int(np.argmin(d))
            if d[i] > ASSOC_MAX_NORM:
                return None  # someone else; keep waiting for our target briefly
        else:
            i = int(np.argmax(areas))  # new lock: the biggest (closest) face wins
            track = Track(self._next_track_id, 0.0, 0.0, 0.0, now)
            self._next_track_id += 1
            self._track = track
        track.cx, track.cy = float(centres[i, 0]), float(centres[i, 1])
        track.area_frac = float(areas[i])
        track.last_seen = now
        return faces[i], track

    @staticmethod
    def _embedding_due(track: Track, row: np.ndarray, now: float) -> bool:
        if min(row[2], row[3]) < EMBED_MIN_FACE_PX:
            return False
        if track.person is None and track.unknown_votes < UNKNOWN_VOTES_TO_ENROLL:
            return now - track.last_embed > 0.5
        return now - track.last_embed > EMBED_RECHECK

    def _identify(self, small: np.ndarray, row: np.ndarray, track: Track, now: float) -> None:
        assert self.recognizer is not None
        track.last_embed = now
        emb = self.recognizer.embed(small, row)
        person, sim = self.memory.match(emb)
        if person is not None:
            track.person, track.similarity = person, sim
            track.unknown_votes = 0
            track.pending_embeddings.clear()
            self.memory.reinforce(person, emb, sim)
            return
        track.similarity = sim
        if track.person is not None:
            # A known track failed a re-check once: keep the identity, it is probably a bad angle.
            return
        track.unknown_votes += 1
        track.pending_embeddings.append(emb)
        if track.unknown_votes >= UNKNOWN_VOTES_TO_ENROLL:
            person = self.memory.enroll(track.pending_embeddings[0], now)
            for extra in track.pending_embeddings[1:]:
                self.memory.reinforce(person, extra, 0.0)
            track.person = person
            track.pending_embeddings.clear()
            logger.info("Enrolled new person #%d", person.person_id)

    def _run(self) -> None:
        next_at = 0.0
        while not self._stop.is_set():
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
            self.memory.save()
