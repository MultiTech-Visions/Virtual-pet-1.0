"""Festival Pet: the robot-side glue.

Runs as a standard Reachy Mini app (dashboard-launchable) on the wireless unit.
Everything is offline: local YuNet/SFace models, procedural beeps, a JSON memory
file, and the emotions library from the daemon's HuggingFace cache.

Threads
    main (this)   50 Hz control loop: senses -> behavior -> motion -> set_target
    vision        ~8 Hz detection, occasional SFace embedding
    sound         renders + pushes phrases so a 1 s purr never stalls motion
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from pathlib import Path

# Never let huggingface_hub try the network on the festival ground: cache only.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import platformdirs
from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini.motion.recorded_move import DEFAULT_EMOTIONS_DATASET, RecordedMove, RecordedMoves
from reachy_mini.utils.interpolation import linear_pose_interpolation
from reachy_mini.vision.look_at import look_at_image_pose
from scipy.spatial.transform import Rotation as R

from festival_pet import sounds
from festival_pet.behavior import Action, Behavior, FaceObs, Observation
from festival_pet.memory import FaceMemory
from festival_pet.motion import MotionComposer
from festival_pet.senses import LoudSoundDetector, PickupDetector, TouchDetector
from festival_pet.vision import Sighting, Vision

logger = logging.getLogger("festival_pet")

DATA_DIR = Path(platformdirs.user_data_dir("festival_pet"))
YUNET_MODEL = DATA_DIR / "models" / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = DATA_DIR / "models" / "face_recognition_sface_2021dec.onnx"
MEMORY_FILE = DATA_DIR / "memory.json"

CONTROL_HZ = 50.0
PLAY_LIBRARY_SOUNDS = False  # True = play Pollen's sidecar sound with library moves instead of our beeps
MOVE_BLEND_S = 0.5


class SoundPlayer:
    """Serialises phrases onto the speaker from a worker thread."""

    def __init__(self, reachy: ReachyMini) -> None:
        self._reachy = reachy
        self._q: queue.Queue[tuple[int, str]] = queue.Queue()
        self._busy_until = 0.0
        self._busy_priority = 0
        self._sample_rate = reachy.media.get_output_audio_samplerate()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="festival-pet-sound", daemon=True)

    def start(self) -> None:
        self._reachy.media.start_playing()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._reachy.media.stop_playing()

    def request(self, emotion: str, priority: int, now: float) -> None:
        """Drop the request if something at least as important is still sounding."""
        if now < self._busy_until and self._busy_priority >= priority:
            return
        self._q.put((priority, emotion))

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                priority, emotion = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            buf = sounds.render_phrase(emotion, sample_rate=self._sample_rate)
            dur = sounds.phrase_duration(buf, self._sample_rate)
            self._busy_until = time.time() + dur
            self._busy_priority = priority
            # Push in ~40 ms chunks so playback starts immediately and the buffer never balloons.
            chunk = int(self._sample_rate * 0.04)
            for i in range(0, len(buf), chunk):
                self._reachy.media.push_audio_sample(buf[i : i + chunk])
            time.sleep(dur)


class FestivalPetApp(ReachyMiniApp):
    """A non-verbal, expressive, face-remembering droid."""

    custom_app_url: str | None = "http://0.0.0.0:8042"
    request_media_backend: str | None = None  # we need camera + audio

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
        if not YUNET_MODEL.exists() or not SFACE_MODEL.exists():
            raise FileNotFoundError(f"Face models missing in {DATA_DIR / 'models'}; run scripts/setup_offline.sh while online.")

        memory = FaceMemory(MEMORY_FILE)
        behavior = Behavior(memory)
        composer = MotionComposer()
        pickup = PickupDetector()
        touch = TouchDetector()
        loud = LoudSoundDetector()
        sound = SoundPlayer(reachy_mini)
        library = RecordedMoves(DEFAULT_EMOTIONS_DATASET)
        vision = Vision(YUNET_MODEL, SFACE_MODEL, memory, reachy_mini.media.get_frame, reachy_mini.get_current_head_pose)

        camera = reachy_mini.media.camera
        if camera is None or camera.K is None or camera.D is None:
            raise RuntimeError("Camera intrinsics unavailable; the app needs the local media backend on the robot.")
        K, D = camera.K, camera.D
        T_head_cam = reachy_mini.T_head_cam

        self._install_status_routes(behavior, memory, vision)

        sound.start()
        vision.start()
        now = time.time()
        behavior.start(now, awake=True)  # the platform wakes the robot before launching an app
        sound.request("wake", 5, now)
        composer.request_gesture("perk", now, 5)

        # Library move playback state
        move: RecordedMove | None = None
        move_t0 = 0.0
        last_pose = np.eye(4)
        last_ants = [0.0, 0.0]
        blend_from: tuple[np.ndarray, list[float], float] | None = None

        period = 1.0 / CONTROL_HZ
        last = time.time()
        next_doa = 0.0
        try:
            while not stop_event.is_set():
                now = time.time()
                dt = max(1e-3, now - last)
                last = now

                # ---------------- senses
                obs = Observation()
                imu = reachy_mini.imu
                if imu is not None:
                    obs.held, obs.shaken = pickup.update(imu["accelerometer"], imu["gyroscope"], now)
                present_ants = reachy_mini.get_present_antenna_joint_positions()
                # Antennas lag their command during fast gestures/moves; only trust displacement when calm.
                calm = move is None and not composer.gesture_active(now)
                obs.touched = touch.update(last_ants, present_ants) if calm else False
                if now >= next_doa:
                    next_doa = now + 0.2
                    obs.loud_yaw_deg = loud.update(reachy_mini.media.get_DoA(), now)
                sighting = vision.latest()
                if sighting is not None and now - sighting.ts < 1.0:
                    obs.face = self._to_face_obs(sighting, K, D, T_head_cam)

                # ---------------- brain
                actions = behavior.tick(obs, now, dt)
                composer.energy = behavior.mood.energy
                composer.mode = {"SLEEPING": "sleeping", "HELD": "held"}.get(behavior.state, "awake")
                composer.set_gaze(behavior.gaze)
                for act in actions:
                    move, move_t0 = self._dispatch(act, now, sound, composer, library, move, move_t0, reachy_mini)

                # ---------------- body
                if move is not None:
                    t = now - move_t0
                    if t >= move.duration - 0.02:
                        move = None
                        blend_from = (last_pose, list(last_ants), now)
                    else:
                        head, ants, _ = move.evaluate(t)
                        last_pose, last_ants = head, [float(ants[0]), float(ants[1])]
                        reachy_mini.set_target(head=head, antennas=last_ants)
                if move is None:
                    head, ants = composer.sample(now, dt)
                    if blend_from is not None:
                        a = (now - blend_from[2]) / MOVE_BLEND_S
                        if a >= 1.0:
                            blend_from = None
                        else:
                            head = linear_pose_interpolation(blend_from[0], head, a)
                            ants = [blend_from[1][i] * (1 - a) + ants[i] * a for i in range(2)]
                    last_pose, last_ants = head, ants
                    reachy_mini.set_target(head=head, antennas=ants)

                memory.save()
                sleep_for = period - (time.time() - now)
                if sleep_for > 0:
                    time.sleep(sleep_for)
        finally:
            vision.stop()
            sound.stop()
            memory.save(force=True)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _to_face_obs(s: Sighting, K: np.ndarray, D: np.ndarray, T_head_cam: np.ndarray) -> FaceObs:
        pose = look_at_image_pose(s.u, s.v, K, D, s.head_pose_at_capture, T_head_cam)
        roll, pitch, yaw = R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
        return FaceObs(s.track_id, float(yaw), float(pitch), s.area_frac, s.person, s.similarity)

    def _dispatch(
        self,
        act: Action,
        now: float,
        sound: SoundPlayer,
        composer: MotionComposer,
        library: RecordedMoves,
        move: RecordedMove | None,
        move_t0: float,
        reachy: ReachyMini,
    ) -> tuple[RecordedMove | None, float]:
        if act.kind == "sound":
            sound.request(act.name, act.priority, now)
        elif act.kind == "gesture":
            if move is None:
                composer.request_gesture(act.name, now, act.priority)
        elif act.kind == "move":
            if move is None:
                move = library.get(act.name)
                head0, ants0, _ = move.evaluate(0.0)
                reachy.goto_target(head=head0, antennas=list(ants0), duration=0.4)
                move_t0 = time.time()
                if PLAY_LIBRARY_SOUNDS and move.sound_path is not None:
                    reachy.media.play_sound(str(move.sound_path))
        elif act.kind == "wake":
            sound.request("wake", 5, now)
            composer.request_gesture("perk", now, 5)
        elif act.kind == "sleep":
            pass  # composer.mode handles the pose; the yawn sound was queued by the brain
        return move, move_t0

    def _install_status_routes(self, behavior: Behavior, memory: FaceMemory, vision: Vision) -> None:
        if self.settings_app is None:
            return

        @self.settings_app.get("/api/status")
        def status() -> dict:
            return {"behavior": behavior.status(), "memory": memory.summary(), "vision": vision.stats}

        @self.settings_app.post("/api/forget")
        def forget() -> dict:
            memory.people.clear()
            memory._dirty = True
            memory.save(force=True)
            return {"ok": True}


if __name__ == "__main__":
    app = FestivalPetApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
