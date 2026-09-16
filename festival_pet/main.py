"""Festival Pet: the robot-side glue.

Runs as a standard Reachy Mini app (dashboard-launchable) on the wireless unit.
Everything is offline: local YuNet/SFace models, a Vosk keyword model, procedural
beeps, a JSON memory file, and the emotions library from the daemon's HF cache.

Threads
    main (this)   50 Hz control loop: senses -> behavior -> motion -> set_target
    vision        ~8 Hz detection, occasional SFace embedding
    audio-sense   mic stream -> beat tracker, scratch detector, name spotter
    sound         renders + pushes phrases so a 1 s purr never stalls motion

``Pet`` takes a ``RobotIO`` so the same loop runs against the real robot or the
MuJoCo simulator with fake senses (see scripts/sim_harness.py).
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import platformdirs
from scipy.spatial.transform import Rotation as R

from festival_pet import sounds
from festival_pet.audio_features import BeatTracker, ScratchDetector
from festival_pet.behavior import Action, Behavior, FaceObs, Observation
from festival_pet.memory import FaceMemory
from festival_pet.motion import MotionComposer
from festival_pet.senses import LoudSoundDetector, PickupDetector, TouchDetector
from festival_pet.vision import Sighting

logger = logging.getLogger("festival_pet")

DATA_DIR = Path(platformdirs.user_data_dir("festival_pet"))
YUNET_MODEL = DATA_DIR / "models" / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = DATA_DIR / "models" / "face_recognition_sface_2021dec.onnx"
VOSK_MODEL = DATA_DIR / "models" / "vosk-model-small-en-us-0.15"
MEMORY_FILE = DATA_DIR / "memory.json"

CONTROL_HZ = 50.0
PLAY_LIBRARY_SOUNDS = False  # True = play Pollen's sidecar sound with library moves instead of our beeps
MOVE_BLEND_S = 0.5
FREE_DANCE_BPM = 108.0
AUDIO_RATE = 16000


class RobotIO(Protocol):
    """Everything the pet needs from the robot; implemented for the real robot and for the sim."""

    K: np.ndarray
    D: np.ndarray
    T_head_cam: np.ndarray

    def head_pose(self) -> np.ndarray: ...
    def imu(self) -> dict | None: ...
    def present_antennas(self) -> list[float]: ...
    def doa(self) -> tuple[float, bool] | None: ...
    def audio_chunk(self) -> np.ndarray | None: ...  # mono float32 @ 16 kHz, or None when nothing new
    def play(self, buf: np.ndarray) -> None: ...
    def play_file(self, path: str) -> None: ...
    def set_target(self, head: np.ndarray, antennas: list[float]) -> None: ...
    def goto(self, head: np.ndarray, antennas: list[float], duration: float) -> None: ...


class MoveLike(Protocol):
    duration: float
    sound_path: Path | None

    def evaluate(self, t: float) -> tuple[np.ndarray, np.ndarray, float]: ...


class SoundPlayer:
    """Serialises phrases onto the speaker from a worker thread."""

    def __init__(self, io: RobotIO, sample_rate: int = AUDIO_RATE) -> None:
        self._io = io
        self._q: queue.Queue[tuple[int, str]] = queue.Queue()
        self._busy_until = 0.0
        self._busy_priority = 0
        self._sample_rate = sample_rate
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="festival-pet-sound", daemon=True)
        self.played: list[str] = []  # recent history, handy for the status page and tests

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

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
            self.played.append(emotion)
            del self.played[:-30]
            # Push in ~40 ms chunks so playback starts immediately and the buffer never balloons.
            chunk = int(self._sample_rate * 0.04)
            for i in range(0, len(buf), chunk):
                self._io.play(buf[i : i + chunk])
            time.sleep(dur)


class AudioSense:
    """Mic stream -> beat, scratch, and name/command events, from one worker thread."""

    SPEECH_HANGOVER = 1.0

    def __init__(self, io: RobotIO, spotter, doa_period: float = 0.1) -> None:
        self._io = io
        self.beat = BeatTracker()
        self.scratch = ScratchDetector()
        self.spotter = spotter  # NameSpotter or None
        self._events: queue.Queue[tuple[str, str, float | None]] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="festival-pet-audio", daemon=True)
        self._doa_period = doa_period
        self.last_voice_yaw: float | None = None
        self.stats = {"chunks": 0, "listening": False}

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def poll(self) -> list[tuple[str, str, float | None]]:
        out = []
        while True:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                return out

    def _run(self) -> None:
        speech_until = 0.0
        listening = False
        next_doa = 0.0
        while not self._stop.is_set():
            now = time.time()
            chunk = self._io.audio_chunk()
            if chunk is None:
                time.sleep(0.005)
                continue
            self.stats["chunks"] += 1
            self.beat.push(chunk, now)
            if self.scratch.push(chunk, now):
                self._events.put(("scratch", "", None))
            if self.spotter is None:
                continue
            if now >= next_doa:
                next_doa = now + self._doa_period
                doa = self._io.doa()
                if doa is not None and doa[1]:
                    speech_until = now + self.SPEECH_HANGOVER
                    self.last_voice_yaw = LoudSoundDetector.doa_to_yaw(doa[0])
            if now < speech_until:
                listening = True
                self.spotter.push(chunk)
                for w in self.spotter.poll():
                    self._events.put(("word", w, self.last_voice_yaw))
            elif listening:
                listening = False
                self.spotter.flush()
                for w in self.spotter.poll():
                    self._events.put(("word", w, self.last_voice_yaw))
            self.stats["listening"] = listening


@dataclass
class PetParts:
    """Everything a Pet is assembled from, so the harness can swap pieces."""

    io: RobotIO
    memory: FaceMemory
    library: Callable[[str], MoveLike]  # move name -> move
    latest_sighting: Callable[[], Sighting | None]
    spotter: object | None
    behavior: Behavior
    composer: MotionComposer


class Pet:
    """The control loop. ``step`` is public so the harness can drive it deterministically."""

    def __init__(self, parts: PetParts) -> None:
        self.p = parts
        self.sound = SoundPlayer(parts.io)
        self.audio = AudioSense(parts.io, parts.spotter)
        self.pickup = PickupDetector()
        self.touch = TouchDetector()
        self.loud = LoudSoundDetector()
        self.move: MoveLike | None = None
        self.move_t0 = 0.0
        self.last_pose = np.eye(4)
        self.last_ants = [0.0, 0.0]
        self.blend_from: tuple[np.ndarray, list[float], float] | None = None
        self.actions_log: list[tuple[float, str, str]] = []
        self._next_doa = 0.0
        self._last = 0.0

    # ------------------------------------------------------------------ lifecycle
    def start(self, now: float) -> None:
        self.sound.start()
        self.audio.start()
        self.p.behavior.start(now, awake=True)  # the platform wakes the robot before launching an app
        self._dispatch(Action("wake", "start", 5), now)
        self._last = now

    def stop(self) -> None:
        self.audio.stop()
        self.sound.stop()
        self.p.memory.save(force=True)

    def run(self, stop_event: threading.Event) -> None:
        self.start(time.time())
        period = 1.0 / CONTROL_HZ
        try:
            while not stop_event.is_set():
                now = time.time()
                self.step(now)
                sleep_for = period - (time.time() - now)
                if sleep_for > 0:
                    time.sleep(sleep_for)
        finally:
            self.stop()

    # ------------------------------------------------------------------ one tick
    def step(self, now: float) -> None:
        io, beh, comp = self.p.io, self.p.behavior, self.p.composer
        dt = max(1e-3, now - self._last)
        self._last = now

        # ---------------- senses
        obs = Observation()
        imu = io.imu()
        if imu is not None:
            obs.held, obs.shaken = self.pickup.update(imu["accelerometer"], imu["gyroscope"], now)
        # Antennas lag their command while animated; the detector raises its threshold then.
        busy = self.move is not None or comp.gesture_active(now)
        obs.touched = self.touch.update(self.last_ants, io.present_antennas(), busy)
        if now >= self._next_doa:
            self._next_doa = now + 0.2
            obs.loud_yaw_deg = self.loud.update(io.doa(), now)
        sighting = self.p.latest_sighting()
        if sighting is not None and now - sighting.ts < 1.0:
            obs.face = self._to_face_obs(sighting)
        for kind, value, yaw in self.audio.poll():
            if kind == "scratch":
                obs.scratched = True
            elif value == "reachy":
                obs.name_heard = True
                obs.voice_yaw_deg = yaw
            else:
                obs.command = value
                obs.voice_yaw_deg = yaw
        if self.audio.beat.music:
            obs.music_bpm = self.audio.beat.state.bpm
            obs.music_confidence = self.audio.beat.state.confidence

        # ---------------- brain
        actions = beh.tick(obs, now, dt)
        comp.energy = beh.mood.energy
        comp.mode = {"SLEEPING": "sleeping", "HELD": "held"}.get(beh.state, "awake")
        comp.set_gaze(beh.gaze)
        comp.groove = None
        comp.mirror_roll = 0.0
        for act in actions:
            self._dispatch(act, now)

        # ---------------- body
        if self.move is not None:
            t = now - self.move_t0
            if t >= self.move.duration - 0.02:
                self.move = None
                self.blend_from = (self.last_pose, list(self.last_ants), now)
            else:
                head, ants, _ = self.move.evaluate(t)
                self.last_pose, self.last_ants = head, [float(ants[0]), float(ants[1])]
                io.set_target(head, self.last_ants)
        if self.move is None:
            head, ants = comp.sample(now, dt)
            if self.blend_from is not None:
                a = (now - self.blend_from[2]) / MOVE_BLEND_S
                if a >= 1.0:
                    self.blend_from = None
                else:
                    from reachy_mini.utils.interpolation import linear_pose_interpolation

                    head = linear_pose_interpolation(self.blend_from[0], head, a)
                    ants = [self.blend_from[1][i] * (1 - a) + ants[i] * a for i in range(2)]
            self.last_pose, self.last_ants = head, ants
            io.set_target(head, ants)
        self.p.memory.save()

    # ------------------------------------------------------------------ helpers
    def _to_face_obs(self, s: Sighting) -> FaceObs:
        from reachy_mini.vision.look_at import look_at_image_pose

        io = self.p.io
        pose = look_at_image_pose(s.u, s.v, io.K, io.D, s.head_pose_at_capture, io.T_head_cam)
        roll, pitch, yaw = R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
        return FaceObs(s.track_id, float(yaw), float(pitch), s.area_frac, s.person, s.similarity, s.roll_deg)

    def _dispatch(self, act: Action, now: float) -> None:
        comp = self.p.composer
        if act.kind not in ("groove", "mirror"):
            self.actions_log.append((now, act.kind, act.name))
            del self.actions_log[:-2000]
        if act.kind == "sound":
            self.sound.request(act.name, act.priority, now)
        elif act.kind == "gesture":
            if self.move is None:
                comp.request_gesture(act.name, now, act.priority)
        elif act.kind == "move":
            if self.move is None:
                move = self.p.library(act.name)
                head0, ants0, _ = move.evaluate(0.0)
                self.p.io.goto(head0, [float(ants0[0]), float(ants0[1])], 0.4)
                self.move, self.move_t0 = move, time.time()
                if PLAY_LIBRARY_SOUNDS and move.sound_path is not None:
                    self.p.io.play_file(str(move.sound_path))
        elif act.kind == "wake":
            self.sound.request("wake", 5, now)
            comp.request_gesture("perk", now, 5)
        elif act.kind == "sleep":
            pass  # composer.mode handles the pose; the yawn sound was queued by the brain
        elif act.kind == "groove":
            beat = self.audio.beat
            if beat.music:
                phase, period, t0 = beat.phase(now), beat.state.period, beat._last_beat_time
            else:  # no music: the little-dance trick bobs to its own inner tempo
                period, t0 = 60.0 / FREE_DANCE_BPM, 0.0
                phase = (now / period) % 1.0
            bar = ((now - t0) / (4 * period)) % 1.0
            comp.groove = (phase, bar, float(act.name))
        elif act.kind == "mirror":
            comp.mirror_roll = float(act.name)

    def status(self) -> dict:
        b = self.audio.beat.state
        return {
            "behavior": self.p.behavior.status(),
            "memory": self.p.memory.summary(),
            "audio": {"bpm": round(b.bpm, 1), "confidence": round(b.confidence, 2), "music": self.audio.beat.music, **self.audio.scratch.stats, **self.audio.stats},
            "recent_sounds": self.sound.played[-10:],
            "recent_actions": [f"{k}:{n}" for _, k, n in self.actions_log[-15:]],
        }


# ============================================================================ real robot


class ReachyIO:
    """RobotIO over the SDK's ReachyMini with the local media backend."""

    def __init__(self, reachy) -> None:
        self._r = reachy
        camera = reachy.media.camera
        if camera is None or camera.K is None or camera.D is None:
            raise RuntimeError("Camera intrinsics unavailable; the app needs the local media backend on the robot.")
        self.K, self.D, self.T_head_cam = camera.K, camera.D, reachy.T_head_cam
        in_rate = reachy.media.get_input_audio_samplerate()
        out_rate = reachy.media.get_output_audio_samplerate()
        if in_rate != AUDIO_RATE or out_rate != AUDIO_RATE:
            raise RuntimeError(f"Expected {AUDIO_RATE} Hz audio in/out, got {in_rate}/{out_rate}")
        reachy.media.start_recording()
        reachy.media.start_playing()

    def close(self) -> None:
        self._r.media.stop_recording()
        self._r.media.stop_playing()

    def head_pose(self):
        return self._r.get_current_head_pose()

    def imu(self):
        return self._r.imu

    def present_antennas(self):
        return self._r.get_present_antenna_joint_positions()

    def doa(self):
        return self._r.media.get_DoA()

    def audio_chunk(self):
        data = self._r.media.get_audio_sample()
        if data is None:
            return None
        return data.mean(axis=1).astype(np.float32) if data.ndim == 2 else data.astype(np.float32)

    def play(self, buf):
        self._r.media.push_audio_sample(buf)

    def play_file(self, path):
        self._r.media.play_sound(path)

    def set_target(self, head, antennas):
        self._r.set_target(head=head, antennas=antennas)

    def goto(self, head, antennas, duration):
        self._r.goto_target(head=head, antennas=antennas, duration=duration)


def build_real_pet(reachy, memory_file: Path = MEMORY_FILE, name_spotting: bool = True) -> tuple[Pet, object, ReachyIO]:
    """Assemble the pet for the real robot. Returns (pet, vision, io) so the caller can start/stop them."""
    from reachy_mini.motion.recorded_move import DEFAULT_EMOTIONS_DATASET, RecordedMoves

    from festival_pet.vision import Vision

    for path in (YUNET_MODEL, SFACE_MODEL):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run scripts/setup_offline.sh while online.")
    io = ReachyIO(reachy)
    memory = FaceMemory(memory_file)
    library = RecordedMoves(DEFAULT_EMOTIONS_DATASET)
    vision = Vision(YUNET_MODEL, SFACE_MODEL, memory, reachy.media.get_frame, reachy.get_current_head_pose)
    spotter = None
    if name_spotting:
        from festival_pet.hearing import NameSpotter

        spotter = NameSpotter(VOSK_MODEL)
    parts = PetParts(io, memory, library.get, vision.latest, spotter, Behavior(memory), MotionComposer())
    return Pet(parts), vision, io


try:
    from reachy_mini import ReachyMini, ReachyMiniApp
except ImportError:  # the pure modules stay importable off-robot
    ReachyMiniApp = object  # type: ignore[misc,assignment]


class FestivalPetApp(ReachyMiniApp):  # type: ignore[misc]
    """A non-verbal, expressive, face-remembering droid."""

    custom_app_url: str | None = "http://0.0.0.0:8042"
    request_media_backend: str | None = None  # we need camera + audio

    def run(self, reachy_mini: "ReachyMini", stop_event: threading.Event) -> None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
        os.environ["HF_HUB_OFFLINE"] = "1"  # never touch the network on the festival ground
        pet, vision, io = build_real_pet(reachy_mini)
        self._install_status_routes(pet)
        vision.start()
        try:
            pet.run(stop_event)
        finally:
            vision.stop()
            io.close()

    def _install_status_routes(self, pet: Pet) -> None:
        if self.settings_app is None:
            return

        @self.settings_app.get("/api/status")
        def status() -> dict:
            return pet.status()

        @self.settings_app.post("/api/forget")
        def forget() -> dict:
            pet.p.memory.people.clear()
            pet.p.memory._dirty = True
            pet.p.memory.save(force=True)
            return {"ok": True}


if __name__ == "__main__":
    app = FestivalPetApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
