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
import math
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import platformdirs
from pydantic import BaseModel
from scipy.spatial.transform import Rotation as R

from festival_pet import sounds
from festival_pet.audio_features import BeatTracker, LevelMeter, RubDetector, ScratchDetector, _tune
from festival_pet.behavior import Action, Behavior, FaceObs, Observation
from festival_pet.memory import FaceMemory
from festival_pet.motion import MotionComposer
from festival_pet.senses import LoudSoundDetector, PickupDetector, SelfMotionGate, TouchDetector
from festival_pet.vision import Sighting

logger = logging.getLogger("festival_pet")


class RingLogHandler(logging.Handler):
    """Keeps the last N log lines in memory for the web page."""

    def __init__(self, capacity: int = 400) -> None:
        super().__init__()
        self.lines: deque[str] = deque(maxlen=capacity)
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


LOG_RING = RingLogHandler()

DATA_DIR = Path(platformdirs.user_data_dir("festival_pet"))
YUNET_MODEL = DATA_DIR / "models" / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = DATA_DIR / "models" / "face_recognition_sface_2021dec.onnx"
VOSK_MODEL = DATA_DIR / "models" / "vosk-model-small-en-us-0.15"
PERSON_MODEL = DATA_DIR / "models" / "person_detection_mediapipe_2023mar.onnx"
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
    def set_target(self, head: np.ndarray, antennas: list[float], body_yaw: float) -> None: ...
    def goto(self, head: np.ndarray, antennas: list[float], duration: float) -> None: ...
    def sleep_body(self) -> None: ...  # nest the head, then torque off
    def wake_body(self) -> None: ...  # torque on, lift to neutral


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
        self.rub = RubDetector()
        self.meter = LevelMeter()
        self._calib: dict | None = None  # {"kind", "phase", "until", "baseline": [..], "active": [..]}
        self.calibration_result: dict | None = None
        self.last_speech_time = -1e9
        self.speech_edge = False  # set when speech starts after a pause; consumed by the main loop
        self.spotter = spotter  # NameSpotter or None
        self._events: queue.Queue[tuple[str, str, float | None]] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="festival-pet-audio", daemon=True)
        self._doa_period = doa_period
        self.last_voice_yaw: float | None = None
        self.doa: tuple[float, bool] | None = None  # latest reading; the ONLY place the USB mic array is polled
        self._doa_backoff_until = 0.0
        self._doa_errors = 0
        self.stats = {"chunks": 0, "listening": False, "doa_errors": 0}

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    # ------------------------------------------------------------ calibration
    def start_calibration(self, kind: str, now: float) -> None:
        """3 s of quiet baseline, then 3 s during which the user rubs/scratches; thresholds set from the two."""
        if kind not in ("rub", "scratch"):
            raise KeyError(kind)
        self._calib = {"kind": kind, "phase": "baseline", "until": now + 3.0, "baseline": [], "active": []}
        self.calibration_result = {"kind": kind, "phase": "baseline: keep quiet for 3 s"}

    def _calibration_step(self, now: float) -> None:
        c = self._calib
        if c is None:
            return
        snap = {**self.rub.stats, **self.scratch.stats}
        c[c["phase"]].append(snap)
        if now < c["until"]:
            return
        if c["phase"] == "baseline":
            c["phase"], c["until"] = "active", now + 3.0
            self.calibration_result = {"kind": c["kind"], "phase": f"NOW {'rub the head' if c['kind'] == 'rub' else 'scratch the belly'} for 3 s"}
            return
        keyE = "rub_energy" if c["kind"] == "rub" else "band_energy"
        base = {keyE: float(np.median([x[keyE] for x in c["baseline"]])), "flatness": float(np.median([x["flatness"] for x in c["baseline"]]))}
        act = {keyE: float(np.percentile([x[keyE] for x in c["active"]], 80)), "flatness": float(np.median([x["flatness"] for x in c["active"]]))}
        self._calib = None
        if act[keyE] < 3.0 * base[keyE]:
            self.calibration_result = {"kind": c["kind"], "phase": "failed", "baseline": base, "active": act,
                                       "reason": "the touch was not clearly louder than the quiet phase (need 3x); thresholds unchanged"}
            logger.warning("calibration of %s failed: baseline=%s active=%s", c["kind"], base, act)
            return
        tuned = _tune(c["kind"], base, act)
        det = self.rub if c["kind"] == "rub" else self.scratch
        for k, v in tuned.items():
            if k != "observed_ratio":
                setattr(det, k, float(v))
        self.calibration_result = {"kind": c["kind"], "phase": "done", "baseline": base, "active": act, "set": tuned}
        logger.info("calibrated %s: baseline=%s active=%s -> %s", c["kind"], base, act, tuned)

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
            scratched = self.scratch.push(chunk, now)
            rubbed = self.rub.push(chunk, now)
            self.meter.push(chunk, now, {"rub": self.rub.stats["rub_energy"], "flat": self.rub.stats["flatness"], "band": self.scratch.stats["band_energy"], "clicks": self.scratch.stats["clicks_in_window"]})
            self._calibration_step(now)
            if scratched:
                self._events.put(("scratch", "", None))
            if rubbed:
                self._events.put(("pet", "", None))
            if now >= next_doa and now >= self._doa_backoff_until:
                next_doa = now + self._doa_period
                try:
                    self.doa = self._io.doa()
                except Exception as e:
                    # The ReSpeaker DoA is a USB control transfer; it occasionally errors. Log it, back off, keep living.
                    self._doa_errors += 1
                    self.stats["doa_errors"] = self._doa_errors
                    self._doa_backoff_until = now + min(30.0, 2.0 * self._doa_errors)
                    logger.warning("DoA read failed (%d so far, backing off %.0fs): %r", self._doa_errors, self._doa_backoff_until - now, e)
                if self.doa is not None and self.doa[1]:
                    if now - self.last_speech_time > 1.5:
                        self.speech_edge = True  # someone started talking after a pause
                    self.last_speech_time = now
                    speech_until = now + self.SPEECH_HANGOVER
                    self.last_voice_yaw = LoudSoundDetector.doa_to_yaw(self.doa[0])
            if self.spotter is None:
                continue
            # Listen when the mic array flags speech OR the level is clearly above the recent noise floor,
            # so a stale/absent DoA flag can never leave it deaf.
            floor = self.meter.noise_floor
            if self.meter.rms > 3.0 * floor and self.meter.rms > 0.004:
                speech_until = max(speech_until, now + self.SPEECH_HANGOVER)
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
            for sentence in self.spotter.poll_transcript():
                self._events.put(("heard", sentence, self.last_voice_yaw))
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
        self.self_motion = SelfMotionGate()
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
        self.muted = False
        self.groove_scale = 1.0  # user knob on top of the brain's intensity
        self.pickup_enabled = False  # IMU is in the head; off by default until tuned on the real robot
        self.asleep = False  # motors off, camera paused; ears stay on
        self._touch_resync_at = -1.0  # after sleep/wake the antennas settle somewhere new: re-zero the ear detector then
        self.set_vision_active: Callable[[bool], None] = lambda active: None
        self.transcript: deque[tuple[float, str, list[str]]] = deque(maxlen=30)
        self._last_obs = Observation()

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
        # The IMU is in the head: ignore it while we are the ones moving the head.
        self_moving = self.self_motion.update(self.last_pose, now)
        imu = io.imu()
        if imu is not None:
            held, shaken = self.pickup.update(imu["accelerometer"], imu["gyroscope"], now, self_moving)
            if self.pickup_enabled:
                obs.held, obs.shaken = held, shaken
        # Antennas lag their command while animated; the detector raises its threshold then.
        busy = self.move is not None or comp.gesture_active(now)
        present_ants = io.present_antennas()
        if self._touch_resync_at >= 0:
            if now >= self._touch_resync_at:  # motors settled after sleep/wake: wherever the ears rest now is "untouched"
                self.last_ants = list(present_ants)
                self.touch = TouchDetector()
                self._touch_resync_at = -1.0
            obs.touched = False
        else:
            obs.touched = self.touch.update(self.last_ants, present_ants, busy, dt)
        obs.touched_side = self.touch.last_side
        obs.petting = self.audio.rub.rubbing
        if now >= self._next_doa:
            self._next_doa = now + 0.2
            obs.loud_yaw_deg = self.loud.update(self.audio.doa, now)
        sighting = self.p.latest_sighting()
        if sighting is not None and now - sighting.ts < 1.0:
            if sighting.kind == "face":
                obs.face = self._to_face_obs(sighting)
            else:
                obs.body = self._to_face_obs(sighting)
        for kind, value, yaw in self.audio.poll():
            if kind == "scratch":
                obs.scratched = True
            elif kind == "pet":
                obs.petted = True
            elif kind == "heard":
                obs.heard_text = value
                obs.voice_yaw_deg = yaw
            elif value == "reachy":
                obs.name_heard = True
                obs.voice_yaw_deg = yaw
            else:
                obs.command = value
                obs.voice_yaw_deg = yaw
        if self.audio.speech_edge:
            self.audio.speech_edge = False
            obs.voice_started = True
            obs.voice_yaw_deg = self.audio.last_voice_yaw
        if self.audio.beat.music:
            obs.music_bpm = self.audio.beat.state.bpm
            obs.music_confidence = self.audio.beat.state.confidence
        self._last_obs = obs

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
                io.set_target(head, self.last_ants, math.radians(comp.body_yaw))
        if self.move is None:
            head, ants, body_yaw = comp.sample(now, dt)
            if self.blend_from is not None:
                a = (now - self.blend_from[2]) / MOVE_BLEND_S
                if a >= 1.0:
                    self.blend_from = None
                else:
                    from reachy_mini.utils.interpolation import linear_pose_interpolation

                    head = linear_pose_interpolation(self.blend_from[0], head, a)
                    ants = [self.blend_from[1][i] * (1 - a) + ants[i] * a for i in range(2)]
            self.last_pose, self.last_ants = head, ants
            if not self.asleep:
                io.set_target(head, ants, math.radians(body_yaw))
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
            if not self.muted:
                self.sound.request(act.name, act.priority, now)
        elif act.kind == "gesture":
            if self.move is None:
                name, _, side = act.name.partition(":")  # "flinch:+" forces the side of a sided gesture
                comp.request_gesture(name, now, act.priority, {"+": 1.0, "-": -1.0, "": None}[side])
        elif act.kind == "move":
            if self.move is None:
                move = self.p.library(act.name)
                head0, ants0, _ = move.evaluate(0.0)
                self.p.io.goto(head0, [float(ants0[0]), float(ants0[1])], 0.4)
                self.move, self.move_t0 = move, time.time()
                if PLAY_LIBRARY_SOUNDS and move.sound_path is not None:
                    self.p.io.play_file(str(move.sound_path))
        elif act.kind == "wake":
            if self.asleep:
                self.p.io.wake_body()  # torque on, lift the head; blocks ~2 s
                self.asleep = False
                self.set_vision_active(True)
                self.move, self.blend_from = None, None
                self._touch_resync_at = now + 1.0
            self.sound.request("wake", 5, now)
            comp.request_gesture("perk", now, 5)
        elif act.kind == "sleep":
            if not self.asleep:
                self.set_vision_active(False)  # camera off while asleep; ears stay on for name / noise / pets
                self.p.io.sleep_body()  # nest the head, then torque off; blocks ~4 s
                self.asleep = True
                self.move = None
                self._touch_resync_at = now + 2.5
        elif act.kind == "groove":
            beat = self.audio.beat
            if beat.music:
                phase, period, t0 = beat.phase(now), beat.state.period, beat._last_beat_time
            else:  # no music: the little-dance trick bobs to its own inner tempo
                period, t0 = 60.0 / FREE_DANCE_BPM, 0.0
                phase = (now / period) % 1.0
            bar = ((now - t0) / (4 * period)) % 1.0
            comp.groove = (phase, bar, float(act.name) * self.groove_scale)
        elif act.kind == "mirror":
            comp.mirror_roll = float(act.name)
        elif act.kind == "heard":
            self.transcript.append((now, act.name, act.name.split("|")[1:]))

    def status(self) -> dict:
        b = self.audio.beat.state
        return {
            "behavior": self.p.behavior.status(),
            "memory": self.p.memory.summary(),
            "audio": {"bpm": round(b.bpm, 1), "confidence": round(b.confidence, 2), "music": self.audio.beat.music, **self.audio.scratch.stats, **self.audio.rub.stats, **self.audio.stats},
            "recent_sounds": self.sound.played[-10:],
            "recent_actions": [f"{k}:{n}" for _, k, n in self.actions_log[-15:]],
        }

    def mind(self) -> dict:
        now = time.time()
        o = self._last_obs
        b = self.audio.beat.state
        comp = self.p.composer
        return {
            "mind": self.p.behavior.mind(now),
            "asleep": self.asleep,
            "transcript": [{"t": round(now - t, 1), "text": txt.split("|")[0], "intents": ints} for t, txt, ints in reversed(self.transcript)],
            "senses": {
                "body": None if o.body is None else {"yaw": round(o.body.yaw_deg, 1), "pitch": round(o.body.pitch_deg, 1), "size": round(o.body.area_frac, 3)},
                "face": None if o.face is None else {"track": o.face.track_id, "yaw": round(o.face.yaw_deg, 1), "pitch": round(o.face.pitch_deg, 1), "size": round(o.face.area_frac, 3), "person": None if o.face.person is None else o.face.person.person_id, "similarity": round(o.face.similarity, 2), "tilt": round(o.face.roll_deg, 1)},
                "held": o.held, "shaken": o.shaken, "imu": self.pickup.stats, "head_rate": round(self.self_motion.rate, 2), "ears": self.touch.stats,
                "music": {"bpm": round(b.bpm, 1), "confidence": round(b.confidence, 2), "grooving": comp.groove is not None, "intensity": round(comp.groove[2], 2) if comp.groove else 0.0},
                "listening": self.audio.stats["listening"], "voice_yaw": self.audio.last_voice_yaw,
                "doa": None if self.audio.doa is None else {"angle_deg": round(math.degrees(self.audio.doa[0]), 0), "speech": self.audio.doa[1]},
                "speech_s_ago": round(now - self.audio.last_speech_time, 1) if self.audio.last_speech_time > 0 else None,
                "levels": {"rms": round(self.audio.meter.rms, 5), "peak": round(self.audio.meter.peak, 4), "max_rms_3s": round(self.audio.meter.max_rms_3s, 5), "chunks": self.audio.stats["chunks"]},
                "scratch": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.audio.scratch.stats.items()},
                "head_pet": {"rubbing": self.audio.rub.rubbing, **{k: round(v, 6) for k, v in self.audio.rub.stats.items()}},
            },
            "controls": {"muted": self.muted, "pickup": self.pickup_enabled, "body_finder": getattr(getattr(self, "vision", None), "body_enabled", None), "groove_scale": self.groove_scale, "scratch_onset_ratio": self.audio.scratch.onset_ratio, "scratch_floor": self.audio.scratch.floor, "rub_level_ratio": self.audio.rub.level_ratio, "rub_flatness_min": self.audio.rub.flatness_min, "rub_floor": self.audio.rub.floor, "match_threshold": self.p.memory.match_threshold},
            "calibration": self.audio.calibration_result,
            "audio_history": self.audio.meter.history(),
            "recent_actions": [{"t": round(now - t, 1), "a": f"{k}:{n}"} for t, k, n in reversed(self.actions_log[-20:])],
            "memory": self.p.memory.summary(),
        }

    def control(self, cmd: str, value: str | float | None) -> dict:
        """Manual controls from the web page. Raises on unknown commands / names."""
        now = time.time()
        beh = self.p.behavior
        if cmd == "sound":
            if str(value) not in sounds.EMOTIONS:
                raise KeyError(f"unknown sound '{value}'")
            self._dispatch(Action("sound", str(value), 5), now)
        elif cmd == "gesture":
            self.p.composer.request_gesture(str(value), now, 5)  # raises KeyError on unknown names
            self.actions_log.append((now, "gesture", f"{value} (manual)"))
        elif cmd == "move":
            self._dispatch(Action("move", str(value), 5), now)
        elif cmd == "sleep":
            beh.state, beh._state_since = "SLEEPING", now
            beh._think(now, "told to sleep from the control page")
        elif cmd == "wake":
            beh.state, beh._state_since = "WAKING", now
            self._dispatch(Action("wake", "control", 5), now)
        elif cmd == "mute":
            self.muted = bool(value)
        elif cmd == "pickup":
            self.pickup_enabled = bool(value)
        elif cmd == "groove_scale":
            self.groove_scale = float(value)
        elif cmd == "scratch_onset_ratio":
            self.audio.scratch.onset_ratio = float(value)
        elif cmd == "scratch_floor":
            self.audio.scratch.floor = float(value)
        elif cmd == "rub_level_ratio":
            self.audio.rub.level_ratio = float(value)
        elif cmd == "rub_flatness_min":
            self.audio.rub.flatness_min = float(value)
        elif cmd == "rub_floor":
            self.audio.rub.floor = float(value)
        elif cmd == "calibrate":
            self.audio.start_calibration(str(value), now)
        elif cmd == "body_finder":
            vision = getattr(self, "vision", None)
            if vision is None:
                raise KeyError("no vision on this pet")
            vision.body_enabled = bool(value)
        elif cmd == "match_threshold":
            self.p.memory.match_threshold = float(value)
        elif cmd == "forget":
            self.p.memory.people.clear()
            self.p.memory._dirty = True
            self.p.memory.save(force=True)
        else:
            raise KeyError(f"unknown control '{cmd}'")
        return {"ok": True}


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

    def set_target(self, head, antennas, body_yaw):
        self._r.set_target(head=head, antennas=antennas, body_yaw=body_yaw)

    def goto(self, head, antennas, duration):
        self._r.goto_target(head=head, antennas=antennas, duration=duration)

    def sleep_body(self):
        self._r.goto_sleep()  # SDK: lift if needed, "pfiou" sound, nest the head over 2 s
        self._r.disable_motors()  # torque off: the head rests in its cradle, no motor hum

    def wake_body(self):
        from reachy_mini.reachy_mini import INIT_ANTENNAS_JOINT_POSITIONS, INIT_HEAD_POSE

        self._r.enable_motors()
        self._r.goto_target(head=INIT_HEAD_POSE, antennas=INIT_ANTENNAS_JOINT_POSITIONS, duration=1.5)


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
    vision = Vision(YUNET_MODEL, SFACE_MODEL, memory, reachy.media.get_frame, reachy.get_current_head_pose,
                    person_model=PERSON_MODEL if PERSON_MODEL.exists() else None)
    if not PERSON_MODEL.exists():
        logger.warning("person model missing (%s): body-finding disabled; rerun the installer", PERSON_MODEL)
    spotter = None
    if name_spotting:
        from festival_pet.hearing import NameSpotter

        spotter = NameSpotter(VOSK_MODEL)
    parts = PetParts(io, memory, library.get, vision.latest, spotter, Behavior(memory), MotionComposer())
    pet = Pet(parts)
    pet.set_vision_active = vision.set_active
    pet.vision = vision
    return pet, vision, io


class Control(BaseModel):
    """A manual control request from the web page."""

    cmd: str
    value: str | float | bool | None = None


class Merge(BaseModel):
    keep: int
    other: int


def install_routes(app, pet: Pet) -> None:
    """Web API behind the pet's page (port 8042). Shared by the app and the tests."""
    from fastapi import HTTPException

    @app.get("/api/status")
    def status() -> dict:
        return pet.status()

    @app.get("/api/mind")
    def mind() -> dict:
        return pet.mind()

    @app.get("/api/log")
    def log(n: int = 200) -> dict:
        lines = list(LOG_RING.lines)
        return {"lines": lines[-n:]}

    @app.get("/api/catalog")
    def catalog() -> dict:
        from festival_pet.motion import GESTURES

        return {"sounds": list(sounds.EMOTIONS), "gestures": list(GESTURES), "moves": ["curious1", "welcoming1", "loving1", "dance1", "dance2", "dance3", "laughing1", "surprised1", "yes1", "no1", "sleep1", "cheerful1"]}

    @app.post("/api/control")
    def control(c: Control) -> dict:
        try:
            return pet.control(c.cmd, c.value)
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/forget")
    def forget() -> dict:
        return pet.control("forget", None)

    @app.get("/api/people/{person_id}/face.jpg")
    def face(person_id: int):
        from fastapi.responses import FileResponse

        path = pet.p.memory.thumbnail_path(person_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail="no face stored")
        return FileResponse(path, media_type="image/jpeg")

    @app.delete("/api/people/{person_id}")
    def delete_person(person_id: int) -> dict:
        try:
            pet.p.memory.delete(person_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no person #{person_id}")
        pet.p.memory.save(force=True)
        return {"ok": True}

    @app.post("/api/people/merge")
    def merge_people(m: Merge) -> dict:
        try:
            kept = pet.p.memory.merge(m.keep, m.other)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=f"no person {e}")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        pet.p.memory.save(force=True)
        return {"ok": True, "kept": kept.person_id, "embeddings": len(kept.embeddings)}


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
        logging.getLogger().addHandler(LOG_RING)
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
        install_routes(self.settings_app, pet)


if __name__ == "__main__":
    app = FestivalPetApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
