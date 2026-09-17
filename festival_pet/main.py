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

import json
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

from festival_pet import build_info, sounds
from festival_pet.audio_features import BeatTracker, LevelMeter, RubDetector, ScratchDetector, _tune
from festival_pet.behavior import Action, Behavior, FaceObs, Observation
from festival_pet.keypad import KeyMap, KeypadListener
from festival_pet.memory import FaceMemory
from festival_pet.motion import BODY_YAW_LIMIT, MotionComposer, turn_pose
from festival_pet.senses import LoudSoundDetector, PickupDetector, PoseHistory, SelfMotionGate, TouchDetector
from festival_pet.vision import Sighting
from festival_pet.tap_tempo import TapTempo
from festival_pet.visual_rhythm import DanceDetector

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
SETTINGS_FILE = DATA_DIR / "settings.json"

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
    def motor_mode(self) -> str: ...  # "enabled", "disabled" or "gravity_compensation", as the daemon reports it


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
        self._env: np.ndarray = np.zeros(0, dtype=np.float32)  # loudness envelope of the phrase being played, 50 Hz
        self._env_t0 = 0.0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    @property
    def busy_until(self) -> float:
        return self._busy_until

    def level(self, now: float) -> float:
        """Normalised loudness (0..1) of what the speaker is saying right now; 0 when silent."""
        i = int((now - self._env_t0) * 50)
        if i < 0 or i >= len(self._env):
            return 0.0
        return float(self._env[i])

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
            hop = self._sample_rate // 50
            n = len(buf) // hop
            env = np.sqrt(np.mean(buf[: n * hop].reshape(n, hop) ** 2, axis=1)) if n else np.zeros(0, np.float32)
            self._env = (env / max(1e-6, float(env.max()))).astype(np.float32) if n else env
            self._env_t0 = time.time() + 0.08  # roughly the output latency, so the sway lands with the sound
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
        self.deaf_until = 0.0  # the mics sit next to the speaker: ignore touch/voice events while we make noise
        self.own_sound_until: Callable[[], float] = lambda: 0.0
        self.enabled = True  # "ears off": drain the mic but do no processing (dead mics, or to save CPU)
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
            if not self.enabled:
                continue
            self.beat.push(chunk, now)
            scratched = self.scratch.push(chunk, now)
            rubbed = self.rub.push(chunk, now)
            self.meter.push(chunk, now, {"rub": self.rub.stats["rub_energy"], "flat": self.rub.stats["flatness"], "band": self.scratch.stats["band_energy"], "clicks": self.scratch.stats["clicks_in_window"]})
            self._calibration_step(now)
            deaf = now < self.deaf_until or now < self.own_sound_until() + 0.5
            self.stats["deaf"] = deaf
            # Our own beeps / the daemon's sleep and wake sounds must never count as pets or scratches.
            if scratched and not deaf:
                self._events.put(("scratch", "", None))
            if rubbed and not deaf:
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
        self.audio.own_sound_until = lambda: self.sound.busy_until
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
        self.tap = TapTempo()  # hand-tapped beat from the control page or a paired keypad
        self.keypad = KeypadListener()
        self.keymap = KeyMap()
        self.pose_history: PoseHistory | None = None  # set on the real robot; fed every tick for the vision thread
        self.manual_groove = False  # groove to the tapped beat instead of what it hears/sees
        self._dance_seen = False  # edge: seed the tap clock once per dance
        self.pickup_enabled = False  # IMU is in the head; off by default until tuned on the real robot
        self.asleep = False  # motors off, camera paused; ears stay on
        self._touch_settle_left = -1.0  # seconds of ticks to wait after sleep/wake before re-zeroing the ear detector
        self.set_vision_active: Callable[[bool], None] = lambda active: None
        self.transcript: deque[tuple[float, str, list[str]]] = deque(maxlen=30)
        self.dance = DanceDetector()
        self._last_rhythm_ts = 0.0
        self._last_seen_angles = (0.0, 0.0)
        self.face_history: deque[tuple[float, float, float, str, bool]] = deque(maxlen=200)  # ~20 s at face rate
        self.settings_file: Path | None = None
        self._last_obs = Observation()

    # ------------------------------------------------------------------ lifecycle
    def start(self, now: float) -> None:
        self.sound.start()
        self.audio.start()
        self.keypad.start()
        self.p.behavior.start(now, awake=True)  # the platform wakes the robot before launching an app
        self._dispatch(Action("wake", "start", 5), now)
        self._last = now

    def stop(self) -> None:
        self.keypad.stop()
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
        if self.pose_history is not None:
            self.pose_history.record(now, io.head_pose())  # measured pose, timestamped, for pairing with camera frames
        imu = io.imu()
        if imu is not None:
            held, shaken = self.pickup.update(imu["accelerometer"], imu["gyroscope"], now, self_moving)
            if self.pickup_enabled:
                obs.held, obs.shaken = held, shaken
        # Antennas lag their command while animated; the detector raises its threshold then.
        busy = self.move is not None or comp.gesture_active(now)
        present_ants = io.present_antennas()
        if self._touch_settle_left >= 0:
            # Count ticks, not wall time: the blocking sleep/wake move already ate seconds before this tick ran.
            self._touch_settle_left -= min(dt, 0.05)
            if self._touch_settle_left < 0:  # motors settled: wherever the ears rest now is "untouched"
                self.last_ants = list(present_ants)
                # Limp antennas can still creep; asleep we want a deliberate, held push before waking.
                self.touch = TouchDetector(press_rad=0.45, persist_ticks=15) if self.asleep else TouchDetector()
            obs.touched = False
        else:
            obs.touched = self.touch.update(self.last_ants, present_ants, busy, dt)
        obs.touched_side = self.touch.last_side
        obs.petting = self.audio.rub.rubbing
        if now >= self._next_doa:
            self._next_doa = now + 0.2
            deaf = now < self.audio.deaf_until or now < self.sound.busy_until + 0.5
            obs.loud_yaw_deg = None if deaf else self.loud.update(self.audio.doa, now)
        sighting = self.p.latest_sighting()
        if sighting is not None and now - sighting.ts < 1.0:
            seen = self._to_face_obs(sighting)
            if sighting.kind == "face":
                obs.face = seen
            else:
                obs.body = seen
            if sighting.ts != self._last_rhythm_ts:  # one rhythm sample per detection
                self._last_rhythm_ts = sighting.ts
                # World-frame angles: the camera is in the head, so image coordinates would measure our own bob.
                self._last_seen_angles = (seen.yaw_deg, seen.pitch_deg)
                self.dance.push(sighting.ts, seen.yaw_deg, seen.pitch_deg)
                self.face_history.append((sighting.ts, seen.yaw_deg, seen.pitch_deg, sighting.kind, self.dance.state.dancing))
        elif self.dance.state.dancing and now - self._last_rhythm_ts > 2.0:
            self.dance.push(now, *self._last_seen_angles)  # nobody in view: hold still, let the lock run out
        if self.dance.state.dancing:
            obs.dance_bpm = self.dance.state.bpm
            if not self._dance_seen:  # seen dancing: hand the tempo to the tap clock so manual groove / "1" pick it up
                self._dance_seen = True
                bpm = self.dance.state.bpm
                self.tap.set_bpm(bpm, beat_at=now - self.dance.phase(now) * 60.0 / bpm)
                self.actions_log.append((now, "tap", f"seeded {bpm:.0f} bpm from seeing them dance"))
        else:
            self._dance_seen = False
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

        # ---------------- keypad (a paired keyboard in someone's hand)
        fired = []
        while True:
            try:
                ev = self.keypad.events.get_nowait()
            except queue.Empty:
                break
            hit = self.keymap.feed(ev, now)
            if hit is not None:
                fired.append(hit)
        fired += self.keymap.tick(now)
        for action, t_ev in fired:
            self.key_action(action, t_ev, now)

        # ---------------- brain
        actions = beh.tick(obs, now, dt)
        comp.energy = beh.mood.energy
        comp.mode = {"SLEEPING": "sleeping", "HELD": "held"}.get(beh.state, "awake")
        comp.set_gaze(beh.gaze)
        comp.groove = None
        comp.mirror_roll = 0.0
        comp.mimic = None
        comp.voice_level = self.sound.level(now)  # the body moves with every beep it makes
        for act in actions:
            self._dispatch(act, now)
        if self.manual_groove and self.tap.active and not self.asleep and beh.state not in ("HELD", "SLEEPING", "WAKING"):
            # Tapped-in beat wins over anything it heard or saw; 0.6 is a plain head-bob before the user's dials.
            comp.groove = (self.tap.phase(now), self.tap.bar_phase(now), 0.6 * self.groove_scale)
            comp.groove_phrase = self.tap.phrase_phase(now) if self.tap.downbeat_known else None
        else:
            comp.groove_phrase = None

        # ---------------- body
        if self.move is not None:
            t = now - self.move_t0
            if t >= self.move.duration - 0.02:
                self.move = None
                self.blend_from = (self.last_pose, list(self.last_ants), now)
            else:
                head, ants, move_body = self.move.evaluate(t)
                # Moves are recorded body-forward: turn them with the body and add the move's own body swing.
                body = max(-BODY_YAW_LIMIT, min(BODY_YAW_LIMIT, comp.body_yaw + math.degrees(float(move_body))))
                head = turn_pose(head, body)
                self.last_pose, self.last_ants = head, [float(ants[0]), float(ants[1])]
                io.set_target(head, self.last_ants, math.radians(body))
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
        return FaceObs(s.track_id, float(yaw), float(pitch), s.area_frac, s.person, s.similarity, s.roll_deg, s.head_yaw_deg, s.head_pitch_deg)

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
                self.p.io.goto(turn_pose(head0, comp.body_yaw), [float(ants0[0]), float(ants0[1])], 0.4)
                self.move, self.move_t0 = move, time.time()
                if PLAY_LIBRARY_SOUNDS and move.sound_path is not None:
                    self.p.io.play_file(str(move.sound_path))
        elif act.kind == "wake":
            # Never trust the flag alone: the daemon boots asleep (--no-wake-up-on-start) and an app can be
            # launched into that, so a limp robot is woken whatever we think our state is.
            if self.asleep or self.p.io.motor_mode() != "enabled":
                self.audio.deaf_until = float("inf")
                try:
                    self.p.io.wake_body()  # daemon wake_up: motors on, lift; blocks ~2 s
                except Exception:
                    logger.exception("wake move failed; carrying on awake")
                self.audio.deaf_until = time.time() + 1.5
                self.asleep = False
                self.set_vision_active(True)
                comp.body_yaw = 0.0
                self.move, self.blend_from = None, None
                self._touch_settle_left = 1.5
            self.sound.request("wake", 5, now)
            comp.request_gesture("perk", now, 5)
        elif act.kind == "sleep":
            if not self.asleep:
                self.set_vision_active(False)  # face detection off while asleep; ears stay on for name / noise / pets
                self.move = None
                self.audio.deaf_until = float("inf")  # the sleep move plays a sound and the motors whirr right next to the mics
                try:
                    self.p.io.sleep_body()  # centre the body, then the daemon's own sleep move (ends limp); blocks ~6 s
                    self.asleep = True
                    comp.body_yaw = 0.0
                    self._touch_settle_left = 5.0
                    self.audio.deaf_until = time.time() + 3.0  # let the room settle before listening for a wake
                except Exception:
                    logger.exception("sleep move failed; staying awake")
                    self.audio.deaf_until = time.time() + 2.0
                    self.set_vision_active(True)
                    self.p.behavior.state, self.p.behavior._state_since = "IDLE", time.time()
        elif act.kind == "groove":
            beat = self.audio.beat
            intensity, _, source = act.name.partition("|")
            if source == "visual" and self.dance.state.dancing:  # dance along with what it sees
                period, t0 = 60.0 / self.dance.state.bpm, 0.0
                phase = self.dance.phase(now)
            elif beat.music:
                phase, period, t0 = beat.phase(now), beat.state.period, beat._last_beat_time
            else:  # no music: the little-dance trick bobs to its own inner tempo
                period, t0 = 60.0 / FREE_DANCE_BPM, 0.0
                phase = (now / period) % 1.0
            bar = ((now - t0) / (4 * period)) % 1.0
            comp.groove = (phase, bar, float(intensity) * self.groove_scale)
        elif act.kind == "mimic":
            y, p_, r = (float(x) for x in act.name.split(","))
            comp.mimic = (y, p_, r)
        elif act.kind == "mirror":
            comp.mirror_roll = float(act.name)
        elif act.kind == "heard":
            self.transcript.append((now, act.name, act.name.split("|")[1:]))

    def key_action(self, action: str, t_ev: float, now: float) -> None:
        """One keypad action. ``t_ev`` is the key's own timestamp: taps use it so USB/BT latency does not smear the beat."""
        beh, comp = self.p.behavior, self.p.composer
        beh._last_interaction = now  # someone is playing with it: not lonely
        self.actions_log.append((now, "key", action))
        if action == "tap":
            self.tap.tap(t_ev)
        elif action == "downbeat":
            self.tap.tap(t_ev, downbeat=True)
        elif action in ("tilt_left", "tilt_right"):
            comp.request_gesture("tilt", now, 3, side=1.0 if action == "tilt_left" else -1.0)
        elif action == "nod":
            comp.request_gesture("nod", now, 3, reps=2)
        elif action == "happy":
            self._dispatch(Action("sound", "happy", 3), now)
            comp.request_gesture("bounce", now, 3)
        elif action == "manual_groove":
            self.control("manual_groove", not self.manual_groove)
            self._dispatch(Action("sound", "happy" if self.manual_groove else "curious", 3), now)
        elif action in ("wake", "sleep"):
            self.control(action, None)
        elif action == "mute":
            self.control("mute", not self.muted)
        elif action == "none":
            pass
        else:
            raise KeyError(f"unknown key action '{action}'")

    def _feeling(self, now: float) -> dict:
        """The most recent gesture/move and sound, for the top of the Mind page."""
        out: dict = {"motion": None, "sound": None}
        for t, kind, name in reversed(self.actions_log):
            key = "motion" if kind in ("gesture", "move") else "sound" if kind == "sound" else None
            if key is not None and out[key] is None:
                out[key] = {"name": name, "t": round(now - t, 1)}
            if out["motion"] is not None and out["sound"] is not None:
                break
        return out

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
            "feeling": self._feeling(now),
            "build": build_info(),
            "asleep": self.asleep,
            "transcript": [{"t": round(now - t, 1), "text": txt.split("|")[0], "intents": ints} for t, txt, ints in reversed(self.transcript)],
            "senses": {
                "body": None if o.body is None else {"yaw": round(o.body.yaw_deg, 1), "pitch": round(o.body.pitch_deg, 1), "size": round(o.body.area_frac, 3)},
                "face": None if o.face is None else {"track": o.face.track_id, "yaw": round(o.face.yaw_deg, 1), "pitch": round(o.face.pitch_deg, 1), "size": round(o.face.area_frac, 3), "person": None if o.face.person is None else o.face.person.person_id, "similarity": round(o.face.similarity, 2), "tilt": round(o.face.roll_deg, 1)},
                "held": o.held, "shaken": o.shaken, "imu": self.pickup.stats, "head_rate": round(self.self_motion.rate, 2), "ears": self.touch.stats,
                "music": {"bpm": round(b.bpm, 1), "confidence": round(b.confidence, 2), "grooving": comp.groove is not None, "intensity": round(comp.groove[2], 2) if comp.groove else 0.0},
                "keypad": {"devices": list(self.keypad.devices.values()), "last_key": None if self.keypad.last_key is None else {"key": self.keypad.last_key[0], "t": round(now - self.keypad.last_key[1], 1)}, "error": self.keypad.error},
                "tap": {"bpm": round(self.tap.bpm, 1), "beat": self.tap.beat_in_bar(now) if self.tap.active else 0, "bar": self.tap.bar_in_phrase(now) if self.tap.active else 0, "downbeat_known": self.tap.downbeat_known},
                "dance": {"dancing": self.dance.state.dancing, "bpm": round(self.dance.state.bpm, 1), "confidence": round(self.dance.state.confidence, 2), "amplitude": round(self.dance.state.amplitude, 3), "holds_for_s": round(max(0.0, self.dance.locked_for - now), 1) if self.dance.state.dancing else 0.0},
                "mimic": None if comp.mimic is None else {"yaw": round(comp.mimic[0], 1), "pitch": round(comp.mimic[1], 1), "roll": round(comp.mimic[2], 1)},
                "listening": self.audio.stats["listening"], "voice_yaw": self.audio.last_voice_yaw,
                "doa": None if self.audio.doa is None else {"angle_deg": round(math.degrees(self.audio.doa[0]), 0), "speech": self.audio.doa[1]},
                "speech_s_ago": round(now - self.audio.last_speech_time, 1) if self.audio.last_speech_time > 0 else None,
                "levels": {"rms": round(self.audio.meter.rms, 5), "peak": round(self.audio.meter.peak, 4), "max_rms_3s": round(self.audio.meter.max_rms_3s, 5), "chunks": self.audio.stats["chunks"]},
                "scratch": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.audio.scratch.stats.items()},
                "head_pet": {"rubbing": self.audio.rub.rubbing, **{k: round(v, 6) for k, v in self.audio.rub.stats.items()}},
            },
            "controls": {"muted": self.muted, "pickup": self.pickup_enabled, "ears": self.audio.enabled, "mimic_flip": self.p.composer.mimic_flip, "body_finder": getattr(getattr(self, "vision", None), "body_enabled", None), "groove_scale": self.groove_scale, "manual_groove": self.manual_groove, "keymap": self.keymap.keys, "camera_lag_ms": None if self.pose_history is None else round(self.pose_history.lag_s * 1000), "bpm": round(self.tap.bpm, 1), **{f"groove_{k}": v for k, v in comp.groove_mix.as_dict().items()}, "scratch_onset_ratio": self.audio.scratch.onset_ratio, "scratch_floor": self.audio.scratch.floor, "rub_level_ratio": self.audio.rub.level_ratio, "rub_flatness_min": self.audio.rub.flatness_min, "rub_floor": self.audio.rub.floor, "match_threshold": self.p.memory.match_threshold},
            "calibration": self.audio.calibration_result,
            "face_history": [{"t": round(t - now, 2), "yaw": round(y, 1), "pitch": round(p_, 1), "kind": k, "dancing": d} for t, y, p_, k, d in self.face_history if now - t <= 20.0],
            "dance_params": {"min_amp": self.dance._min_amp, "min_conf": self.dance._min_conf},
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
        elif cmd == "ears":
            self.audio.enabled = bool(value)
        elif cmd == "mimic_flip":
            self.p.composer.mimic_flip = bool(value)
        elif cmd == "groove_scale":
            self.groove_scale = float(value)
        elif cmd == "manual_groove":
            self.manual_groove = bool(value)
        elif cmd == "tap":  # value: true = this tap is the "1"
            self.tap.tap(now, downbeat=bool(value))
            self.actions_log.append((now, "tap", f"{'ONE ' if value else ''}{self.tap.bpm:.0f} bpm"))
        elif cmd == "keymap":  # "KEY:press_action:hold_action"
            key, press, hold = str(value).split(":")
            self.keymap.set(key, press, hold)
        elif cmd == "key":  # fire a key action as if pressed (testing from the page)
            self.key_action(str(value), now, now)
        elif cmd == "preview":
            vision = getattr(self, "vision", None)
            if vision is None:
                raise KeyError("no vision on this pet")
            vision.preview = bool(value)
        elif cmd == "capture_face":
            vision = getattr(self, "vision", None)
            if vision is None:
                raise KeyError("no vision on this pet")
            if self._last_obs.face is None:
                raise ValueError("no face in view to capture")
            vision.capture_now = True
            self.actions_log.append((now, "capture", "another view of the face"))
        elif cmd == "camera_lag_ms":
            if self.pose_history is None:
                raise KeyError("no camera on this pet")
            self.pose_history.lag_s = float(value) / 1000.0
        elif cmd == "bpm":
            if float(value) == 0.0:
                self.tap.clear()
            else:
                self.tap.set_bpm(float(value))
        elif cmd in ("groove_bob", "groove_sway", "groove_body", "groove_ears"):
            setattr(self.p.composer.groove_mix, cmd[len("groove_"):], float(value))
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
        self.save_settings()
        return {"ok": True}

    # ------------------------------------------------------------------ settings persistence
    _SETTING_KEYS = ("muted", "pickup", "ears", "mimic_flip", "groove_scale", "manual_groove", "keymap", "camera_lag_ms", "groove_bob", "groove_sway", "groove_body", "groove_ears", "scratch_onset_ratio", "scratch_floor", "rub_level_ratio", "rub_flatness_min", "rub_floor", "match_threshold", "body_finder")

    def _settings(self) -> dict:
        c = self.mind()["controls"]
        return {k: c[k] for k in self._SETTING_KEYS if k in c and c[k] is not None}

    def save_settings(self) -> None:
        if self.settings_file is None:
            return
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.settings_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._settings(), indent=1))
        tmp.replace(self.settings_file)

    def load_settings(self) -> None:
        if self.settings_file is None or not self.settings_file.exists():
            return
        data = json.loads(self.settings_file.read_text())
        names = {"muted": "mute"}  # setting key -> control name where they differ
        for k, v in data.items():
            if k not in self._SETTING_KEYS:
                continue
            if k == "body_finder" and getattr(self, "vision", None) is None:
                continue
            if k == "camera_lag_ms" and self.pose_history is None:
                continue
            if k == "keymap":
                self.keymap.keys.clear()
                for key, m in v.items():
                    self.control("keymap", f"{key}:{m['press']}:{m['hold']}")
                continue
            self.control(names.get(k, k), v)
        logger.info("settings restored from %s: %s", self.settings_file, data)


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

    # The daemon owns the canonical sleep/wake moves (the same ones the dashboard uses):
    # goto_sleep quiesces tracking, lifts to neutral, plays the sound, nests the head and
    # ends with the motors limp; wake_up re-enables them and lifts. We call those over REST
    # and wait for the move task to finish.
    def _daemon_move(self, name: str, timeout: float) -> None:
        import requests

        base = self._r._daemon_http_url
        uuid = requests.post(f"{base}/api/move/play/{name}", timeout=10).json()["uuid"]
        deadline = time.time() + timeout
        while time.time() < deadline:
            running = [m["uuid"] for m in requests.get(f"{base}/api/move/running", timeout=5).json()]
            if uuid not in running:
                return
            time.sleep(0.2)
        raise TimeoutError(f"daemon move {name} did not finish within {timeout}s")

    def motor_mode(self) -> str:
        import requests

        return requests.get(f"{self._r._daemon_http_url}/api/motors/status", timeout=5).json()["mode"]

    def sleep_body(self):
        from reachy_mini.reachy_mini import INIT_ANTENNAS_JOINT_POSITIONS, INIT_HEAD_POSE

        # Centre the body first: the daemon's sleep only moves the head, and a turned body leaves it nesting sideways.
        self._r.goto_target(head=INIT_HEAD_POSE, antennas=INIT_ANTENNAS_JOINT_POSITIONS, duration=1.2, body_yaw=0.0)
        self._daemon_move("goto_sleep", timeout=15.0)  # the dashboard's sleep: nest, then motors limp
        mode = self.motor_mode()
        if mode != "disabled":
            logger.warning("daemon sleep left motors '%s'; disabling them explicitly", mode)
            self._r.disable_motors()
        logger.info("asleep: motor mode %s", self.motor_mode())

    def wake_body(self):
        # A disabled robot ignores wake_up (reachy_mini issue #1306): torque must come back first.
        self._r.enable_motors()  # pins targets to the present pose, so nothing snaps
        self._daemon_move("wake_up", timeout=10.0)
        logger.info("awake: motor mode %s", self.motor_mode())

    # --- things the sunset dashboard used to do
    def get_volume(self) -> int:
        import requests

        return requests.get(f"{self._r._daemon_http_url}/api/volume/current", timeout=5).json()["volume"]

    def set_volume(self, volume: int) -> int:
        import requests

        r = requests.post(f"{self._r._daemon_http_url}/api/volume/set", json={"volume": int(volume)}, timeout=10)
        r.raise_for_status()
        return r.json()["volume"]

    def power(self, action: str) -> str:
        """'shutdown' or 'reboot' the robot.

        The wireless image's sudoers (/etc/sudoers.d/010-pollen-reachy) allows exact commands only; the
        robot's own power button runs ``sudo shutdown -h now``, so we use that exact form (and -r for reboot).
        """
        import subprocess

        cmd = {"shutdown": ["sudo", "shutdown", "-h", "now"], "reboot": ["sudo", "shutdown", "-r", "now"]}[action]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20, stdin=subprocess.DEVNULL)
        if proc.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd)} failed ({proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}")
        return f"{action} requested"


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
    poses = PoseHistory(reachy.get_current_head_pose)
    vision = Vision(YUNET_MODEL, SFACE_MODEL, memory, reachy.media.get_frame, poses.lagged,
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
    pet.pose_history = poses
    pet.settings_file = SETTINGS_FILE
    pet.load_settings()
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

    @app.get("/api/bt")
    def bt() -> dict:
        from festival_pet import keypad

        try:
            return keypad.bt_status()
        except FileNotFoundError:
            raise HTTPException(status_code=501, detail="bluetoothctl is not installed on this robot (apt install bluez while online)")

    @app.post("/api/bt/scan")
    def bt_scan() -> dict:
        from festival_pet import keypad

        return {"devices": keypad.bt_scan()}

    @app.post("/api/bt/pair")
    def bt_pair(c: Control) -> dict:
        from festival_pet import keypad

        mac, _, pin = str(c.value).partition("|")
        try:
            return {"detail": keypad.bt_pair(mac, pin or "1234")}
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/bt/forget")
    def bt_forget(c: Control) -> dict:
        from festival_pet import keypad

        keypad.bt_forget(str(c.value))
        return {"ok": True}

    @app.post("/api/forget")
    def forget() -> dict:
        return pet.control("forget", None)

    @app.get("/api/volume")
    def volume() -> dict:
        io = pet.p.io
        if not hasattr(io, "get_volume"):
            raise HTTPException(status_code=501, detail="no volume control on this robot IO")
        return {"volume": io.get_volume()}

    @app.post("/api/volume")
    def set_volume(c: Control) -> dict:
        io = pet.p.io
        if not hasattr(io, "set_volume"):
            raise HTTPException(status_code=501, detail="no volume control on this robot IO")
        return {"volume": io.set_volume(int(float(c.value)))}

    @app.post("/api/power/{action}")
    def power(action: str) -> dict:
        io = pet.p.io
        if action not in ("shutdown", "reboot"):
            raise HTTPException(status_code=400, detail="action must be shutdown or reboot")
        if not hasattr(io, "power"):
            raise HTTPException(status_code=501, detail="no power control on this robot IO")
        try:
            if action == "shutdown" and not pet.asleep:
                pet._dispatch(Action("sleep", "shutdown", 5), time.time())  # nest first so the head is not left standing
            return {"ok": True, "detail": io.power(action)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/camera.jpg")
    def camera():
        from fastapi.responses import Response

        vision = getattr(pet, "vision", None)
        jpeg = None if vision is None else vision.last_jpeg
        if jpeg is None:
            raise HTTPException(status_code=404, detail="preview is off (or no frame yet)")
        return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

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
        logger.info("festival_pet %s", build_info())
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
