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
from festival_pet.keypad import LAYERS, KeyMap, KeypadListener
from festival_pet.memory import FaceMemory
from festival_pet.mime import MimeGame
from festival_pet.motion import BODY_YAW_LIMIT, MotionComposer, turn_pose, SOLO_GESTURES
from festival_pet import songs
from festival_pet.senses import ImuRubDetector, LoudSoundDetector, PickupDetector, PoseHistory, SelfMotionGate, TouchDetector
from festival_pet.vision import Sighting
from festival_pet.pose import ArmSigns
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
POSE_MODEL = DATA_DIR / "models" / "pose_estimation_mediapipe_2023mar.onnx"
MEMORY_FILE = DATA_DIR / "memory.json"
SETTINGS_FILE = DATA_DIR / "settings.json"
SONGS_FILE = DATA_DIR / "songs.json"

CONTROL_HZ = 50.0
PLAY_LIBRARY_SOUNDS = False  # True = play Pollen's sidecar sound with library moves instead of our beeps
MOVE_BLEND_S = 0.5
FREE_DANCE_BPM = 108.0
AUDIO_RATE = 16000
ARMS_MAX_AGE_S = 0.6  # an arm read older than this is nobody's arms
# Keypad: the dancing layer's nudges and the caring layer's snacks and mushrooms
NUDGE_WINDOW_S = 2.5  # direction taps this close together count up
NUDGE_TURN_TAPS = 3  # that many in a row and the body turns to dance that way
NUDGE_TURN_DEG = 60.0
NUDGE_TURN_BEATS = 8  # how long it dances facing that way before coming back to whoever it was with
SNACK_WINDOW_S = 15.0
SNACK_HICCUP_AT = 5  # snacks this close together: hiccups
SNACK_ACHE_AT = 9  # tummy ache: no more snacks for a while
SNACK_ACHE_S = 60.0
SNACK_ENERGY = 0.04
TRIP_S = 120.0  # a mushroom: dizzy first, then bouncy, curious and grooving harder for this long
TRIP_MORE_S = 60.0  # a second one during the trip
TRIP_ENERGY, TRIP_CRASH = 0.3, 0.15
TRIP_GROOVE = 1.5
BOOP_WINDOW_S = 3.0
BOOP_DANCE_AT, BOOP_BOW_AT = 7, 13  # boop it that many times in a row: a little dance; a bow to the house
FLOURISH_EVERY_BEATS = 8  # dance-along: every two bars it stops copying and throws in two beats of its own
FLOURISH_BEATS = 2
COMBO_TAPS = 4  # left right left right on the dancing layer...
COMBO_WINDOW_S = 1.0  # ...this fast: stop dancing


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
        self._q: queue.Queue[tuple[int, str, np.ndarray | None]] = queue.Queue()
        self._busy_until = 0.0
        self._busy_priority = 0
        self._sample_rate = sample_rate
        self._stop = threading.Event()
        self._cut = threading.Event()  # set by cut(): drop the rest of what is playing
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

    def cut(self) -> None:
        """Stop what is playing (a song, mid-verse) and forget anything queued behind it."""
        self._cut.set()
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._busy_until = 0.0
        self._env = np.zeros(0, dtype=np.float32)

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
        self._q.put((priority, emotion, None))

    def request_buffer(self, buf: np.ndarray, label: str, priority: int, now: float) -> None:
        """Play a pre-rendered buffer (a song) under the same priority rule."""
        if now < self._busy_until and self._busy_priority >= priority:
            return
        self._q.put((priority, label, buf))

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                priority, emotion, given = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            buf = given if given is not None else sounds.render_phrase(emotion, sample_rate=self._sample_rate)
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
            self._cut.clear()
            t_end = time.time() + dur
            for i in range(0, len(buf), chunk):
                if self._cut.is_set():
                    break
                self._io.play(buf[i : i + chunk])
            while time.time() < t_end and not self._cut.is_set():
                time.sleep(0.02)


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
        self.imu_rub = ImuRubDetector()
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
        self.voice = "full"  # "full" | "quiet" (about half as chatty: no idle noises, half the small ones) | "off"
        self.groove_scale = 1.0  # user knob on top of the brain's intensity
        self.tap = TapTempo()  # hand-tapped beat from the control page or a paired keypad
        self.keypad = KeypadListener()
        self.keymap = KeyMap()
        self.mime = MimeGame()
        self.singing_enabled = False  # one of its idle activities when on
        self.songs: list[dict] = []  # the repertoire (saved songs)
        self.last_song: dict | None = None
        self._singing_until = 0.0
        self.songs_file: Path | None = None
        self.pose_history: PoseHistory | None = None  # set on the real robot; fed every tick for the vision thread
        # Actions asked for from the web page / a keypad. They are run by the control loop, never on the
        # HTTP thread: sleep and wake block for seconds on daemon moves, and a loop still streaming the awake
        # pose meanwhile yanks the antennas back up the moment the nest finishes.
        self._from_page: queue.Queue[Action] = queue.Queue()
        self.manual_groove = False  # groove to the tapped beat instead of what it hears/sees
        self._dance_seen = False  # edge: seed the tap clock once per dance
        self.pickup_enabled = False  # "held in hand": the body never turns; it asks to be turned instead
        self._short_since = 0.0
        self._next_ask = 0.0
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
        # Dance-along: while someone dances (or the tapped clock runs) and their arms can be read, the antennas
        # copy their arms live, with a flourish of its own every couple of bars.
        self.dance_along = True
        self._copying_arms = False
        self._flourish_until = 0.0
        self._flourish_len = 1.0
        self._flourish_riff = 0
        self._beats = 0
        self._last_phase = 0.0
        self._copy_last = 0.0
        self.signs = ArmSigns()  # waves and hugs, read from the arm readings
        self._signs_last = 0.0  # ts of the last arm reading fed to it (each reading counts once)
        # Keypad state: direction nudges, snacks, the trip, boops (see key_action)
        self._nudge: list[tuple[float, float]] = []  # (time, side) of recent direction taps
        self._combo: list[tuple[float, float]] = []  # (time, side) of ALL recent direction taps, for the stop combo
        self._turn: tuple[float, float] | None = None  # (world yaw to dance facing, until) after repeated nudges
        self._snacks: list[float] = []
        self._ache_until = 0.0
        self._trip_until = 0.0
        self._trip_doses = 0
        self._boops: list[float] = []

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

        # ---------------- what the page asked for (on this thread, so nothing else moves the robot meanwhile)
        while True:
            try:
                act = self._from_page.get_nowait()
            except queue.Empty:
                break
            self._dispatch(act, now)

        # ---------------- senses
        obs = Observation()
        # The IMU is in the head: ignore it while we are the ones moving the head.
        self_moving = self.self_motion.update(self.last_pose, now)
        if self.pose_history is not None:
            self.pose_history.record(now, io.head_pose())  # measured pose, timestamped, for pairing with camera frames
        imu = io.imu()
        if imu is not None:
            self.pickup.update(imu["accelerometer"], imu["gyroscope"], now, self_moving)  # stats only; "held" is the switch below
            if self.imu_rub.update(self.pickup.stats["gyro"], self_moving, now):
                obs.petted = True  # a hand just started rubbing the head (felt through the IMU: works with ears off)
            if self.imu_rub.calibration is not None and self.imu_rub.calibration["phase"] not in ("done", "failed"):
                if self.imu_rub.calibration_step(self.pickup.stats["gyro"], self_moving, now):
                    self.save_settings()  # the floor it found is a setting
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
        obs.petting = self.audio.rub.rubbing or self.imu_rub.rubbing
        comp.petted = obs.petting
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
        vision = getattr(self, "vision", None)
        if vision is not None:
            arms = vision.latest_arms()
            if arms is not None and now - arms.ts <= ARMS_MAX_AGE_S:
                obs.arms = arms
                # waves and hugs, but not while raised arms mean something else (a game move, dancing)
                if arms.ts != self._signs_last and not self.mime.active and not self._copying_arms:
                    self._signs_last = arms.ts
                    sign = self.signs.feed(arms, now)
                    if sign is not None:
                        self.actions_log.append((now, "arms", " ".join(sign)))
                        if sign[0] == "wave":
                            obs.waved = sign[1]
                        else:
                            obs.hugged = True
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
        for action, t_ev in fired:
            self.key_action(action, t_ev, now)
        if self._trip_until and now >= self._trip_until:  # the mushroom wears off: the crash
            self._trip_until, self._trip_doses = 0.0, 0
            beh.mood.energy -= TRIP_CRASH
            beh.mood.clamp()
            beh._think(now, "...and I'm back. that was a lot. tired now")
            self._dispatch(Action("sound", "yawn", 2), now)
            self._dispatch(Action("gesture", "droop", 2), now)

        # ---------------- Simon says (leads; the brain's own games and reactions wait)
        if self.mime.active:
            aim = obs.face if obs.face is not None else obs.body  # the arm game only needs to know where they are
            for item in self.mime.tick(aim, now, obs.arms):
                self._mime_action(item, now)
            beh._next_react = max(beh._next_react, now + 3.0)
            if beh.mimicking:
                beh.mimicking = False
            comp.body_follow = False  # the body stays where it is: a recentre swung the camera off their face
        elif not comp.body_follow and not comp.held:
            comp.body_follow = True

        # ---------------- singing (the brain chooses it; we perform it)
        if now < self._singing_until:
            song = self.last_song
            comp.groove = (self.tap.phase(now), self.tap.bar_phase(now), 0.7 * self.groove_scale) if song else comp.groove
            beh._next_react = max(beh._next_react, now + 2.0)
            if now >= self._singing_until - 0.02:
                comp.request_gesture("bow", now, 4)  # take a bow
                beh._think(now, "thank you, thank you")

        # ---------------- brain
        solo = comp._gesture is not None and comp._gesture.name in SOLO_GESTURES and comp.gesture_active(now)
        if self._turn is not None and (now >= self._turn[1] or comp.held or self.asleep):
            self._turn = None  # the keypad turn is over: back to whoever it was with, mid-groove, no fuss
        obs.busy = "mime" if self.mime.active else "sing" if now < self._singing_until else "gesture" if solo else "turn" if self._turn else None
        # Manual groove with a tempo in: we are dancing. The brain treats it like a beat (no games, songs, mirror
        # or nod-copying start) and the tapped beat below wins over anything it heard or saw.
        grooving = self.manual_groove and self.tap.active and not self.asleep and beh.state not in ("HELD", "SLEEPING", "WAKING")
        obs.grooving = grooving
        beh.can_sing = self.singing_enabled and not self.asleep and self.move is None and not grooving
        actions = beh.tick(obs, now, dt)
        comp.energy = beh.mood.energy
        comp.mode = {"SLEEPING": "sleeping", "HELD": "held"}.get(beh.state, "awake")
        if self._turn is not None:
            comp.set_gaze((self._turn[0], beh.gaze[1] if beh.gaze is not None else 0.0))  # dancing that way for a few beats
        else:
            comp.set_gaze(beh.gaze)
        comp.groove = None
        comp.mirror_roll = 0.0
        comp.mimic = None
        comp.voice_level = self.sound.level(now)  # the body moves with every beep it makes
        for act in actions:
            self._dispatch(act, now)
        if grooving:
            # 0.6 is a plain head-bob before the user's dials.
            comp.groove = (self.tap.phase(now), self.tap.bar_phase(now), 0.6 * self.groove_scale)
            comp.groove_phrase = self.tap.phrase_phase(now) if self.tap.downbeat_known else None
        else:
            comp.groove_phrase = None
        if comp.groove is not None and now < self._trip_until:
            comp.groove = (comp.groove[0], comp.groove[1], comp.groove[2] * TRIP_GROOVE)  # tripping: everything grooves harder
        self._dance_along(obs, now)
        if vision is not None:
            vision.pose_live = self._copying_arms or (self.mime.active and self.mime.kind == "arms") or (self.signs.watching and now < self.signs.watching_until)

        # ---------------- body
        if self.move is not None:
            t = now - self.move_t0
            if t >= self.move.duration - 0.02:
                self.move = None
                self.blend_from = (self.last_pose, list(self.last_ants), now)
            else:
                head, ants, move_body = self.move.evaluate(t)
                # Moves are recorded body-forward: turn them with the body and add the move's own body swing.
                body = 0.0 if comp.held else max(-BODY_YAW_LIMIT, min(BODY_YAW_LIMIT, comp.body_yaw + math.degrees(float(move_body))))
                head = turn_pose(head, body)
                self.last_pose, self.last_ants = head, [float(ants[0]), float(ants[1])]
                if not self.asleep:
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
        # Held in hand and straining to look somewhere the head cannot reach: ask to be turned that way.
        if comp.held and not self.asleep and abs(comp.yaw_short) > 8.0 and self.move is None:
            if self._short_since == 0.0:
                self._short_since = now
            elif now - self._short_since > 1.0 and now >= self._next_ask:
                self._next_ask = now + 7.0
                side = "+" if comp.yaw_short > 0 else "-"
                beh._think(now, f"I can't see that far {'left' if side == '+' else 'right'}... turn me!")
                self._dispatch(Action("sound", "huff", 3), now)
                comp.request_gesture("point", now, 3, side=1.0 if side == "+" else -1.0)
                self.actions_log.append((now, "ask", "turn me " + ("left" if side == "+" else "right")))
        else:
            self._short_since = 0.0
        self.p.memory.save()

    # ------------------------------------------------------------------ dance-along
    def _dance_along(self, obs: Observation, now: float) -> None:
        """Copy a dancer's arms with the antennas, to the beat, with a riff of its own every couple of bars.

        On while there is a beat to dance to (someone seen dancing, or the tapped clock with manual groove on,
        or music) and their arms are readable; off during Simon says, a library move, sleep or being held.
        """
        comp, beh = self.p.composer, self.p.behavior
        allowed = (self.dance_along and comp.groove is not None and not self.mime.active and self.move is None
                   and not self.asleep and beh.state not in ("HELD", "SLEEPING", "WAKING"))
        if not allowed or obs.arms is None:
            # a missed arm read or two is not the end; anything else stops it at once
            if self._copying_arms and (not allowed or now - self._copy_last > 1.0):
                self._copying_arms = False
                comp.show_arms(None)
                self.actions_log.append((now, "arms", "stopped copying"))
            return
        self._copy_last = now
        if not self._copying_arms:
            self._copying_arms = True
            self._flourish_until, self._beats, self._last_phase = 0.0, 0, comp.groove[0]
            beh._think(now, "dancing with you: my antennas are your arms")
            self.actions_log.append((now, "arms", "copying"))
        # beats are counted where the groove's phase wraps, whatever clock is driving it (tapped, heard or seen)
        phase = comp.groove[0]
        if phase < self._last_phase - 0.5:
            self._beats += 1
            if now >= self._flourish_until and self._beats % FLOURISH_EVERY_BEATS == 0:
                period = self.tap.groove_period if self.tap.active else 60.0 / (self.dance.state.bpm if self.dance.state.dancing else FREE_DANCE_BPM)
                self._flourish_len = FLOURISH_BEATS * period
                self._flourish_until = now + self._flourish_len
                self._flourish_riff = comp.rng.randrange(3)
                beh._think(now, "my turn! a little something of my own")
                self.actions_log.append((now, "arms", f"flourish {self._flourish_riff}"))
        self._last_phase = phase
        if now < self._flourish_until:
            u = 1.0 - (self._flourish_until - now) / self._flourish_len  # 0..1 over the flourish
            half = int(u * FLOURISH_BEATS * 2) % 2  # flips every half beat
            if self._flourish_riff == 0:  # alternate: one up, one out, swapping on the half beat
                comp.show_arms(180.0 if half else 90.0, 90.0 if half else 180.0, now, 0.5)
            elif self._flourish_riff == 1:  # both pump up and out together
                comp.show_arms(180.0 if half else 90.0, 180.0 if half else 90.0, now, 0.5)
            else:  # a wave: one goes up as the other comes down
                comp.show_arms(180.0 * u, 180.0 * (1 - u), now, 0.5)
            return
        a = obs.arms
        left, right = (a.right_deg, a.left_deg) if comp.mimic_flip else (a.left_deg, a.right_deg)  # mirror: their right is our left
        comp.show_arms(left, right, now, 0.5)

    # ------------------------------------------------------------------ helpers
    def _to_face_obs(self, s: Sighting) -> FaceObs:
        from reachy_mini.vision.look_at import look_at_image_pose

        io = self.p.io
        pose = look_at_image_pose(s.u, s.v, io.K, io.D, s.head_pose_at_capture, io.T_head_cam)
        roll, pitch, yaw = R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
        return FaceObs(s.track_id, float(yaw), float(pitch), s.area_frac, s.person, s.similarity, s.roll_deg, s.head_yaw_deg, s.head_pitch_deg, s.smile)

    def _dispatch(self, act: Action, now: float) -> None:
        comp = self.p.composer
        if act.kind not in ("groove", "mirror"):
            self.actions_log.append((now, act.kind, act.name))
            del self.actions_log[:-2000]
        if act.kind == "sound":
            if self.voice == "full" or (self.voice == "quiet" and (act.priority >= 3 or (act.priority == 2 and self.p.composer.rng.random() < 0.5))):
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
        elif act.kind == "ears":
            what, _, which = act.name.partition(":")
            if what == "away":
                comp.ears_away(int(which), now)
            elif what == "tuck":
                comp.ears_tuck(int(which), now)
            elif what == "clear":
                comp.ears_clear()
            else:
                raise KeyError(f"unknown ears action '{act.name}'")
        elif act.kind == "activity":
            # The brain decided what to do; the games and songs it cannot run itself start here.
            if act.name == "sing":
                self.sing(now)
            elif act.name == "mime":
                self.start_simon(None, now)
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

    # ------------------------------------------------------------------ singing
    def sing(self, now: float, song: dict | None = None) -> dict:
        """Sing ``song``, or a saved one (half the time, if any), or make one up. Returns the song."""
        rng = self.p.composer.rng
        if song is None:
            song = rng.choice(self.songs) if self.songs and rng.random() < 0.5 else songs.compose(rng)
        buf = songs.render(song)
        dur = songs.duration(song)
        self.sound.request_buffer(buf, "song:" + song["name"], 2, now)
        self.last_song = song
        self._singing_until = now + dur
        self.tap.set_bpm(song["bpm"], beat_at=now + 0.08)  # bob along; the page shows the tempo too
        self.audio.deaf_until = max(self.audio.deaf_until, now + dur + 0.5)
        self.p.behavior._think(now, "a song! " + songs.describe(song))
        self.actions_log.append((now, "song", song["name"]))
        return song

    def save_last_song(self) -> dict:
        if self.last_song is None:
            raise ValueError("it has not sung anything yet")
        if self.last_song not in self.songs:
            self.songs.append(self.last_song)
            if self.songs_file is not None:
                self.songs_file.parent.mkdir(parents=True, exist_ok=True)
                self.songs_file.write_text(json.dumps(self.songs, indent=1))
        return self.last_song

    def load_songs(self) -> None:
        if self.songs_file is not None and self.songs_file.exists():
            self.songs = json.loads(self.songs_file.read_text())

    def start_simon(self, kind: str | None, now: float) -> str | None:
        """Start Simon says. ``kind`` "arms" or "head", or None to pick: the arm game when their arms can be
        read, else the head game when there is a face (the close-up game: a face that fills the frame has no
        arms in it). Returns the kind started, or None when there is nobody to play with."""
        o = self._last_obs
        if kind is None:
            kind = "arms" if o.arms is not None else "head" if o.face is not None else None
        elif kind == "arms" and o.arms is None:
            kind = "head" if o.face is not None else None
            if kind is not None:
                self.p.behavior._think(now, "can't see your arms from here: the head game instead")
        elif kind == "head" and o.face is None:
            kind = None
        if kind is None:
            return None
        self.mime.mirror_image = self.p.composer.mimic_flip
        for item in self.mime.start(now, kind=kind):
            self._mime_action(item, now)
        return kind

    def _mime_action(self, item: tuple, now: float) -> None:
        kind = item[0]
        comp = self.p.composer
        if kind == "sound":
            self._dispatch(Action("sound", item[1], 3), now)
        elif kind == "gesture":
            comp.request_gesture(item[1], now, 3)
        elif kind == "hold":
            comp.hold = None if item[1] is None else (item[1][0], item[1][1], item[1][2], now + item[1][3])
        elif kind == "arms":
            if item[1] is None:
                comp.show_arms(None)
            else:
                comp.show_arms(item[1][0], item[1][1], now, item[1][2])
        elif kind == "capture":
            vision = getattr(self, "vision", None)
            if vision is not None:
                vision.capture_now = True
        elif kind == "clock":
            if item[1] is None:
                comp.ears_clear()
            else:
                i, frac = item[1], item[2]
                comp.ear_clock(i, frac, now)
        elif kind == "think":
            self.p.behavior._think(now, item[1])
        else:
            raise KeyError(f"unknown mime action {item!r}")
        self.p.behavior._last_interaction = now

    def key_action(self, action: str, t_ev: float, now: float) -> None:
        """One keypad action (keypad.LAYERS). ``t_ev`` is the key's own timestamp: taps use it so USB/BT latency
        does not smear the beat. Runs inside ``step`` after the senses, so a caring / petting key can add to this
        tick's observation and the brain reacts as it would to the real thing."""
        beh, comp, obs = self.p.behavior, self.p.composer, self._last_obs
        beh._last_interaction = now  # someone is playing with it: not lonely
        self.actions_log.append((now, "key", action))
        if action in LAYERS["dancing"]:
            if not self.manual_groove:  # the first press on this layer turns manual groove on
                self.control("manual_groove", True)
                self._dispatch(Action("sound", "happy", 3), now)
            if action == "tap":
                self.tap.tap(t_ev)
            elif action == "downbeat":
                self.tap.tap(t_ev, downbeat=True)
            else:
                side = 1.0 if action == "groove_left" else -1.0
                self._combo = [(t, s) for t, s in self._combo if now - t <= COMBO_WINDOW_S] + [(now, side)]
                if len(self._combo) >= COMBO_TAPS and all(a[1] == -b[1] for a, b in zip(self._combo[-COMBO_TAPS:], self._combo[-COMBO_TAPS + 1:])):
                    # left right left right, fast: that's enough dancing
                    self._combo.clear()
                    self._nudge.clear()
                    self._turn = None
                    comp.groove_lean = 0.0
                    self.control("manual_groove", False)
                    self.tap.clear()
                    beh._think(now, "okay, okay! done dancing. phew")
                    self._dispatch(Action("sound", "content", 3), now)
                    self._dispatch(Action("gesture", "shake_off", 3), now)
                    return
                self._nudge_groove(side, now)
        elif action == "snack":
            self._snack(now)
        elif action == "mushroom":
            self._mushroom(now)
        elif action in ("pet", "head_pat"):
            obs.petted = True  # a virtual head pat: the brain leans in and purrs, and remembers who
        elif action == "boop":
            self._boop(now)
        elif action == "chin_scratch":
            beh._think(now, "chin scratches... mmm")
            self._dispatch(Action("sound", "content", 3), now)
            self._dispatch(Action("gesture", "snuggle", 3), now)
            beh.mood.social += 0.05
            beh.mood.clamp()
            if beh._engaged_person is not None:
                beh.memory.add_pet(beh._engaged_person)
        elif action == "ear_rub":
            obs.touched, obs.touched_side, obs.petting = True, comp.rng.randrange(2), True  # an ear massage, not a tickle
        elif action == "belly_rub":
            obs.scratched = True
        else:
            raise KeyError(f"unknown key action '{action}'")

    def _nudge_groove(self, side: float, now: float) -> None:
        """A direction tap on the dancing layer: guidance, not a puppet string. One tap leans the groove that way
        (head tilt and antennas); NUDGE_TURN_TAPS taps in a row and the body turns to dance facing that way for
        NUDGE_TURN_BEATS, then comes back to whoever it was with. Never turns when held in a hand."""
        beh, comp = self.p.behavior, self.p.composer
        self._nudge = [(t, s) for t, s in self._nudge if now - t <= NUDGE_WINDOW_S and s == side] + [(now, side)]
        comp.groove_lean = side
        if len(self._nudge) < NUDGE_TURN_TAPS or comp.held:
            return
        self._nudge.clear()
        period = self.tap.groove_period if self.tap.active else 60.0 / FREE_DANCE_BPM
        here = self._turn[0] if self._turn is not None else (beh.gaze[0] if beh.gaze is not None else comp.body_yaw)
        yaw = max(-BODY_YAW_LIMIT, min(BODY_YAW_LIMIT, here + side * NUDGE_TURN_DEG))
        self._turn = (yaw, now + NUDGE_TURN_BEATS * period)
        beh._think(now, f"okay okay, dancing over to the {'left' if side > 0 else 'right'} for a bit... back in {NUDGE_TURN_BEATS} beats")
        self._dispatch(Action("sound", "excited", 2), now)

    def _snack(self, now: float) -> None:
        beh = self.p.behavior
        if now < self._ache_until:
            beh._think(now, f"no more snacks. tummy hurts. ({self._ache_until - now:.0f} s)")
            self._dispatch(Action("sound", "huff", 3), now)
            self.p.composer.request_gesture("shake", now, 3, reps=2)
            return
        self._snacks = [t for t in self._snacks if now - t <= SNACK_WINDOW_S] + [now]
        n = len(self._snacks)
        beh.mood.energy += SNACK_ENERGY
        beh.mood.clamp()
        if n >= SNACK_ACHE_AT:
            self._snacks.clear()
            self._ache_until = now + SNACK_ACHE_S
            beh._think(now, "urgh... too many snacks. tummy ache. no more for a minute")
            self._dispatch(Action("sound", "sad", 3), now)
            self.p.composer.request_gesture("droop", now, 3)
        elif n >= SNACK_HICCUP_AT:
            beh._think(now, f"snack {n}... hic!")
            self._dispatch(Action("sound", "hiccup", 3), now)
            self.p.composer.request_gesture("hiccup", now, 3)
        else:
            beh._think(now, f"a snack! (energy {beh.mood.energy:.2f})")
            self._dispatch(Action("sound", "happy", 3), now)
            self.p.composer.request_gesture("perk", now, 3)

    def _mushroom(self, now: float) -> None:
        beh, comp = self.p.behavior, self.p.composer
        if now >= self._trip_until:  # first dose: dizzy, then the boost
            self._trip_until, self._trip_doses = now + TRIP_S, 1
            beh.mood.energy += TRIP_ENERGY
            beh.mood.curiosity = 1.0
            beh.mood.clamp()
            beh._think(now, "oh. OH. everything is... very interesting all of a sudden")
            self._dispatch(Action("sound", "dizzy", 4), now)
            comp.request_gesture("dizzy", now, 4)
        elif self._trip_doses == 1:
            self._trip_until, self._trip_doses = self._trip_until + TRIP_MORE_S, 2
            beh._think(now, "another one? okay... whoa")
            self._dispatch(Action("sound", "giggle", 3), now)
            comp.request_gesture("wiggle", now, 3)
        else:  # a third: it all comes out
            self._trip_until, self._trip_doses = 0.0, 0
            beh.mood.energy -= TRIP_CRASH
            beh.mood.clamp()
            beh._think(now, "too much... aaah... AAAH...")
            comp.request_gesture("sneeze", now, 4)
            self._dispatch(Action("sound", "sneeze", 4), now)

    def _boop(self, now: float) -> None:
        beh, comp = self.p.behavior, self.p.composer
        self._boops = [t for t in self._boops if now - t <= BOOP_WINDOW_S] + [now]
        n = len(self._boops)
        if n == BOOP_BOW_AT:
            self._boops.clear()
            beh._think(now, "thirteen boops. you have unlocked... the bow")
            self._dispatch(Action("sound", "tada", 4), now)
            comp.request_gesture("bow", now, 4)
        elif n == BOOP_DANCE_AT:
            beh._think(now, "seven boops! that calls for a little dance")
            beh._little_dance_until = now + 6.0
            self._dispatch(Action("sound", "excited", 3), now)
        else:
            self._dispatch(Action("sound", "surprised" if n == 1 else "giggle", 2), now)
            comp.request_gesture("boop", now, 3)

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
            "mime": self.mime.status(now),
            "arms": None if o.arms is None else {"left": o.arms.left, "right": o.arms.right, "left_deg": round(o.arms.left_deg), "right_deg": round(o.arms.right_deg), "conf": round(o.arms.conf, 2), "shoulder_px": round(o.arms.shoulder_px),
                                                "watching": self.signs.watching and now < self.signs.watching_until},
            "dance_along": {"on": self.dance_along, "copying": self._copying_arms, "flourish": self._copying_arms and now < self._flourish_until,
                            "antennas": None if comp.arms is None or now >= comp.arms_until else [round(comp.arms[0]), round(comp.arms[1])]},
            "song": {"singing": now < self._singing_until, "last": None if self.last_song is None else {"name": self.last_song["name"], "bpm": self.last_song["bpm"], "bars": self.last_song["bars"], "saved": self.last_song in self.songs}, "repertoire": [x["name"] for x in self.songs], "next_in_s": round(max(0.0, self.p.behavior._cool.get("sing", now) - now)) if self.singing_enabled else None},
            "build": build_info(),
            "asleep": self.asleep,
            "transcript": [{"t": round(now - t, 1), "text": txt.split("|")[0], "intents": ints} for t, txt, ints in reversed(self.transcript)],
            "senses": {
                "body": None if o.body is None else {"yaw": round(o.body.yaw_deg, 1), "pitch": round(o.body.pitch_deg, 1), "size": round(o.body.area_frac, 3)},
                "face": None if o.face is None else {"track": o.face.track_id, "yaw": round(o.face.yaw_deg, 1), "pitch": round(o.face.pitch_deg, 1), "size": round(o.face.area_frac, 3), "person": None if o.face.person is None else o.face.person.person_id, "similarity": round(o.face.similarity, 2), "tilt": round(o.face.roll_deg, 1), "head_yaw": round(o.face.head_yaw_deg, 1), "head_pitch": round(o.face.head_pitch_deg, 1), "smile": round(o.face.smile, 2)},
                "vision": None if getattr(self, "vision", None) is None else self.vision.status(now),
                "held": o.held, "shaken": o.shaken, "imu": self.pickup.stats, "imu_rub": self.imu_rub.stats, "head_rate": round(self.self_motion.rate, 2), "ears": self.touch.stats,
                "music": {"bpm": round(b.bpm, 1), "confidence": round(b.confidence, 2), "grooving": comp.groove is not None, "intensity": round(comp.groove[2], 2) if comp.groove else 0.0},
                "keypad": {"devices": list(self.keypad.devices.values()), "last_key": None if self.keypad.last_key is None else {"key": self.keypad.last_key[0], "t": round(now - self.keypad.last_key[1], 1), "does": self.keymap.lookup(self.keypad.last_key[0])}, "error": self.keypad.error,
                           "lean": round(comp.groove_lean, 2), "turn": None if self._turn is None else {"yaw": round(self._turn[0]), "for_s": round(self._turn[1] - now, 1)},
                           "snacks": len([t for t in self._snacks if now - t <= SNACK_WINDOW_S]), "tummy_ache_s": round(max(0.0, self._ache_until - now)), "trip_s": round(max(0.0, self._trip_until - now)), "trip_doses": self._trip_doses,
                           "boops": len([t for t in self._boops if now - t <= BOOP_WINDOW_S])},
                "tap": {"bpm": round(self.tap.bpm, 1), "beat": self.tap.beat_in_bar(now) if self.tap.active else 0, "bar": self.tap.bar_in_phrase(now) if self.tap.active else 0, "downbeat_known": self.tap.downbeat_known, "halftime": self.tap.active and self.tap.halftime},
                "dance": {"dancing": self.dance.state.dancing, "bpm": round(self.dance.state.bpm, 1), "confidence": round(self.dance.state.confidence, 2), "amplitude": round(self.dance.state.amplitude, 3), "holds_for_s": round(max(0.0, self.dance.locked_for - now), 1) if self.dance.state.dancing else 0.0},
                "mimic": None if comp.mimic is None else {"yaw": round(comp.mimic[0], 1), "pitch": round(comp.mimic[1], 1), "roll": round(comp.mimic[2], 1)},
                "listening": self.audio.stats["listening"], "voice_yaw": self.audio.last_voice_yaw,
                "doa": None if self.audio.doa is None else {"angle_deg": round(math.degrees(self.audio.doa[0]), 0), "speech": self.audio.doa[1]},
                "speech_s_ago": round(now - self.audio.last_speech_time, 1) if self.audio.last_speech_time > 0 else None,
                "levels": {"rms": round(self.audio.meter.rms, 5), "peak": round(self.audio.meter.peak, 4), "max_rms_3s": round(self.audio.meter.max_rms_3s, 5), "chunks": self.audio.stats["chunks"]},
                "scratch": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.audio.scratch.stats.items()},
                "head_pet": {"rubbing": self.audio.rub.rubbing, **{k: round(v, 6) for k, v in self.audio.rub.stats.items()}},
            },
            "controls": {"voice": self.voice, "pickup": self.pickup_enabled, "dance_along": self.dance_along, "ears": self.audio.enabled, "mimic_flip": self.p.composer.mimic_flip, "body_finder": getattr(getattr(self, "vision", None), "body_enabled", None), "groove_scale": self.groove_scale, "manual_groove": self.manual_groove, "keymap": self.keymap.layers, "camera_lag_ms": None if self.pose_history is None else round(self.pose_history.lag_s * 1000), "head_forward_mm": round(comp.forward_shift_m * 1000, 1), "singing": self.singing_enabled, "imu_rub_gyro": self.imu_rub.gyro_lo, "bpm": round(self.tap.bpm, 1), **{f"groove_{k}": v for k, v in comp.groove_mix.as_dict().items()}, "scratch_onset_ratio": self.audio.scratch.onset_ratio, "scratch_floor": self.audio.scratch.floor, "rub_level_ratio": self.audio.rub.level_ratio, "rub_flatness_min": self.audio.rub.flatness_min, "rub_floor": self.audio.rub.floor, "match_threshold": self.p.memory.match_threshold},
            "calibration": self.audio.calibration_result,
            "imu_calibration": self.imu_rub.calibration,
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
            self._from_page.put(Action("sound", str(value), 5))
        elif cmd == "gesture":
            from festival_pet.motion import GESTURES

            if str(value) not in GESTURES:
                raise KeyError(f"Unknown gesture '{value}'")
            self._from_page.put(Action("gesture", str(value), 5))
        elif cmd == "move":
            self._from_page.put(Action("move", str(value), 5))
        elif cmd == "sleep":
            beh.state, beh._state_since = "SLEEPING", now
            beh._think(now, "told to sleep from the control page")
            self._from_page.put(Action("sleep", "control", 5))  # the same routine the brain uses: centre, nest, motors off
        elif cmd == "wake":
            beh.state, beh._state_since = "WAKING", now
            self._from_page.put(Action("wake", "control", 5))
        elif cmd == "voice":
            if str(value) not in ("full", "quiet", "off"):
                raise ValueError(f"voice must be full, quiet or off, not '{value}'")
            self.voice = str(value)
        elif cmd == "mute":  # kept for old settings files and scripts: a plain on/off
            self.voice = "off" if bool(value) else "full"
        elif cmd == "pickup":  # held-in-hand mode
            self.pickup_enabled = bool(value)
            self.p.composer.held = self.pickup_enabled
        elif cmd == "ears":
            self.audio.enabled = bool(value)
        elif cmd == "mimic_flip":
            self.p.composer.mimic_flip = bool(value)
        elif cmd == "groove_scale":
            self.groove_scale = float(value)
        elif cmd == "manual_groove":
            self.manual_groove = bool(value)
            if self.manual_groove:  # we're grooving: whatever it was performing stops now
                if self.mime.active:
                    for item in self.mime.stop(now):
                        self._mime_action(item, now)
                if now < self._singing_until:
                    self._singing_until = 0.0
                    self.sound.cut()
        elif cmd == "tap":  # value: true = this tap is the "1"
            self.tap.tap(now, downbeat=bool(value))
            self.actions_log.append((now, "tap", f"{'ONE ' if value else ''}{self.tap.bpm:.0f} bpm"))
        elif cmd == "keymap":  # "layer:slot:KEY"
            layer, slot, key = str(value).split(":")
            self.keymap.set(layer, int(slot), key)
        elif cmd == "key":  # fire a key action as if pressed (testing from the page)
            self.key_action(str(value), now, now)
        elif cmd == "mime":  # value: true (pick), "head", "arms", or false to stop
            if value in (False, 0, None, "stop"):
                for item in self.mime.stop(now):
                    self._mime_action(item, now)
            else:
                kind = None if value is True else str(value)
                if kind not in (None, "head", "arms"):
                    raise ValueError(f"Simon says kind must be head or arms, not '{value}'")
                if self.start_simon(kind, now) is None:
                    raise ValueError("nobody in view to play Simon says with" + (" (and no arms to read)" if kind == "arms" else ""))
        elif cmd == "dance_along":
            self.dance_along = bool(value)
        elif cmd == "singing":
            self.singing_enabled = bool(value)  # the brain picks songs when it feels like one (drives.py)
        elif cmd == "sing":
            if self.asleep:
                raise ValueError("asleep")
            self.sing(now)
        elif cmd == "save_song":
            self.save_last_song()
        elif cmd == "imu_rub_gyro":
            self.imu_rub.gyro_lo = float(value)
        elif cmd == "head_forward_mm":
            self.p.composer.forward_shift_m = float(value) / 1000.0
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
            if str(value) == "imu":
                self.imu_rub.start_calibration(now)
            else:
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
    _SETTING_KEYS = ("voice", "muted", "pickup", "ears", "dance_along", "mimic_flip", "groove_scale", "manual_groove", "keymap", "camera_lag_ms", "head_forward_mm", "singing", "imu_rub_gyro", "groove_bob", "groove_sway", "groove_body", "groove_ears", "scratch_onset_ratio", "scratch_floor", "rub_level_ratio", "rub_flatness_min", "rub_floor", "match_threshold", "body_finder")

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
        names = {"muted": "mute"}  # setting key -> control name where they differ ("muted" is the pre-0.6.4 file)
        for k, v in data.items():
            if k not in self._SETTING_KEYS:
                continue
            if k == "body_finder" and getattr(self, "vision", None) is None:
                continue
            if k == "camera_lag_ms" and self.pose_history is None:
                continue
            if k == "keymap":
                if not all(isinstance(keys, list) for keys in v.values()):
                    logger.warning("settings: the key map is in the pre-0.7.2 press/hold form; keeping the factory layers")
                    continue
                for layer, keys in v.items():
                    for slot, key in enumerate(keys):
                        if key:
                            self.control("keymap", f"{layer}:{slot}:{key}")
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

    def _ants(self) -> str:
        _, ants = self._r.get_current_joint_positions()
        return f"antennas R {ants[0]:+.2f} L {ants[1]:+.2f}"

    def sleep_body(self):
        """Centre the body, nest the head, cut the torque. All of it ours, none of it the daemon's move.

        The daemon's goto_sleep holds the nest for two seconds with its move guard released before it
        disables the motors, and something in that window has been seen yanking the antennas upright at
        full speed. Doing the nest ourselves leaves no window: the nest goto ends, we read the antennas,
        and torque goes off on the next line. The antenna readings are logged at each step so a jump,
        if one still happens, is pinned to a step.
        """
        from reachy_mini.reachy_mini import INIT_ANTENNAS_JOINT_POSITIONS, INIT_HEAD_POSE, SLEEP_ANTENNAS_JOINT_POSITIONS, SLEEP_HEAD_POSE

        # Centre the body first: a turned body leaves the head nesting sideways.
        # goto_target blocks for its duration; give a far-turned body time to come round (45 deg/s, at least 1.2 s).
        body_deg = abs(math.degrees(float(self._r.get_current_joint_positions()[0][0])))
        logger.info("sleep: centring body (%.0f deg), %s", body_deg, self._ants())
        self._r.goto_target(head=INIT_HEAD_POSE, antennas=INIT_ANTENNAS_JOINT_POSITIONS, duration=max(1.2, body_deg / 45.0), body_yaw=0.0)
        logger.info("sleep: centred, %s; nesting", self._ants())
        self._r.media.play_sound("go_sleep.wav")  # the daemon's own sigh
        self._r.goto_target(head=SLEEP_HEAD_POSE, antennas=SLEEP_ANTENNAS_JOINT_POSITIONS, duration=2.0, body_yaw=0.0)
        logger.info("sleep: nested, %s; torque off", self._ants())
        self._r.disable_motors()
        time.sleep(0.3)
        logger.info("asleep: motor mode %s, %s", self.motor_mode(), self._ants())

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
                    person_model=PERSON_MODEL if PERSON_MODEL.exists() else None, pose_model=POSE_MODEL if POSE_MODEL.exists() else None)
    if not PERSON_MODEL.exists():
        logger.warning("person model missing (%s): body-finding disabled; rerun the installer", PERSON_MODEL)
    if not POSE_MODEL.exists():
        logger.warning("pose model missing (%s): arm games and dance-along disabled; rerun the installer", POSE_MODEL)
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
    pet.songs_file = SONGS_FILE
    pet.load_songs()
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

        return {"sounds": list(sounds.EMOTIONS), "meanings": sounds.MEANINGS, "gestures": list(GESTURES), "keypad": LAYERS, "moves": ["curious1", "welcoming1", "loving1", "dance1", "dance2", "dance3", "laughing1", "surprised1", "yes1", "no1", "sleep1", "cheerful1"]}

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
