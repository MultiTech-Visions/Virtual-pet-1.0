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
from fastapi import WebSocket
from pydantic import BaseModel
from scipy.spatial.transform import Rotation as R

from festival_pet import build_info, sounds
from festival_pet.audio_features import BeatTracker, LevelMeter, RubDetector, ScratchDetector, _tune
from festival_pet.behavior import Action, Behavior, FaceObs, Observation
from festival_pet.keypad import LAYERS, KeyMap, KeypadListener
from festival_pet.memory import FaceMemory
from festival_pet.mime import MimeGame
from festival_pet.motion import BODY_YAW_LIMIT, OFFER_YAW, MotionComposer, turn_pose, SOLO_GESTURES
from festival_pet import songs
from festival_pet.senses import ImuRubDetector, LoudSoundDetector, PickupDetector, PoseHistory, SelfMotionGate, TouchDetector
from festival_pet.vision import Sighting
from festival_pet.pose import PLUR_STEP_WINDOW_S, PLUR_STEPS, PLUR_TOL, PLUR_TRAIN_MIN, ArmSigns, plur_features
from festival_pet.tap_tempo import TapTempo
from festival_pet.devconsole import DevConsole
from festival_pet.trace import Recorder
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
MOVE_BODY_RETURN_DEG_S = 60.0  # how fast the body is allowed to unwind after a move that turned it right round
MOVE_BLEND_MAX_S = 3.5
FREE_DANCE_BPM = 108.0
AUDIO_RATE = 16000
ARMS_MAX_AGE_S = 0.6  # an arm read older than this is nobody's arms
# Keypad: the dancing layer's nudges
NUDGE_WINDOW_S = 2.5  # direction taps this close together count up
NUDGE_LEAN_MIN, NUDGE_LEAN_STEP = 0.45, 0.2  # one tap leans this much; each further tap in the window adds this, to 1.0
NUDGE_TILT_TAPS = 3  # that many in a row and it tilts its head that way as well
NUDGE_TILT_EVERY_S = 2.0
# Petting keys: presses keep a hand "on" it and build up, instead of retriggering an animation each time
CUDDLE_STEP = 0.14  # how much one press adds to the build-up
CUDDLE_FADE_S = 6.0  # ...which ebbs away over about this long once the pressing stops
CUDDLE_HOLD_S = 1.2  # a press keeps the virtual hand on it this long: spamming reads as continuous petting
CUDDLE_GAP_S = 3.0  # a gap longer than this starts a fresh session (the milestones below can happen again)
CUDDLE_REACT_S = 2.5  # at most one reaction of its own this often, whatever is being pressed
CUDDLES = {"pet": "head pats...", "head_pat": "head pats...", "chin_scratch": "chin scratches...", "ear_rub": "ear rubs...", "belly_rub": "tummy rubs..."}
# The build-up: (level, what it is thinking, sound, gesture). One each per session, in order.
CUDDLE_STEPS = (
    (0.0, "oh, hello", "curious", "perk"),
    (0.35, "mm, that's nice", "content", "lean"),
    (0.65, "ohhh yes, right there", "purr", "snuggle"),
    (0.95, "I have melted. this is my life now", "purr", "nuzzle"),
)
FLOURISH_EVERY_BEATS = 8  # dance-along: every two bars it stops copying and throws in two beats of its own
FLOURISH_BEATS = 2
# Performing: a song on its own is just a song. If someone actually watches one, it does another, and
# maybe a third, and only then is it a performance worth a proper bow.
WATCHING_YAW_DEG, WATCHING_PITCH_DEG = 30.0, 25.0  # their head is pointed at it within this: eye contact, near enough
ENGAGED_FRAC = 0.55  # this much of the song watched, and it has an audience
SONG_GROOVE = 0.8  # how hard it bobs to its own song, before the section's own energy and the user's dial
SONG_LEAD_S = 0.08  # the speaker's own latency: the choreography starts when the sound does, not when we ask
ENCORE_GAP_S = 1.6  # the pause between songs while it decides
ENCORE_MAX = 3  # songs in one performance
THIRD_SONG_CHANCE = 0.35  # the third is a rarity, so it stays special
# The end of a PLUR handshake: it offers an antenna as a post to slide a bracelet onto, and waits.
KANDI_OFFER_S = 10.0  # how long it holds the antenna out before giving up (they may be digging in a bag)
KANDI_SETTLE_S = 2.0  # after something lands on the antenna, stay frozen this long before moving again
KANDI_GENTLE_S = 20.0  # ...then move smaller for this long while it settles (the upright gate stays on for good)
# Giving one back: it points the ear it is wearing one on, tilts that ear's base down to the low point,
# and lowers the antenna until the bracelet slides off the tip into their hand.
KANDI_GIVE_POINT_S = 0.9  # raise the chosen antenna and look at them
KANDI_GIVE_TILT_S = 1.2  # roll the head over so that antenna's base is the lowest part of it
KANDI_GIVE_LOWER_S = 2.0  # lower it, slowly, until the bracelet runs off the end
KANDI_GIVE_SHAKE_S = 1.4  # ...then jiggle it down there: the antennas have a helical twist near the base and
#                           a bracelet catches on it, so a few quick flicks bounce it off the end
KANDI_GIVE_CATCH_S = 1.0  # hold still while they take it
KANDI_GIVE_BACK_S = 1.0  # and back up to normal
KANDI_GIVE_S = (KANDI_GIVE_POINT_S + KANDI_GIVE_TILT_S + KANDI_GIVE_LOWER_S + KANDI_GIVE_SHAKE_S
                + KANDI_GIVE_CATCH_S + KANDI_GIVE_BACK_S)
KANDI_GIVE_PITCH = 14.0  # it looks down at where the bracelet is going while it sheds it
KANDI_SHAKE_HZ, KANDI_SHAKE_DEG = 4.5, 16.0  # the jiggle: how fast, and how far either side of right down
ANTENNA_UP_DEG, ANTENNA_SHED_DEG = 175.0, 5.0  # the antenna as an "arm": up is vertical, down is laid right back
PLUR_TRAIN_READY_S = 3.0  # "get into the pose" before it starts looking
PLUR_TRAIN_WATCH_S = 2.5  # ...then this long of readings, whose medians become the prototype
PLUR_TRAIN_GAP_S = 1.2  # ...then a breath, so the "got it" lands before the next pose is called for
PLUR_CLOCK_EAR = 0  # the RIGHT antenna runs the countdown: upright is a full window, horizontal is out of time
PLUR_DOWN_EAR = 1  # ...and the left one is laid right down out of the way, so there is one thing to read
PLUR_DOWN_RAD = 2.3  # (left antenna: positive is back and down)
PLUR_FOCUS_S = 2.0  # it keeps paying attention this long after the last thing that happened
COMBO_TAPS = 4  # left right left right on the dancing layer...
COMBO_WINDOW_S = 1.0  # ...this fast: stop dancing


def _same_pose(a: dict, b: dict) -> bool:
    """Would these two trained prototypes match each other? Then the pet cannot tell them apart."""
    return all(abs(a[k] - b[k]) <= tol for k, tol in PLUR_TOL.items() if k in a and k in b)


def _side_arg(value) -> int | None:
    """A control's antenna argument: "right" -> 0, "left" -> 1, true -> None (let the pet choose)."""
    if value is True:
        return None
    side = {"right": 0, "left": 1}.get(str(value).lower())
    if side is None:
        raise ValueError(f"antenna must be 'left' or 'right', not '{value}'")
    return side


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
        self.last_body = 0.0  # degrees of body yaw last commanded: what a move has to be blended back FROM
        # (pose, antennas, body yaw degrees, when the blend started, how long it gets)
        self.blend_from: tuple[np.ndarray, list[float], float, float, float] | None = None
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
        # The performance: how much of the current song is being watched, and how many songs in we are
        self._song_ticks = 0
        self._song_watched = 0
        self._songs_in_set = 0
        self._next_song_at = 0.0
        self._song_t0 = 0.0  # when the audio actually starts, so the choreography lines up with the bars
        self._song_move = ""
        # The kandi trade: which antenna is out (or wearing one), and the clocks for each stage
        self.kandi_on = [False, False]  # antennas wearing a bracelet (right, left): what it can trade away
        self.kandi_side: int | None = None  # 0 right, 1 left
        self._kandi_offer_until = 0.0
        self._kandi_got_at = 0.0  # when something landed on the offered antenna (0 = still waiting)
        self._kandi_give_t0 = 0.0  # when the giving-one-back routine started (0 = not running)
        self._kandi_shed = False  # the bracelet has been shaken off the end this time round
        self.kandi_roll_deg = 25.0  # how far to tilt the head to shed a bracelet; flip the sign if it leans the wrong way
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
        self._training: dict | None = None  # the teaching routine's script (see train_plur)
        self._focus_gaze: tuple[float, float] | None = None  # where the person was, while it is concentrating
        self._signs_last = 0.0  # ts of the last arm reading fed to it (each reading counts once)
        # Keypad state: direction nudges (see key_action)
        self._nudge: list[tuple[float, float]] = []  # (time, side) of recent direction taps
        self._combo: list[tuple[float, float]] = []  # (time, side) of ALL recent direction taps, for the stop combo
        self._last_tilt = 0.0  # last head tilt from repeated direction taps
        # Petting build-up (see _cuddle_touch): level, last press, what was last pressed, milestones done this session
        self._cuddle = 0.0
        self._cuddle_last = -1e9
        self._cuddle_kind = ""
        self._cuddle_done: set[int] = set()
        self._cuddle_next_react = 0.0
        self._cuddle_pet_at = 0.0  # last time this cuddling was counted toward the person's affection
        self.trace = Recorder()  # the black box: see _trace_row, and /api/trace.jsonl
        self.dev = DevConsole(DATA_DIR, Path(__file__).resolve().parent.parent)  # the Dev tab's terminal
        # Both logs are trimmed as they grow, so the trace follows them by TIME, not by index: an index
        # into a list that drops its head quietly replays or skips entries.
        self._traced_action_at = 0.0
        self._traced_think_at = 0.0

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
        # Antennas lag their command while animated; the detector raises its threshold then. That includes
        # any time WE are driving them to absolute angles — the song choreography, a Simon says flag, the
        # dance-along — or the pet reads its own showmanship as somebody grabbing its ears and flinches.
        busy = self.move is not None or comp.gesture_active(now) or (comp.arms is not None and now < comp.arms_until)
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
        # the petting keys: each press keeps the virtual hand on for a moment and tops up the build-up, which
        # ebbs away once the pressing stops. Spamming the keys reads as one long cuddle, not a fit of animations.
        if now - self._cuddle_last <= CUDDLE_HOLD_S:
            obs.petting = True
        elif self._cuddle > 0.0:
            self._cuddle = self._cuddle * math.exp(-dt / CUDDLE_FADE_S)
            if self._cuddle < 0.01:
                self._cuddle = 0.0
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
                        elif sign[0] == "hug":
                            obs.hugged = True
                        else:
                            self._plur_step(sign[1], now)
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

        if now < self._singing_until:
            obs.touched = False  # mid-performance it throws its own antennas about: that is the act, not a hand
        if self._kandi_give_t0:
            obs.touched = False  # its own antenna is being driven down: that is not someone tickling its ear
            self._kandi_give(now)
        elif self._kandi_offer_until:
            self._kandi_wait(obs, now)

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
        song_groove = None
        if self._singing_until:
            if now < self._singing_until:
                song_groove = self._perform(now)
                beh._next_react = max(beh._next_react, now + 2.0)
                self._song_ticks += 1
                if self._watching(obs):
                    self._song_watched += 1
            else:
                # The first tick at or past the end. This used to look for a tick inside the last 20 ms of
                # the song, so a loop running any slower than 50 Hz missed the end of the song entirely:
                # no bow, no encore, and the antennas left holding the last pose of a song that had stopped.
                self._singing_until = 0.0
                self._song_over(now)
        elif self._next_song_at and now >= self._next_song_at:
            self._next_song_at = 0.0
            self.sing(now, style=self.last_song["style"] if self.last_song else None)  # the encore: same kind of thing

        # ---------------- brain
        solo = comp._gesture is not None and comp._gesture.name in SOLO_GESTURES and comp.gesture_active(now)
        performing = now < self._singing_until or self._next_song_at > 0.0  # a gap between songs is still the act
        trading = self.signs.plur_step > 0 or self._kandi_offer_until > 0.0 or self._kandi_give_t0 > 0.0  # mid-trade: nothing else starts
        obs.busy = ("mime" if self.mime.active else "plur" if self._training is not None else "sing" if performing
                    else "kandi" if trading else "gesture" if solo else None)
        # Manual groove with a tempo in: we are dancing. The brain treats it like a beat (no games, songs, mirror
        # or nod-copying start) and the tapped beat below wins over anything it heard or saw.
        grooving = self.manual_groove and self.tap.active and not self.asleep and beh.state not in ("HELD", "SLEEPING", "WAKING")
        obs.grooving = grooving
        beh.can_sing = self.singing_enabled and not self.asleep and self.move is None and not grooving
        actions = beh.tick(obs, now, dt)
        comp.energy = beh.mood.energy
        comp.mode = {"SLEEPING": "sleeping", "HELD": "held"}.get(beh.state, "awake")
        comp.set_gaze(beh.gaze)  # dancing or not, it stays locked on whoever it is with
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
        elif song_groove is not None:
            # Its own song. This has to be set AFTER the brain's reset above, which was clearing it every
            # tick: the beat tracker cannot hear our own song (the mics are deaf while we play), so nothing
            # put it back and the head stood still through the whole performance.
            comp.groove = song_groove
            comp.groove_phrase = self.tap.phrase_phase(now) if self.tap.downbeat_known else None
        else:
            comp.groove_phrase = None
        self._dance_along(obs, now)
        # Concentrating: while it is being taught the handshake, and from the moment the first pose of a
        # real handshake lands. Both need every frame of the pose model and both need it to stop being a
        # pet for a minute — see _focus.
        if self._training is not None:
            self._train_tick(obs, now)
        elif self.signs.plur_step:
            left = PLUR_STEP_WINDOW_S - (now - self.signs.plur_at)
            self._focus(obs, now, left, PLUR_STEP_WINDOW_S)
        elif self._focus_gaze is not None:
            self._focus_gaze = None
            comp.ears_clear()
        if vision is not None:
            vision.pose_live = (self._training is not None or self.signs.plur_step > 0 or self._copying_arms
                                or (self.mime.active and self.mime.kind == "arms")
                                or (self.signs.watching and now < self.signs.watching_until))

        # ---------------- body
        if self.move is not None:
            t = now - self.move_t0
            if t >= self.move.duration - 0.02:
                self.move = None
                # The BODY has to be blended back too, not just the head. A move that ends with the body
                # swung round (the circus one finishes 180° away) used to have that offset vanish between
                # one tick and the next: the robot whipped back to front at whatever speed the motors could
                # manage, which with bracelets on its ears is alarming. The blend gets longer the further it
                # has to come back, so the return is always about the same, sane, speed.
                back = abs(self.last_body - comp.body_yaw)
                self.blend_from = (self.last_pose, list(self.last_ants), self.last_body, now,
                                   min(MOVE_BLEND_S + back / MOVE_BODY_RETURN_DEG_S, MOVE_BLEND_MAX_S))
            else:
                head, ants, move_body = self.move.evaluate(t)
                # Moves are recorded body-forward: turn them with the body and add the move's own body swing.
                body = 0.0 if comp.held else max(-BODY_YAW_LIMIT, min(BODY_YAW_LIMIT, comp.body_yaw + math.degrees(float(move_body))))
                head = turn_pose(head, body)
                self.last_pose, self.last_ants, self.last_body = head, [float(ants[0]), float(ants[1])], body
                if not self.asleep:
                    io.set_target(head, self.last_ants, math.radians(body))
        if self.move is None:
            head, ants, body_yaw = comp.sample(now, dt)
            if self.blend_from is not None:
                a = (now - self.blend_from[3]) / self.blend_from[4]
                if a >= 1.0:
                    self.blend_from = None
                else:
                    from reachy_mini.utils.interpolation import linear_pose_interpolation

                    head = linear_pose_interpolation(self.blend_from[0], head, a)
                    ants = [self.blend_from[1][i] * (1 - a) + ants[i] * a for i in range(2)]
                    body_yaw = self.blend_from[2] * (1 - a) + body_yaw * a
            self.last_pose, self.last_ants, self.last_body = head, ants, body_yaw
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
        self._trace_tick(obs, now)
        self.p.memory.save()

    # ------------------------------------------------------------------ dance-along
    def _dance_along(self, obs: Observation, now: float) -> None:
        """Copy a dancer's arms with the antennas, to the beat, with a riff of its own every couple of bars.

        On while there is a beat to dance to (someone seen dancing, or the tapped clock with manual groove on,
        or music) and their arms are readable; off during Simon says, a library move, sleep or being held.
        """
        comp, beh = self.p.composer, self.p.behavior
        allowed = (self.dance_along and comp.groove is not None and not self.mime.active and self.move is None
                   and not self.asleep and now >= self._singing_until  # its own song: the antennas are performing it
                   and beh.state not in ("HELD", "SLEEPING", "WAKING"))
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
                if act.name == "jingle":
                    self.jingle(now, act.priority)  # a jingle has a tempo, and the body wants to know it
                else:
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
    def sing(self, now: float, song: dict | None = None, style: str | None = None) -> dict:
        """Sing ``song``, or a saved one (half the time, if any), or make one up. ``style`` ("drumline" /
        "bass") forces what it makes up, and rules out saved songs of the other kind. Returns the song."""
        rng = self.p.composer.rng
        if song is None:
            saved = [s for s in self.songs if style is None or s["style"] == style]
            song = rng.choice(saved) if saved and rng.random() < 0.5 else songs.compose(rng, style)
        buf = songs.render(song)
        dur = songs.duration(song)
        self.sound.request_buffer(buf, "song:" + song["name"], 2, now)
        self.last_song = song
        self._song_t0 = now + SONG_LEAD_S
        self._songs_in_set += 1
        self._song_ticks = self._song_watched = 0
        self._singing_until = now + dur
        # bob along; the page shows the tempo too. Bass music is counted in halftime, so the body moves on
        # the 1 and the 3 rather than on all four of the song's beats (which at 140 would rattle the neck).
        self.tap.set_bpm(songs.body_bpm(song), beat_at=now + 0.08)
        self.audio.deaf_until = max(self.audio.deaf_until, now + dur + 0.5)
        self.p.behavior._think(now, "a song! " + songs.describe(song))
        self.actions_log.append((now, "song", song["name"]))
        return song

    def jingle(self, now: float, priority: int = 2) -> float:
        """Hum a made-up little song, and bob to it. Returns its tempo.

        It is counted in — four taps before the tune starts — and the tap clock is set to the same beat
        from the same instant, so the count-in lands on the pet's own bob and whoever is listening can
        find the beat and groove along with it rather than wondering what that noise was.
        """
        buf, bpm = sounds.render_jingle(self.p.composer.rng, AUDIO_RATE)
        self.sound.request_buffer(buf, "jingle", priority, now)
        self.tap.set_bpm(bpm, beat_at=now + 0.08)
        self.audio.deaf_until = max(self.audio.deaf_until, now + sounds.phrase_duration(buf, AUDIO_RATE) + 0.3)
        self.actions_log.append((now, "sound", f"jingle ({bpm:.0f} bpm)"))
        return bpm

    def _perform(self, now: float) -> tuple[float, float, float] | None:
        """Play the song with the body, not just through the speaker. Returns the groove to bob to.

        The head bobs on the (halftime) beat, harder in a drop and barely at all in a breakdown, and the
        antennas play the phrasing: climbing through a build, slamming on every beat of the drop, thrown
        back and forth through a fill, folded away in the breakdown, held up at the end. The bar list IS
        the arrangement, so the choreography comes straight off it and is different for every song.
        """
        comp, song = self.p.composer, self.last_song
        if song is None:
            return None
        move, bar_u, beat_u = songs.section(song, now - self._song_t0)
        left, right = songs.arms_for(move, bar_u, beat_u)
        comp.show_arms(left, right, now, 0.3)
        self._song_move = move
        return (self.tap.phase(now), self.tap.bar_phase(now), SONG_GROOVE * songs.energy(move) * self.groove_scale)

    def _plur_step(self, step: str, now: float) -> None:
        """One step of the PLUR handshake landed: answer it in kind.

        Peace, love, unity, respect. The last one is the one that matters: it stops dead and offers its
        head, so a bracelet can be threaded over an antenna without the thing breathing and squirming,
        and then it has a look at what it has been given.
        """
        beh = self.p.behavior
        if step == "peace":
            beh._think(now, "peace! (antennas up)")
            self._dispatch(Action("sound", "excited", 3), now)
            self._dispatch(Action("gesture", "peace", 4), now)
        elif step == "love":
            beh._think(now, "...love. aww")
            self._dispatch(Action("sound", "coo", 3), now)
            self._dispatch(Action("gesture", "heart", 4), now)
        elif step == "unity":
            beh._think(now, "...unity")
            self._dispatch(Action("sound", "content", 3), now)
            self._dispatch(Action("gesture", "peace", 4), now)  # the same V it answered peace with
        elif step == "respect":
            beh._think(now, "...and respect")
            self._dispatch(Action("sound", "happy", 3), now)
            self.trade_kandi(now)
        else:
            raise KeyError(f"unknown PLUR step '{step}'")
        beh._last_interaction = now
        beh.mood.boredom = max(0.0, beh.mood.boredom - 0.15)

    def trade_kandi(self, now: float, side: int | None = None) -> None:
        """The whole exchange: give one of its own away, then hold the same ear out for one back.

        If it is not wearing one there is nothing to give, so it goes straight to asking.
        """
        loaded = [i for i in (0, 1) if self.kandi_on[i]]
        if side is not None and side not in (0, 1):
            raise ValueError(f"antenna side must be 0 (right) or 1 (left), not {side}")
        give = side if side is not None and self.kandi_on[side] else (loaded[0] if loaded else None)
        if give is None:
            self.start_kandi(now, side)
            return
        self.kandi_side = give
        self._kandi_give_t0 = now
        self._kandi_shed = False
        self.p.composer.loaded[give] = False  # the upright gate has to be open by the time the antenna comes down
        self._kandi_offer_until = self._kandi_got_at = 0.0
        self.p.behavior._think(now, f"here — this one's for you, off my {'left' if give else 'right'} ear")
        self._dispatch(Action("sound", "fanfare", 4), now)  # da da-da DA: everyone look, something is happening
        self.actions_log.append((now, "kandi", f"giving the one on the {'left' if give else 'right'} antenna"))

    def _kandi_give(self, now: float) -> None:
        """Shed a bracelet off an antenna, a step at a time, by tilting that ear's base down to be the
        lowest part of the head and then lowering the antenna until the bracelet runs off the tip.
        (The caller swallows this tick's ear-touch: the antenna moving under its own command would
        otherwise read as being tickled, and it would flinch in the middle of handing something over.)

        Driven frame by frame through the two overrides that already exist: ``hold`` for the head pose
        (Simon says shows poses with it) and ``show_arms`` for absolute antenna angles (the arm game).
        """
        beh, comp = self.p.behavior, self.p.composer
        side = self.kandi_side
        u = now - self._kandi_give_t0
        aim = beh.gaze[0] if beh.gaze is not None else 0.0
        roll = self.kandi_roll_deg * (-1.0 if side else 1.0)  # toward the giving side, whichever way that is
        tilt = max(0.0, min(1.0, (u - KANDI_GIVE_POINT_S) / KANDI_GIVE_TILT_S))
        back = max(0.0, (u - (KANDI_GIVE_S - KANDI_GIVE_BACK_S)) / KANDI_GIVE_BACK_S)
        ease = max(0.0, tilt - back)
        # The head turns AWAY from the giving side, the same as when it offers one: that is what brings that
        # ear round to the front, where their hand is, instead of leaving it out at the side of the head.
        comp.hold = (aim + OFFER_YAW * (-1.0 if side else 1.0) * ease, KANDI_GIVE_PITCH * ease, roll * ease, now + 0.3)
        down_at = KANDI_GIVE_POINT_S + KANDI_GIVE_TILT_S + KANDI_GIVE_LOWER_S
        drop = max(0.0, min(1.0, (u - KANDI_GIVE_POINT_S - KANDI_GIVE_TILT_S) / KANDI_GIVE_LOWER_S)) * (1.0 - back)
        deg = ANTENNA_UP_DEG + (ANTENNA_SHED_DEG - ANTENNA_UP_DEG) * drop
        if down_at <= u < down_at + KANDI_GIVE_SHAKE_S:  # the jiggle, to bounce it past the twist at the base
            deg += KANDI_SHAKE_DEG * (1.0 + math.sin(2 * math.pi * KANDI_SHAKE_HZ * (u - down_at)))
        comp.show_arms(deg if side == 1 else ANTENNA_UP_DEG, deg if side == 0 else ANTENNA_UP_DEG, now, 0.3)
        if not self._kandi_shed and u >= down_at + KANDI_GIVE_SHAKE_S * 0.6:
            self._kandi_shed = True  # a few flicks in: it has had every chance to come off
            beh._think(now, "...there you go!")
            self._dispatch(Action("sound", "giggle", 3), now)
            self.actions_log.append((now, "kandi", "gave one away"))
        if u < KANDI_GIVE_S:
            return
        self._kandi_give_t0 = 0.0
        comp.hold = None
        comp.show_arms(None)
        beh._think(now, "...your turn?")
        self.start_kandi(now, side)  # and hold the same ear out for one back

    def start_kandi(self, now: float, side: int | None = None) -> int:
        """Hold out an antenna for a bracelet and freeze. ``side`` 0 right, 1 left; by default an empty
        one. Returns the side offered."""
        if side is None:
            side = 0 if not self.kandi_on[0] else 1
        if side not in (0, 1):
            raise ValueError(f"antenna side must be 0 (right) or 1 (left), not {side}")
        self.kandi_side = side
        self._kandi_offer_until = now + KANDI_OFFER_S
        self._kandi_got_at = 0.0
        self.p.composer.hold_still(now, KANDI_OFFER_S + KANDI_SETTLE_S, offer=side)
        self.p.behavior._think(now, f"here: my {'left' if side else 'right'} ear. I'll hold still, slide it on")
        self.actions_log.append((now, "kandi", f"offering the {'left' if side else 'right'} antenna"))
        return side

    def _focus(self, obs: Observation, now: float, left_s: float | None = None, window_s: float = 1.0) -> None:
        """Pay attention to the person in front of it, and SHOW that it is paying attention.

        Mid-handshake (and while being taught one) the pet used to carry on being a pet: glancing off at
        the wall, humming, starting a game, wandering its gaze — so there was no way to tell whether it
        had seen your last pose or what it was waiting for. This pins the gaze on whoever is there, stands
        every other impulse down, reads the pose model every frame, and runs a countdown down one antenna:
        upright means the whole window is left, horizontal means it is about to give up on you.
        """
        beh, comp = self.p.behavior, self.p.composer
        who = obs.face if obs.face is not None else obs.body
        if who is not None:
            self._focus_gaze = (who.yaw_deg, who.pitch_deg)
            beh._last_seen_yaw, beh._last_seen_pitch = who.yaw_deg, who.pitch_deg
        if self._focus_gaze is not None:  # nobody in view for a moment: hold where they were, do not go hunting
            beh.gaze = self._focus_gaze
            comp.set_gaze(self._focus_gaze)
        beh._next_react = max(beh._next_react, now + PLUR_FOCUS_S)
        beh._next_glance = max(beh._next_glance, now + PLUR_FOCUS_S)
        beh._next_jingle = max(beh._next_jingle, now + 30.0)
        beh._look_until = 0.0
        beh.mimicking = False
        # The countdown waits for the answering gesture to finish — peace and love need both antennas —
        # and then it is unmistakable: the left one laid right down, the right one standing up and falling
        # to horizontal as the window runs out.
        if left_s is not None and not comp.gesture_active(now):
            comp.ear_clock(PLUR_CLOCK_EAR, max(0.0, min(1.0, left_s / max(window_s, 1e-6))), now)
            comp.ear_hold[PLUR_DOWN_EAR] = PLUR_DOWN_RAD
            comp.ear_hold_until[PLUR_DOWN_EAR] = now + 0.3

    def train_plur(self, step: str, now: float) -> None:
        """Show it what the PLUR poses look like on a real person, one after another without stopping.

        The rules in pose.py are a guess at where somebody holds their arms; this replaces the guess with
        a measurement. ``step`` is one of the four, or "all" to run the whole handshake through as a
        routine — which is the way to do it, because the poses come one after another in real life and
        stopping between them is exactly when the pet loses you. Per pose: a countdown on an antenna to
        get into it, a beep when it starts watching, two and a half seconds of reading, a beep for yes or
        no, a breath, and straight on to the next one. It focuses the whole way through (see ``_focus``).
        """
        queue = list(PLUR_STEPS) if step in ("all", "", "true", "True") else [step]
        for name in queue:
            if name not in PLUR_STEPS:
                raise KeyError(f"unknown PLUR step '{name}'. Known: {', '.join(PLUR_STEPS)}, all")
        self._training = {"queue": queue, "step": "", "phase": "", "until": now, "samples": [], "at": 0.0, "learned": []}
        self._focus_gaze = None
        self._dispatch(Action("sound", "mime_start", 4), now)
        self.p.behavior._think(now, "teach me the handshake. watch my ear for the countdown")
        self.actions_log.append((now, "plur", "learning " + ", ".join(queue)))
        self._train_next(now)

    def _train_next(self, now: float) -> None:
        """Call for the next pose, or finish."""
        t = self._training
        assert t is not None
        if not t["queue"]:
            learned = t["learned"]
            self._training = None
            self.p.composer.ears_clear()
            self.p.behavior._think(now, ("learned: " + ", ".join(learned)) if learned else "...I didn't get any of those")
            self._dispatch(Action("sound", "tada" if learned else "sad", 4), now)
            self._dispatch(Action("gesture", "tada" if learned else "droop", 3), now)
            self.actions_log.append((now, "plur", f"learning done: {len(learned)} pose(s)"))
            self.save_settings()
            return
        t["step"], t["phase"], t["until"] = t["queue"].pop(0), "ready", now + PLUR_TRAIN_READY_S
        t["samples"], t["at"] = [], 0.0
        self.p.behavior._think(now, f"show me: {t['step']}...")
        self._dispatch(Action("sound", "mime_cue", 4), now)

    def _train_tick(self, obs: Observation, now: float) -> None:
        """One tick of the teaching routine: count down on an ear, read, confirm, move on."""
        t = self._training
        assert t is not None
        beh = self.p.behavior
        phase, left = t["phase"], t["until"] - now
        window = {"ready": PLUR_TRAIN_READY_S, "watch": PLUR_TRAIN_WATCH_S, "gap": PLUR_TRAIN_GAP_S}[phase]
        self._focus(obs, now, left, window)
        if phase == "ready":
            if left <= 0.0:
                t["phase"], t["until"] = "watch", now + PLUR_TRAIN_WATCH_S
                self._dispatch(Action("sound", "yes", 4), now)  # NOW: that is the pose it is reading
                beh._think(now, f"...watching. hold {t['step']}")
            return
        if phase == "watch":
            arms = obs.arms
            if arms is not None and arms.ts != t["at"]:
                t["at"] = arms.ts
                t["samples"].append(plur_features(arms))
            if left > 0.0:
                return
            step, samples = t["step"], t["samples"]
            if len(samples) >= PLUR_TRAIN_MIN:
                proto = {k: float(np.median([s[k] for s in samples])) for k in samples[0]}
                # Two poses it cannot tell apart would make the handshake ambiguous for good, and silently:
                # it would answer whichever came first in the list every time. Say so instead.
                clash = [n for n in self.signs.trained if n != step and _same_pose(self.signs.trained[n], proto)]
                self.signs.trained[step] = proto
                t["learned"].append(step)
                if clash:
                    beh._think(now, f"...but that looks the same to me as {', '.join(clash)}. make them more different?")
                    self.actions_log.append((now, "plur", f"{step} looks the same as {', '.join(clash)}"))
                beh._think(now, f"got it: that's {step}")
                self._dispatch(Action("sound", "tada", 4), now)
                self._dispatch(Action("gesture", "nod", 3), now)
                self.actions_log.append((now, "plur", f"learned {step} from {len(samples)} readings"))
            else:
                beh._think(now, f"...couldn't see your arms well enough for {step}")
                self._dispatch(Action("sound", "confused", 4), now)
                self.actions_log.append((now, "plur", f"learning {step} failed: {len(samples)} readings"))
            t["phase"], t["until"] = "gap", now + PLUR_TRAIN_GAP_S
            return
        if left <= 0.0:  # the breath between poses
            self._train_next(now)

    def cancel_training(self, now: float) -> None:
        if self._training is None:
            return
        self._training = None
        self.p.composer.ears_clear()
        self.actions_log.append((now, "plur", "learning cancelled"))

    def _kandi_regate(self) -> None:
        """Put the upright gate back the way the ear toggles say it should be.

        The gate is lifted off the giving ear for the length of a trade (it has to come down past vertical
        to shed one), so every way out of a trade has to put it back — or that ear would stay free to swing
        about with a fresh bracelet on it.
        """
        self.p.composer.loaded = list(self.kandi_on)

    def cancel_kandi(self, now: float) -> None:
        """Stop mid-trade (it was started by mistake, or nobody came)."""
        self._kandi_offer_until = self._kandi_got_at = self._kandi_give_t0 = 0.0
        self._kandi_shed = False
        self._kandi_regate()
        self.signs.plur_step = 0
        self.p.composer.hold = None
        self.p.composer.show_arms(None)
        self.p.composer.release_still()
        self.actions_log.append((now, "kandi", "trade cancelled"))

    def set_bracelets(self, sides: list[int], now: float, gentle: bool = False) -> None:
        """Say which antennas are wearing a bracelet (0 right, 1 left). A loaded antenna is held near
        vertical whatever else it does, which is all a bracelet needs to stay on; ``gentle`` also makes
        it move smaller for a moment, which is worth it right after one has been put on."""
        for i in sides:
            if i not in (0, 1):
                raise ValueError(f"antenna side must be 0 (right) or 1 (left), not {i}")
        self.kandi_on = [0 in sides, 1 in sides]
        self.p.composer.loaded = list(self.kandi_on)
        if gentle:
            self.p.composer.gentle_until = now + KANDI_GENTLE_S
        worn = [n for i, n in enumerate(("right", "left")) if self.kandi_on[i]]
        self.actions_log.append((now, "kandi", "wearing: " + (", ".join(worn) if worn else "nothing")))

    def _kandi_wait(self, obs: Observation, now: float) -> None:
        """Waiting with an antenna out. The bracelet going on is felt as that antenna being pushed off
        its commanded angle — the same detector that feels an ear tickle — so no extra sensing is needed.
        The touch is swallowed here, or the brain would flinch at the very moment it must not move.
        """
        beh, comp = self.p.behavior, self.p.composer
        touched, side = obs.touched, obs.touched_side
        obs.touched = False  # not an ear tickle: it is the whole point of standing here
        if not self._kandi_got_at:
            if touched and side == self.kandi_side:
                self._kandi_got_at = now
                comp.hold_still(now, KANDI_SETTLE_S, offer=self.kandi_side)  # two more seconds of not moving
                beh._think(now, "...that's it. that's mine now. thank you")
                self._dispatch(Action("sound", "content", 3), now)
                self.actions_log.append((now, "kandi", "something landed on the antenna"))
                return
            if now >= self._kandi_offer_until:  # nobody had one ready
                self._kandi_offer_until = 0.0
                comp.release_still()
                self._kandi_regate()
                beh._think(now, "...no? that's okay. next time")
                self._dispatch(Action("sound", "curious", 2), now)
                self.actions_log.append((now, "kandi", "nothing came: offer over"))
            return
        if now < self._kandi_got_at + KANDI_SETTLE_S:
            return  # let it settle before anything moves
        self._kandi_offer_until = 0.0
        comp.release_still()  # the antenna and head ease back to normal, slowly
        self.set_bracelets(sorted({*[i for i in (0, 1) if self.kandi_on[i]], self.kandi_side}), now, gentle=True)
        beh._think(now, f"I'm wearing it. on my {'left' if self.kandi_side else 'right'} ear. careful now")
        self._dispatch(Action("sound", "tada", 3), now)
        self._dispatch(Action("gesture", "nod", 2), now)  # a nod, not a dance: there is a bracelet on there
        beh.mood.social = min(1.0, beh.mood.social + 0.2)
        if beh._engaged_person is not None:
            beh.memory.add_kandi(beh._engaged_person)
        self.actions_log.append((now, "kandi", "kandi traded"))

    def _watching(self, obs: Observation) -> bool:
        """Is somebody actually watching this? A face in view with their head pointed at us: near enough
        to eye contact for a robot with no eye tracker, and it does not false-fire on someone walking past."""
        face = obs.face
        return face is not None and abs(face.head_yaw_deg) < WATCHING_YAW_DEG and abs(face.head_pitch_deg) < WATCHING_PITCH_DEG

    def _song_over(self, now: float) -> None:
        """A song has finished. If it was watched, line up another; if the set is done, take the bow it earned.

        Alone, it sings its little song and is pleased with itself. With an audience it does two or three
        and then the full house bow, so the big routine means something when you see it.
        """
        beh = self.p.behavior
        watched = self._song_ticks > 0 and self._song_watched / self._song_ticks >= ENGAGED_FRAC
        frac = 0.0 if self._song_ticks == 0 else self._song_watched / self._song_ticks
        self._song_ticks = self._song_watched = 0
        self._song_move = ""
        self.p.composer.show_arms(None)  # the performance is over: give the antennas back
        more = watched and self._songs_in_set < ENCORE_MAX and (self._songs_in_set < 2 or self.p.composer.rng.random() < THIRD_SONG_CHANCE)
        self.actions_log.append((now, "song", f"finished ({frac:.0%} watched){', encore' if more else ''}"))
        if more:
            self._next_song_at = now + ENCORE_GAP_S
            beh._think(now, "they're still watching! one more" if self._songs_in_set == 1 else "okay, ONE more, this one's the good one")
            self._dispatch(Action("sound", "excited", 3), now)
            self._dispatch(Action("gesture", "perk", 3), now)
            return
        if self._songs_in_set >= 2:  # a real set, for a real audience: the whole bow, to the whole house
            beh._think(now, f"thank you! thank you, you're too kind ({self._songs_in_set} songs)")
            self._dispatch(Action("sound", "tada", 4), now)
            self.p.composer.request_gesture("bow", now, 4)
        else:  # nobody much was looking: pleased with itself anyway
            beh._think(now, "...nailed it" if watched else "that was a good one, I thought")
            self._dispatch(Action("sound", "happy" if watched else "content", 2), now)
            self._dispatch(Action("gesture", "tada" if watched else "wiggle", 2), now)
        self._songs_in_set = 0

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
        if self.songs_file is None or not self.songs_file.exists():
            return
        saved = json.loads(self.songs_file.read_text())
        # The drumline beeps are gone, so any saved one is dropped rather than kept as a song that can no
        # longer be played. Songs saved before styles existed at all were drumline ones, so they go too.
        self.songs = [s for s in saved if s.get("style") in songs.STYLES]
        if len(self.songs) != len(saved):
            logger.info("songs: dropped %d saved drumline song(s); that style is gone", len(saved) - len(self.songs))

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
                    comp.groove_lean = 0.0
                    self.control("manual_groove", False)
                    self.tap.clear()
                    beh._think(now, "okay, okay! done dancing. phew")
                    self._dispatch(Action("sound", "content", 3), now)
                    self._dispatch(Action("gesture", "shake_off", 3), now)
                    return
                self._nudge_groove(side, now)
        elif action in CUDDLES:
            self._cuddle_touch(action, now)
        else:
            raise KeyError(f"unknown key action '{action}'")

    def _cuddle_touch(self, kind: str, now: float) -> None:
        """A petting key. Presses do not each fire an animation: they keep a hand on it (``composer.petted``,
        the same continuous fold-and-lean the real rub uses) and build up an affection level that ebbs away.
        The reactions come off that level, each once per session and never more often than CUDDLE_REACT_S,
        so somebody drumming on all four keys gets a pet slowly going gooey, not four animations at once."""
        beh, obs = self.p.behavior, self._last_obs
        if now - self._cuddle_last > CUDDLE_GAP_S:
            self._cuddle_done = set()  # a fresh session: it can go through the whole build-up again
            if self._cuddle < 0.05 and beh.state != "SLEEPING":
                obs.petted = True  # the one edge: "ahh, head pets... leaning in"
        self._cuddle_last = now
        self._cuddle = min(1.0, self._cuddle + CUDDLE_STEP)
        self._cuddle_kind = kind
        obs.petting = True  # a hand is on it (a level, not an edge): the brain purrs and leans at its own pace
        beh.mood.social += 0.02
        beh.mood.boredom -= 0.02
        beh.mood.clamp()
        if beh._engaged_person is not None and now - self._cuddle_pet_at > 2.0:
            self._cuddle_pet_at = now
            beh.memory.add_pet(beh._engaged_person)
        if beh.state == "SLEEPING":
            return
        for i, (at, thought, sound, gesture) in enumerate(CUDDLE_STEPS):
            if self._cuddle >= at and i not in self._cuddle_done and now >= self._cuddle_next_react:
                self._cuddle_done.add(i)
                self._cuddle_next_react = now + CUDDLE_REACT_S
                self._dispatch(Action("sound", sound, 2), now)
                self._dispatch(Action("gesture", gesture, 2), now)
                beh._think(now, f"{CUDDLES[kind]} {thought}")
                return

    def _nudge_groove(self, side: float, now: float) -> None:
        """A direction tap on the dancing layer: which way to groove, not where to point. It leans that way
        (head roll and yaw, antennas, a few degrees of body) and keeps dancing with whoever it is with; keep
        tapping the same way and the lean grows and it tilts its head over too. The body never turns away."""
        beh, comp = self.p.behavior, self.p.composer
        self._nudge = [(t, s) for t, s in self._nudge if now - t <= NUDGE_WINDOW_S and s == side] + [(now, side)]
        comp.groove_lean = side * min(1.0, NUDGE_LEAN_MIN + NUDGE_LEAN_STEP * (len(self._nudge) - 1))
        if len(self._nudge) >= NUDGE_TILT_TAPS and now - self._last_tilt >= NUDGE_TILT_EVERY_S:
            self._last_tilt = now
            beh._think(now, f"yeah, over this way ({'left' if side > 0 else 'right'})")
            self._dispatch(Action("gesture", f"tilt:{'+' if side > 0 else '-'}", 2), now)

    def _trace_row(self, obs: Observation, now: float) -> dict:
        """One line of the black box: what it could see, what it decided, and where it actually went.

        Chosen for the questions that keep coming up and cannot be answered from a chat window — did
        the camera have a face that frame, where did the brain point the gaze, where did the head end
        up, was the pose model returning arms, which remembered spot was it checking, was a spot
        blacklisted. Short keys: this runs ten times a second for days.
        """
        beh, comp = self.p.behavior, self.p.composer
        face, body = obs.face, obs.body
        vision = getattr(self, "vision", None)
        roll, pitch, yaw = R.from_matrix(self.last_pose[:3, :3]).as_euler("xyz", degrees=True)
        row = {
            "st": beh.state, "act": beh.activity, "busy": obs.busy,
            # what it could see
            "face": None if face is None else [round(face.yaw_deg, 1), round(face.pitch_deg, 1), round(face.area_frac, 3), face.track_id],
            "body": None if body is None else [round(body.yaw_deg, 1), round(body.pitch_deg, 1), round(body.area_frac, 3)],
            "arms": None if obs.arms is None else [round(obs.arms.left_deg), round(obs.arms.right_deg), round(obs.arms.conf, 2), round(now - obs.arms.ts, 2)],
            # where it meant to look, and where it went
            "gaze": None if beh.gaze is None else [round(beh.gaze[0], 1), round(beh.gaze[1], 1)],
            "head": [round(float(yaw), 1), round(float(pitch), 1), round(float(roll), 1)],
            "bodyyaw": round(comp.body_yaw, 1), "ants": [round(a, 2) for a in self.last_ants],
            # what it remembers about people, and what it has written off
            "spots": [[round(x.yaw), x.seen, x.faces, round(now - x.at, 1)] for x in beh.seen_spots.recent(now)],
            "search": beh._search_i if beh.state == "SEARCHING" else None,
            "ignored": [round(y) for y, until in beh._body_ignore if until > now],
            "lostfor": round(now - beh._last_face_time, 1) if beh._last_face_time else None,
            # the routines that people are trying to do with it
            "plur": self.signs.plur_step, "train": None if self._training is None else [self._training["step"], self._training["phase"]],
            "kandi": [int(self._kandi_give_t0 > 0), int(self._kandi_offer_until > 0), self.kandi_side, [int(x) for x in self.kandi_on]],
            "song": self._song_move or None,
            # the senses that fire by accident
            "touch": [int(obs.touched), obs.touched_side, int(obs.petting)],
            "ear_dev": [round(x, 2) for x in self.touch.stats["dev"]],
        }
        if vision is not None:
            v = vision.stats
            row["vis"] = [v["frames"], v["faces"], v["bodies"], round(v["detect_ms"]), round(v["pose_ms"]),
                          v["no_frame"], v["errors"], int(vision.pose_live)]
        return row

    def _trace_tick(self, obs: Observation, now: float) -> None:
        """Sample the state, and copy anything new out of the thought and action logs into the trace."""
        tr = self.trace
        if not tr.enabled:
            return
        if tr.due(now):
            tr.tick(now, self._trace_row(obs, now))
        for t, kind, name in self.actions_log:
            if t > self._traced_action_at:
                tr.event(t, "action", f"{kind}:{name}")
        self._traced_action_at = max([self._traced_action_at] + [t for t, _, _ in self.actions_log])
        for t, text in self.p.behavior.thoughts:
            if t > self._traced_think_at:
                tr.event(t, "think", text)
        self._traced_think_at = max([self._traced_think_at] + [t for t, _ in self.p.behavior.thoughts])

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
                                                "watching": self.signs.watching and now < self.signs.watching_until,
                                                "plur": None if not (self.signs.plur_step or self._kandi_offer_until) else {"step": self.signs.plur_step, "next": PLUR_STEPS[self.signs.plur_step] if self.signs.plur_step else None},
                                                "features": {k: round(x, 2) for k, x in plur_features(o.arms).items()}},
            "kandi": {"offering": self._kandi_offer_until > 0.0, "side": None if self.kandi_side is None else ("left" if self.kandi_side else "right"),
                      "waiting_s": round(max(0.0, self._kandi_offer_until - now), 1) if self._kandi_offer_until else None,
                      "got_it": self._kandi_got_at > 0.0, "step": self.signs.plur_step,
                      "giving": self._kandi_give_t0 > 0.0, "roll_deg": self.kandi_roll_deg,
                      "wearing": [n for i, n in enumerate(("right", "left")) if self.kandi_on[i]],
                      "settling_s": round(max(0.0, comp.gentle_until - now), 1) if comp.gentle_until > now else None,
                      "damp": comp.kandi_damp, "trained": {k: {n: round(x, 2) for n, x in v.items()} for k, v in self.signs.trained.items()},
                      "training": None if self._training is None else {"step": self._training["step"], "phase": self._training["phase"],
                                                                       "queue": list(self._training["queue"]), "learned": list(self._training["learned"]),
                                                                       "left_s": round(max(0.0, self._training["until"] - now), 1),
                                                                       "seen": len(self._training["samples"])},
                      "focused": self._focus_gaze is not None},
            "dance_along": {"on": self.dance_along, "copying": self._copying_arms, "flourish": self._copying_arms and now < self._flourish_until,
                            "antennas": None if comp.arms is None or now >= comp.arms_until else [round(comp.arms[0]), round(comp.arms[1])]},
            "song": {"singing": now < self._singing_until, "last": None if self.last_song is None else {"name": self.last_song["name"], "style": self.last_song["style"], "bpm": self.last_song["bpm"], "bars": self.last_song["bars"], "saved": self.last_song in self.songs}, "repertoire": [x["name"] for x in self.songs],
                     "section": self._song_move or None, "set": {"songs": self._songs_in_set, "watched": None if self._song_ticks == 0 else round(self._song_watched / self._song_ticks, 2), "encore_in_s": round(max(0.0, self._next_song_at - now), 1) if self._next_song_at else None},
                     "next_in_s": round(max(0.0, self.p.behavior._cool.get("sing", now) - now)) if self.singing_enabled else None},
            "build": build_info(),
            "trace": self.trace.stats(now),
            "asleep": self.asleep,
            "transcript": [{"t": round(now - t, 1), "text": txt.split("|")[0], "intents": ints} for t, txt, ints in reversed(self.transcript)],
            "senses": {
                "body": None if o.body is None else {"yaw": round(o.body.yaw_deg, 1), "pitch": round(o.body.pitch_deg, 1), "size": round(o.body.area_frac, 3)},
                "face": None if o.face is None else {"track": o.face.track_id, "yaw": round(o.face.yaw_deg, 1), "pitch": round(o.face.pitch_deg, 1), "size": round(o.face.area_frac, 3), "person": None if o.face.person is None else o.face.person.person_id, "similarity": round(o.face.similarity, 2), "tilt": round(o.face.roll_deg, 1), "head_yaw": round(o.face.head_yaw_deg, 1), "head_pitch": round(o.face.head_pitch_deg, 1), "smile": round(o.face.smile, 2)},
                "vision": None if getattr(self, "vision", None) is None else self.vision.status(now),
                "held": o.held, "shaken": o.shaken, "imu": self.pickup.stats, "imu_rub": self.imu_rub.stats, "head_rate": round(self.self_motion.rate, 2), "ears": self.touch.stats,
                "music": {"bpm": round(b.bpm, 1), "confidence": round(b.confidence, 2), "grooving": comp.groove is not None, "intensity": round(comp.groove[2], 2) if comp.groove else 0.0},
                "keypad": {"devices": list(self.keypad.devices.values()), "last_key": None if self.keypad.last_key is None else {"key": self.keypad.last_key[0], "t": round(now - self.keypad.last_key[1], 1), "does": self.keymap.lookup(self.keypad.last_key[0])}, "error": self.keypad.error,
                           "lean": round(comp.groove_lean, 2), "cuddle": round(self._cuddle, 2)},
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
            "controls": {"voice": self.voice, "pickup": self.pickup_enabled, "dance_along": self.dance_along, "ears": self.audio.enabled, "mimic_flip": self.p.composer.mimic_flip, "body_finder": getattr(getattr(self, "vision", None), "body_enabled", None), "groove_scale": self.groove_scale, "manual_groove": self.manual_groove, "keymap": self.keymap.layers, "camera_lag_ms": None if self.pose_history is None else round(self.pose_history.lag_s * 1000), "head_forward_mm": round(comp.forward_shift_m * 1000, 1), "singing": self.singing_enabled, "imu_rub_gyro": self.imu_rub.gyro_lo, "kandi_roll": self.kandi_roll_deg, "kandi_damp": comp.kandi_damp, "plur_trained": self.signs.trained, "bpm": round(self.tap.bpm, 1), **{f"groove_{k}": v for k, v in comp.groove_mix.as_dict().items()}, "scratch_onset_ratio": self.audio.scratch.onset_ratio, "scratch_floor": self.audio.scratch.floor, "rub_level_ratio": self.audio.rub.level_ratio, "rub_flatness_min": self.audio.rub.flatness_min, "rub_floor": self.audio.rub.floor, "match_threshold": self.p.memory.match_threshold},
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
        elif cmd == "kandi":  # value: true for the whole trade, "left"/"right" to pick an ear, false to stop
            if value in (False, 0, None, "off"):
                self.cancel_kandi(now)
            else:
                self.trade_kandi(now, _side_arg(value))
        elif cmd == "ask_kandi":  # just the asking half: hold an ear out, give nothing away
            self.start_kandi(now, _side_arg(value))
        elif cmd == "bracelet":  # which antennas are wearing one: "none", "right", "left" or "both"
            sides = {"none": [], "off": [], "right": [0], "left": [1], "both": [0, 1]}.get(str(value).lower() if value is not True else "both")
            if sides is None:
                raise ValueError(f"bracelets must be none, left, right or both, not '{value}'")
            self.set_bracelets(sides, now)
        elif cmd == "kandi_roll":  # how far the head tilts to shed one; negative if it leans the wrong way
            self.kandi_roll_deg = float(value)
            if abs(self.kandi_roll_deg) > 40.0:
                raise ValueError("a shed tilt beyond 40 degrees will not get the head back level nicely")
        elif cmd == "kandi_damp":  # how much movement to take out while it is wearing one (0 = none, 1 = still)
            damp = float(value)
            if not 0.0 <= damp <= 1.0:
                raise ValueError(f"kandi damping runs from 0 (move normally) to 1 (hold still), not {value}")
            self.p.composer.kandi_damp = damp
        elif cmd == "train_plur":  # "all" for the whole routine, or one of the four poses by name; false to stop
            if value in (False, 0, None, "stop"):
                self.cancel_training(now)
            else:
                self.train_plur("all" if value is True else str(value), now)
        elif cmd == "forget_plur":  # throw the trained poses away and go back to the built-in rules
            self.signs.trained = {} if value in (True, None, "", "all") else {k: v for k, v in self.signs.trained.items() if k != str(value)}
            self.actions_log.append((now, "plur", "forgot trained poses"))
        elif cmd == "plur_trained":  # restoring the lot from the settings file
            if not isinstance(value, dict):
                raise ValueError("trained PLUR poses must be a mapping of step name to its numbers")
            for step in value:
                if step not in PLUR_STEPS:
                    raise KeyError(f"unknown PLUR step '{step}'. Known: {', '.join(PLUR_STEPS)}")
            self.signs.trained = {k: {n: float(x) for n, x in v.items()} for k, v in value.items()}
        elif cmd == "dance_along":
            self.dance_along = bool(value)
        elif cmd == "singing":
            self.singing_enabled = bool(value)  # the brain picks songs when it feels like one (drives.py)
        elif cmd == "sing":  # value: true (its own choice), or a style name
            if self.asleep:
                raise ValueError("asleep")
            style = None if value in (True, None, "") else str(value)
            if style is not None and style not in songs.STYLES:
                raise ValueError(f"song style must be one of {', '.join(songs.STYLES)}, not '{value}'")
            self.sing(now, style=style)
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
        elif cmd == "record":  # the black box on or off (on by default; it is a ring, it never fills up)
            self.trace.enabled = bool(value)
            self.trace.event(now, "note", "recording " + ("on" if self.trace.enabled else "off"))
        elif cmd == "mark":  # "it just did the thing, NOW": a labelled line to find in the file
            self.trace.mark(now, "" if value in (True, None) else str(value))
        elif cmd == "clear_trace":
            self.trace.clear(now)
        elif cmd == "forget":
            self.p.memory.people.clear()
            self.p.memory._dirty = True
            self.p.memory.save(force=True)
        else:
            raise KeyError(f"unknown control '{cmd}'")
        self.save_settings()
        return {"ok": True}

    # ------------------------------------------------------------------ settings persistence
    _SETTING_KEYS = ("voice", "muted", "pickup", "ears", "dance_along", "mimic_flip", "groove_scale", "manual_groove", "keymap", "camera_lag_ms", "head_forward_mm", "singing", "imu_rub_gyro", "groove_bob", "groove_sway", "groove_body", "groove_ears", "kandi_roll", "kandi_damp", "plur_trained", "scratch_onset_ratio", "scratch_floor", "rub_level_ratio", "rub_flatness_min", "rub_floor", "match_threshold", "body_finder")

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
                    if layer not in LAYERS:  # a layer that has since been dropped (the 0.7.x "caring" one)
                        logger.warning("settings: no '%s' layer any more; its keys are free for the layers that are left", layer)
                        continue
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

    # The Dev tab's terminal needs xterm.js, which is vendored into the package: there is no network at
    # the festival to fetch it from, and the page has to work with the robot on its own hotspot.
    VENDOR = {"xterm.js": "application/javascript", "xterm-addon-fit.js": "application/javascript",
              "xterm.css": "text/css", "LICENSE-xterm.txt": "text/plain"}

    @app.get("/static/vendor/{name}")
    def vendor(name: str):
        from fastapi.responses import FileResponse

        if name not in VENDOR:  # an explicit list, not a path join: this one takes a name off the network
            raise HTTPException(status_code=404, detail=f"no vendored file '{name}'")
        return FileResponse(Path(__file__).resolve().parent / "static" / "vendor" / name, media_type=VENDOR[name])

    # ---------------------------------------------------------------- the Dev tab
    # A terminal on the robot, so a Claude Code session can see the camera, the IMU, the trace and the
    # code, and drive the pet through its own API, instead of any of it being described second-hand.
    # Off until somebody sets a passphrase: the rest of this page is sliders, but this is a shell.
    @app.get("/api/dev")
    def dev_status() -> dict:
        return pet.dev.status()

    @app.post("/api/dev")
    def dev_control(c: Control) -> dict:
        d = pet.dev
        try:
            if c.cmd == "enable":
                d.enable(str(c.value))
            elif c.cmd == "disable":
                d.disable()
            elif c.cmd == "key":
                d.set_key(str(c.value))
            elif c.cmd == "clear_key":
                d.clear_key()
            else:
                raise KeyError(f"unknown dev command '{c.cmd}'")
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"ok": True, **d.status()}

    @app.websocket("/ws/dev")
    async def dev_terminal(ws: WebSocket) -> None:
        """The pty, both ways. The passphrase is checked here, on every connection, not in the page."""
        import asyncio

        await ws.accept()
        d = pet.dev
        if not d.allows(ws.query_params.get("token")):
            await ws.send_text("\r\n[the dev console is off, or that passphrase is wrong]\r\n")
            await ws.close()
            return
        cmd, cols, rows = ws.query_params.get("cmd", ""), int(ws.query_params.get("cols", 100)), int(ws.query_params.get("rows", 30))
        if not d.running or cmd:
            d.start(cmd, cols, rows)
        else:
            d.resize(cols, rows)
        loop = asyncio.get_running_loop()

        async def pump() -> None:  # the terminal talking: read in a thread, the fd is blocking
            while True:
                data = await loop.run_in_executor(None, d.read)
                if not data:
                    break
                await ws.send_bytes(data)

        out = asyncio.create_task(pump())
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    d.write(msg["bytes"])
                elif msg.get("text"):
                    text = msg["text"]
                    if text.startswith("\x00resize:"):  # the only out-of-band message: a window size
                        c, _, r = text[len("\x00resize:"):].partition(",")
                        d.resize(int(c), int(r))
                    else:
                        d.write(text.encode())
        except Exception:
            logger.debug("dev console socket closed", exc_info=True)
        finally:
            out.cancel()
            try:
                await ws.close()
            except Exception:
                pass

    @app.get("/api/trace.jsonl")
    def trace(last_s: float | None = None, name: str = "festival-pet-trace"):
        """The black box, as a download. One JSON object per line: a header, then rows oldest first.

        ``last_s`` trims it to the last N seconds, for when the interesting thing just happened and the
        ring is holding an hour of it standing about.
        """
        from fastapi.responses import StreamingResponse

        now = time.time()
        pet.trace.header = {"build": build_info(), "controls": pet.mind()["controls"],
                            "memory": pet.p.memory.summary(), "log_tail": list(LOG_RING.lines)[-60:]}
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
        body = (line + "\n" for line in pet.trace.lines(now, last_s))
        return StreamingResponse(body, media_type="application/x-ndjson",
                                 headers={"Content-Disposition": f'attachment; filename="{name}-{stamp}.jsonl"'})

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
