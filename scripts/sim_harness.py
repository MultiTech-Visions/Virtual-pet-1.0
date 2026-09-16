"""Drive the pet against the SDK's MuJoCo simulator with scripted fake senses.

Start the simulated daemon first (from the SDK):
    MUJOCO_GL=disable reachy-mini-daemon --sim --headless --no-wake-up-on-start
then:
    python scripts/sim_harness.py [--vosk /path/to/vosk-model-small-en-us-0.15] [--speed 1.0]

The harness owns time: it runs the control loop in real time against the sim
(so head motion is really executed), while a scripted timeline injects faces,
IMU readings, mic audio (real synthesized speech + a click track + scratches)
and reports which behaviors fired. It exits non-zero if an expected event is
missing, so it doubles as an integration test.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reachy_mini import ReachyMini  # noqa: E402
from reachy_mini.media.camera_constants import ReachyMiniWirelessCamSpecs  # noqa: E402
from reachy_mini.media.camera_utils import intrinsics_for_size  # noqa: E402
from reachy_mini.motion.recorded_move import DEFAULT_EMOTIONS_DATASET, RecordedMoves  # noqa: E402
from reachy_mini.vision.look_at import default_head_to_camera_transform  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from festival_pet.behavior import Behavior  # noqa: E402
from festival_pet.main import AUDIO_RATE, Pet, PetParts, ReachyIO  # noqa: E402
from festival_pet.memory import FaceMemory  # noqa: E402
from festival_pet.motion import MotionComposer  # noqa: E402
from festival_pet.vision import Sighting  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
FRAME_W, FRAME_H = 1280, 720


# ----------------------------------------------------------------------------- fake senses
class Timeline:
    """Scripted world: what is true at time t (seconds since start)."""

    def __init__(self) -> None:
        self.face: list[tuple] = []  # (t0, t1, yaw, pitch, area[, yaw_end]) — yaw_end makes the person walk sideways
        self.held: list[tuple[float, float]] = []
        self.shake: list[tuple[float, float]] = []
        self.touch: list[tuple[float, float]] = []
        self.audio = np.zeros(int(AUDIO_RATE * 120), dtype=np.float32)
        self.speech_flags: list[tuple[float, float, float]] = []  # (t0, t1, doa angle)

    def add_audio(self, at: float, wav: np.ndarray, gain: float = 1.0) -> None:
        i = int(at * AUDIO_RATE)
        self.audio[i : i + len(wav)] += gain * wav[: len(self.audio) - i]

    def add_speech(self, at: float, name: str, doa: float = math.pi / 2) -> None:
        d, sr = sf.read(FIXTURES / f"{name}.wav", dtype="float32")
        assert sr == AUDIO_RATE
        self.add_audio(at, d, 0.7)
        self.speech_flags.append((at - 0.1, at + len(d) / sr + 0.2, doa))

    def add_music(self, t0: float, t1: float, bpm: float) -> None:
        n = int((t1 - t0) * AUDIO_RATE)
        rng = np.random.default_rng(0)
        x = rng.standard_normal(n).astype(np.float32) * 0.01
        period = int(AUDIO_RATE * 60.0 / bpm)
        t = np.arange(int(0.05 * AUDIO_RATE)) / AUDIO_RATE
        kick = (np.sin(2 * np.pi * 170 * t) * np.exp(-t * 60)).astype(np.float32)
        for s in range(0, n - len(kick), period):
            x[s : s + len(kick)] += kick
        tt = np.arange(n) / AUDIO_RATE
        x += 0.05 * np.sin(2 * np.pi * 220 * tt).astype(np.float32)
        self.add_audio(t0, x, 0.8)

    def add_scratch(self, at: float, n_clicks: int = 6, gap: float = 0.11) -> None:
        rng = np.random.default_rng(7)
        for k in range(n_clicks):
            i = int((at + k * gap + rng.uniform(-0.02, 0.02)) * AUDIO_RATE)
            dur = int(0.004 * AUDIO_RATE)
            burst = np.diff(np.concatenate([[0.0], rng.standard_normal(dur)])).astype(np.float32)
            self.audio[i : i + dur] += 0.9 * burst / np.max(np.abs(burst))

    @staticmethod
    def _active(spans, t):
        return any(a <= t < b for a, b, *_ in spans)

    def face_at(self, t):
        for entry in self.face:
            t0, t1, yaw, pitch, area = entry[:5]
            if t0 <= t < t1:
                if len(entry) == 6:
                    yaw = yaw + (entry[5] - yaw) * (t - t0) / (t1 - t0)
                return yaw, pitch, area
        return None

    def speech_at(self, t):
        for t0, t1, doa in self.speech_flags:
            if t0 <= t < t1:
                return doa
        return None


class FakeIO:
    """RobotIO that forwards motion to the sim and fabricates every sense from the Timeline."""

    def __init__(self, reachy: ReachyMini, tl: Timeline, t_start: float) -> None:
        self._r = reachy
        self._tl = tl
        self._t0 = t_start
        specs = ReachyMiniWirelessCamSpecs()
        self.K = intrinsics_for_size(specs.K, specs.default_resolution.value[3], (FRAME_W, FRAME_H))
        self.D = specs.D
        self.T_head_cam = default_head_to_camera_transform()
        self._audio_pos = 0
        self._commanded = [-0.1745, 0.1745]
        self.set_target_calls = 0

    def t(self):
        return time.time() - self._t0

    def head_pose(self):
        return self._r.get_current_head_pose()

    def imu(self):
        t = self.t()
        accel, gyro = [0.0, 0.0, 9.81], [0.0, 0.0, 0.0]
        for t0, t1 in self._tl.held:
            if t0 <= t < t1:
                if t - t0 < 1.0:  # the lift itself: a strong transient
                    accel = [1.0 * math.sin(9 * t), 0.5 * math.cos(7 * t), 9.81 + 3.5 * math.sin(11 * t)]
                    gyro = [1.6 * math.sin(5 * t), 0.3, 0.0]
                else:  # held in hands: mild jitter
                    accel = [0.3 * math.sin(9 * t), 0.2 * math.cos(7 * t), 9.81 + 1.0 * math.sin(11 * t)]
                    gyro = [0.5 * math.sin(5 * t), 0.1, 0.0]
        if Timeline._active(self._tl.shake, t):
            gyro = [6.0 * math.sin(30 * t), 0.0, 0.0]
        return {"accelerometer": accel, "gyroscope": gyro, "quaternion": [1, 0, 0, 0], "temperature": 30.0}

    def present_antennas(self):
        present = list(self._r.get_present_antenna_joint_positions())
        if Timeline._active(self._tl.touch, self.t()):
            present[1] += 0.6  # someone is pushing the left antenna
        return present

    def doa(self):
        d = self._tl.speech_at(self.t())
        return (d, True) if d is not None else (math.pi / 2, False)

    def audio_chunk(self):
        # Deliver audio in real time, in 20 ms chunks.
        target = int(self.t() * AUDIO_RATE)
        if target - self._audio_pos < 320:
            return None
        chunk = self._tl.audio[self._audio_pos : self._audio_pos + 320]
        self._audio_pos += 320
        return chunk

    def play(self, buf):
        pass  # no speakers in the sim; SoundPlayer.played records what would have sounded

    def play_file(self, path):
        pass

    def set_target(self, head, antennas, body_yaw):
        self.set_target_calls += 1
        self._commanded = list(antennas)
        self._r.set_target(head=head, antennas=antennas, body_yaw=body_yaw)

    # Same daemon moves as on the real robot (the sim daemon serves them too).
    _daemon_move = ReachyIO._daemon_move
    sleep_body = ReachyIO.sleep_body
    wake_body = ReachyIO.wake_body

    def goto(self, head, antennas, duration):
        self._r.goto_target(head=head, antennas=antennas, duration=duration)


class FakeVision:
    """Turns the Timeline's face (a world direction) into pixel Sightings like the real detector would."""

    def __init__(self, io: FakeIO, tl: Timeline, memory: FaceMemory) -> None:
        self._io, self._tl, self._mem = io, tl, memory
        self._track = 1
        self._last_face_t = -10.0
        self._person = None

    def latest(self):
        f = self._tl.face_at(self._io.t())
        now = time.time()
        if f is None:
            return None
        yaw, pitch, area = f
        if self._io.t() - self._last_face_t > 4.0:  # mirrors Vision.FORGET_AFTER
            self._track += 1
        self._last_face_t = self._io.t()
        head = self._io.head_pose()
        ray_world = R.from_euler("xyz", [0, pitch, yaw], degrees=True).as_matrix() @ np.array([1.0, 0.0, 0.0])
        ray_cam = (head @ self._io.T_head_cam)[:3, :3].T @ ray_world
        if ray_cam[2] <= 0.05:
            return None  # behind the camera
        u = self._io.K[0, 0] * ray_cam[0] / ray_cam[2] + self._io.K[0, 2]
        v = self._io.K[1, 1] * ray_cam[1] / ray_cam[2] + self._io.K[1, 2]
        if not (0 < u < FRAME_W and 0 < v < FRAME_H):
            return None  # out of the field of view
        return Sighting(self._track, float(u), float(v), area, self._person, 0.0, head, now, 0.0)


# ----------------------------------------------------------------------------- scenario
def build_timeline() -> Timeline:
    tl = Timeline()
    tl.face += [(3.0, 5.0, 25.0, -5.0, 0.05)]  # stranger appears to the left
    tl.face += [(5.0, 9.0, 25.0, -5.0, 0.05, 60.0)]  # ...and walks further left: the body must follow to keep them
    tl.face += [(9.0, 12.5, 60.0, 0.0, 0.05, -10.0)]  # ...then walks back across to the right
    tl.face += [(14.5, 20.0, -10.0, 0.0, 0.05)]  # peekaboo: hidden 12.5-14.5 (2 s, longer than a detector dropout)
    tl.add_speech(21.0, "m3_reachy")
    tl.add_speech(23.5, "m3_reachy_dance")
    tl.add_speech(27.0, "m3_reachy_dance")
    tl.face += [(20.0, 34.0, 0.0, 0.0, 0.05)]
    tl.add_music(36.0, 60.0, 112.0)
    tl.add_scratch(50.0)
    tl.touch += [(54.0, 54.6)]
    tl.held += [(56.0, 66.0)]
    tl.shake += [(60.0, 60.5)]
    tl.face += [(68.0, 72.0, 5.0, 0.0, 0.05)]
    tl.add_speech(74.0, "m3_reachy")  # "Reachy ... sleep" -> motors off
    tl.add_speech(76.0, "hello_there")  # (only 'hello' is in the fixtures; the sleep command is injected below)
    tl.add_speech(84.0, "m3_reachy")  # its name wakes it back up
    return tl


SLEEP_CMD_AT = 77.5  # the fixtures have no spoken "sleep"; the harness injects the command word here

EXPECTED = [  # (window t0, t1, kind, name)
    (0.0, 2.0, "wake", "start"),
    (3.0, 6.0, "sound", "hello_new"),
    (14.4, 16.0, "sound", "giggle"),  # peekaboo
    (21.0, 23.5, "sound", "name"),
    (23.5, 27.0, "gesture", "bounce"),  # little dance
    (27.0, 30.0, "move", "dance"),  # lively dance
    (36.0, 50.0, "sound", "happy"),  # noticed the music
    (50.0, 53.0, "sound", "ticklish"),
    (54.0, 56.0, "sound", "giggle|ticklish"),  # ear tickle
    (56.0, 58.0, "sound", "surprised"),  # picked up
    (58.0, 66.0, "sound", "purr|content"),
    (60.0, 63.0, "sound", "dizzy"),
    (66.0, 69.0, "gesture", "shake_off"),  # set down
    (77.0, 80.0, "sleep", "asked"),  # told to sleep: motors off
    (83.0, 88.0, "wake", "name|sound"),  # its voice/name wakes it (the loud-voice startle may win the race)
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vosk", default=os.environ.get("VOSK_MODEL", ""))
    ap.add_argument("--duration", type=float, default=90.0)
    args = ap.parse_args()

    reachy = ReachyMini(media_backend="no_media", connection_mode="localhost_only")
    tl = build_timeline()
    t_start = time.time()
    io = FakeIO(reachy, tl, t_start)
    memory = FaceMemory(Path("/tmp/festival_pet_sim_memory.json"))
    memory.people.clear()
    library = RecordedMoves(DEFAULT_EMOTIONS_DATASET)
    vision = FakeVision(io, tl, memory)
    spotter = None
    if args.vosk:
        from festival_pet.hearing import NameSpotter

        spotter = NameSpotter(Path(args.vosk))
    else:
        print("!! no --vosk model: name/command events will be missing")
    pet = Pet(PetParts(io, memory, library.get, vision.latest, spotter, Behavior(memory), MotionComposer()))
    pet.pickup_enabled = True  # off by default on the robot (IMU is in the head); the scenario scripts a pickup

    stop = threading.Event()
    last_state = None
    groove_seen = False
    gaze_err: list[tuple[float, float]] = []
    music_groove = False
    asleep_seen = False
    pet.start(time.time())
    try:
        sleep_injected = False
        while io.t() < args.duration:
            now = time.time()
            if not sleep_injected and io.t() >= SLEEP_CMD_AT:
                sleep_injected = True
                pet.audio._events.put(("word", "sleep", None))
            pet.step(now)
            if pet.asleep and not asleep_seen:
                asleep_seen = True
                print(f"[{io.t():6.2f}] asleep: motors off, vision paused")
            st = pet.p.behavior.state
            if st != last_state:
                print(f"[{io.t():6.2f}] state -> {st}")
                last_state = st
            f = tl.face_at(io.t())
            if f is not None and int(io.t() * 2) != int((io.t() - 0.02) * 2) and int(io.t()) % 2 == 0:
                roll, pitch, yaw = R.from_matrix(io.head_pose()[:3, :3]).as_euler("xyz", degrees=True)
                gaze_err.append((abs(yaw - f[0]), abs(pitch - f[1])))
                print(f"[{io.t():6.2f}] face at yaw {f[0]:+.0f} pitch {f[1]:+.0f} | sim head yaw {yaw:+.1f} pitch {pitch:+.1f}")
            grooving_now = pet.p.composer.groove is not None
            if grooving_now and not groove_seen:
                b = pet.audio.beat.state
                src = f"music {b.bpm:.1f} bpm (conf {b.confidence:.2f})" if pet.audio.beat.music else "inner tempo (no music)"
                print(f"[{io.t():6.2f}] grooving to {src}, intensity {pet.p.composer.groove[2]:.2f}")
                if pet.audio.beat.music:
                    music_groove = True
            groove_seen = grooving_now
            time.sleep(max(0.0, 0.02 - (time.time() - now)))
    finally:
        pet.stop()

    log = [(t - t_start, k, n) for t, k, n in pet.actions_log]
    print("\nactions:")
    for t, k, n in log:
        print(f"  [{t:6.2f}] {k}:{n}")
    b = pet.audio.beat.state
    print(f"\nmusic groove seen: {music_groove}; set_target calls: {io.set_target_calls}")
    ok = True
    for t0, t1, kind, name in EXPECTED:
        names = name.split("|")
        hit = any(t0 <= t < t1 and k == kind and any(n.startswith(x) for x in names) for t, k, n in log)
        if kind in ("move",) or "dance" in name:
            hit = hit or spotter is None  # can't expect speech-driven tricks without a model
        if name == "name":
            hit = hit or spotter is None
        print(f"  {'OK ' if hit else 'MISS'} {t0:5.1f}-{t1:5.1f} {kind}:{name}")
        ok &= hit
    ok &= music_groove
    ok &= asleep_seen and not pet.asleep  # slept and woke back up
    print(f"slept: {asleep_seen}, awake at end: {not pet.asleep}")
    settled = [e for e in gaze_err[2:]]  # skip the first samples while the head is still turning
    if settled:
        yaw_err = float(np.median([e[0] for e in settled]))
        print(f"gaze: median |yaw error| {yaw_err:.1f} deg over {len(settled)} samples (gestures add a few deg)")
        ok &= yaw_err < 8.0
    print("RESULT:", "PASS" if ok else "FAIL")
    os._exit(0 if ok else 1)  # skip the SDK's slow websocket teardown


if __name__ == "__main__":
    main()
