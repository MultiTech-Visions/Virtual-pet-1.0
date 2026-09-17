"""A paired Bluetooth (or USB) keyboard as a hand controller: taps, tilts, downbeats.

Reads Linux input events straight from ``/dev/input/event*`` with no extra package: every
device whose capabilities say "keyboard" is opened, and key presses/releases are queued with
the kernel's own timestamp (good to the millisecond, which is what tapping a beat needs).
Devices that come and go (a keypad going to sleep, a fresh pairing) are picked up on the next
rescan. The pet's user must be able to read ``/dev/input`` (group ``input``; the restore
script adds it).

``KeyMap`` turns key names into pet actions: a short press does one thing, a press held for
``HOLD_S`` does another. Only the keys named in the map do anything; the rest are ignored.
"""

from __future__ import annotations

import os
import queue
import select
import struct
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

INPUT_DIR = Path("/dev/input")
SYS_INPUT = Path("/sys/class/input")
EVENT_FMT = "llHHi"  # struct input_event: timeval (2 longs), type, code, value
EVENT_SIZE = struct.calcsize(EVENT_FMT)
EV_KEY = 0x01
KEY_A = 30  # a device that reports KEY_A is a keyboard for our purposes

# Linux keycodes for the keys anyone would plausibly map (evdev names without KEY_).
KEYCODES = {
    "ESC": 1, "1": 2, "2": 3, "3": 4, "4": 5, "5": 6, "6": 7, "7": 8, "8": 9, "9": 10, "0": 11, "MINUS": 12, "EQUAL": 13,
    "BACKSPACE": 14, "TAB": 15, "Q": 16, "W": 17, "E": 18, "R": 19, "T": 20, "Y": 21, "U": 22, "I": 23, "O": 24, "P": 25,
    "ENTER": 28, "LEFTCTRL": 29, "A": 30, "S": 31, "D": 32, "F": 33, "G": 34, "H": 35, "J": 36, "K": 37, "L": 38,
    "LEFTSHIFT": 42, "Z": 44, "X": 45, "C": 46, "V": 47, "B": 48, "N": 49, "M": 50, "COMMA": 51, "DOT": 52, "SLASH": 53,
    "RIGHTSHIFT": 54, "LEFTALT": 56, "SPACE": 57, "CAPSLOCK": 58,
    "F1": 59, "F2": 60, "F3": 61, "F4": 62, "F5": 63, "F6": 64, "F7": 65, "F8": 66, "F9": 67, "F10": 68, "F11": 87, "F12": 88,
    "KP7": 71, "KP8": 72, "KP9": 73, "KPMINUS": 74, "KP4": 75, "KP5": 76, "KP6": 77, "KPPLUS": 78, "KP1": 79, "KP2": 80,
    "KP3": 81, "KP0": 82, "KPDOT": 83, "KPENTER": 96, "RIGHTCTRL": 97, "KPSLASH": 98, "RIGHTALT": 100,
    "HOME": 102, "UP": 103, "PAGEUP": 104, "LEFT": 105, "RIGHT": 106, "END": 107, "DOWN": 108, "PAGEDOWN": 109,
    "INSERT": 110, "DELETE": 111, "MUTE": 113, "VOLUMEDOWN": 114, "VOLUMEUP": 115, "PLAYPAUSE": 164, "NEXTSONG": 163, "PREVIOUSSONG": 165,
}
KEYNAMES = {v: k for k, v in KEYCODES.items()}

ACTIONS = ("tap", "downbeat", "tilt_left", "tilt_right", "nod", "manual_groove", "wake", "sleep", "mute", "happy", "none")
HOLD_S = 0.8  # hold a key this long for its hold action

# The PCsensor MK424 keypad sends A B C D from the factory; a QWERTY keyboard gets the same plus obvious extras.
DEFAULT_KEYMAP: dict[str, dict[str, str]] = {
    "A": {"press": "tilt_left", "hold": "manual_groove"},
    "B": {"press": "tap", "hold": "none"},
    "C": {"press": "downbeat", "hold": "none"},
    "D": {"press": "tilt_right", "hold": "manual_groove"},
    "SPACE": {"press": "tap", "hold": "none"},
    "ENTER": {"press": "downbeat", "hold": "none"},
    "LEFT": {"press": "tilt_left", "hold": "none"},
    "RIGHT": {"press": "tilt_right", "hold": "none"},
    "UP": {"press": "nod", "hold": "wake"},
    "DOWN": {"press": "none", "hold": "sleep"},
    "M": {"press": "mute", "hold": "none"},
}


@dataclass
class KeyEvent:
    key: str  # name from KEYNAMES
    down: bool
    t: float  # kernel timestamp, same clock as time.time()
    device: str


def keyboard_devices() -> list[tuple[Path, str]]:
    """(event node, device name) for every input device that reports KEY_A."""
    out = []
    if not SYS_INPUT.exists():
        return out
    for d in sorted(SYS_INPUT.glob("event*")):
        try:
            caps = (d / "device" / "capabilities" / "key").read_text().split()
            name = (d / "device" / "name").read_text().strip()
        except OSError:
            continue
        # capabilities/key is space-separated hex words, most significant first
        bits = int("".join(w.zfill(16) for w in caps), 16)
        if bits >> KEY_A & 1:
            out.append((INPUT_DIR / d.name, name))
    return out


class KeypadListener:
    """Thread that reads every keyboard-like device and queues KeyEvents."""

    def __init__(self, rescan_s: float = 2.0) -> None:
        self.events: queue.Queue[KeyEvent] = queue.Queue()
        self.devices: dict[str, str] = {}  # node -> name, currently open
        self.last_key: tuple[str, float] | None = None
        self.error: str | None = None
        self._rescan = rescan_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="festival-pet-keypad", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        fds: dict[int, tuple[str, str]] = {}  # fd -> (node, name)
        next_scan = 0.0
        while not self._stop.is_set():
            now = time.time()
            if now >= next_scan:
                next_scan = now + self._rescan
                open_nodes = {node for node, _ in fds.values()}
                for node, name in keyboard_devices():
                    if str(node) in open_nodes:
                        continue
                    try:
                        fd = os.open(node, os.O_RDONLY | os.O_NONBLOCK)
                    except PermissionError as e:
                        self.error = f"{node}: {e.strerror} (is the pet's user in the 'input' group?)"
                        continue
                    except OSError:
                        continue
                    fds[fd] = (str(node), name)
                    self.error = None
                self.devices = {node: name for node, name in fds.values()}
            if not fds:
                time.sleep(0.2)
                continue
            readable, _, _ = select.select(list(fds), [], [], 0.25)
            for fd in readable:
                node, name = fds[fd]
                try:
                    data = os.read(fd, EVENT_SIZE * 64)
                except OSError:  # device went away
                    os.close(fd)
                    del fds[fd]
                    self.devices = {n: nm for n, nm in fds.values()}
                    continue
                for off in range(0, len(data) - EVENT_SIZE + 1, EVENT_SIZE):
                    sec, usec, etype, code, value = struct.unpack_from(EVENT_FMT, data, off)
                    if etype != EV_KEY or value == 2 or code not in KEYNAMES:  # 2 = auto-repeat
                        continue
                    ev = KeyEvent(KEYNAMES[code], value == 1, sec + usec / 1e6, name)
                    if ev.down:
                        self.last_key = (ev.key, ev.t)
                    self.events.put(ev)
        for fd in fds:
            os.close(fd)


@dataclass
class KeyMap:
    """Press/hold semantics on top of raw key events. ``resolve`` returns the action for a completed press."""

    keys: dict[str, dict[str, str]] = field(default_factory=lambda: {k: dict(v) for k, v in DEFAULT_KEYMAP.items()})
    _down_since: dict[str, float] = field(default_factory=dict)
    _hold_fired: set[str] = field(default_factory=set)

    def set(self, key: str, press: str, hold: str) -> None:
        key = key.upper()
        if key not in KEYCODES:
            raise KeyError(f"unknown key '{key}'")
        for a in (press, hold):
            if a not in ACTIONS:
                raise KeyError(f"unknown action '{a}'")
        if press == "none" and hold == "none":
            self.keys.pop(key, None)
        else:
            self.keys[key] = {"press": press, "hold": hold}

    def feed(self, ev: KeyEvent, now: float) -> tuple[str, float] | None:
        """Returns (action, event time) when a press or hold completes, else None.

        A press action fires on key-down (so tapping a beat has no release latency) unless the key also
        has a hold action, in which case a short press fires on release and a long one fires the hold.
        """
        m = self.keys.get(ev.key)
        if m is None:
            return None
        if ev.down:
            self._down_since[ev.key] = ev.t
            self._hold_fired.discard(ev.key)
            if m["hold"] == "none" and m["press"] != "none":
                return (m["press"], ev.t)
            return None
        t0 = self._down_since.pop(ev.key, None)
        if t0 is None or ev.key in self._hold_fired:
            return None
        if m["hold"] != "none" and ev.t - t0 >= HOLD_S:
            return (m["hold"], t0)
        if m["hold"] != "none" and m["press"] != "none":
            return (m["press"], t0)
        return None

    def tick(self, now: float) -> list[tuple[str, float]]:
        """Fire hold actions as soon as the hold time passes, without waiting for the release."""
        out = []
        for key, t0 in list(self._down_since.items()):
            m = self.keys.get(key)
            if m is not None and m["hold"] != "none" and key not in self._hold_fired and now - t0 >= HOLD_S:
                self._hold_fired.add(key)
                out.append((m["hold"], t0))
        return out


# ----------------------------------------------------------------------------- Bluetooth pairing (bluez)


def _bluetoothctl(commands: list[str], timeout: float, answer_pin: str | None = None) -> str:
    """Run a scripted bluetoothctl session; answers PIN / passkey prompts. Raises on a missing bluetoothctl."""
    proc = subprocess.Popen(["bluetoothctl"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    out: list[str] = []
    stop = threading.Event()

    def reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            out.append(line)
            low = line.lower()
            if answer_pin and ("enter pin code" in low or "enter passkey" in low):
                proc.stdin.write(answer_pin + "\n")  # type: ignore[union-attr]
                proc.stdin.flush()  # type: ignore[union-attr]
            elif "confirm passkey" in low or "authorize service" in low:
                proc.stdin.write("yes\n")  # type: ignore[union-attr]
                proc.stdin.flush()  # type: ignore[union-attr]
            if stop.is_set():
                break

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    assert proc.stdin is not None
    for c in commands:
        if c.startswith("sleep "):
            time.sleep(float(c.split()[1]))
            continue
        proc.stdin.write(c + "\n")
        proc.stdin.flush()
        time.sleep(0.3)
    proc.stdin.write("quit\n")
    proc.stdin.flush()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
    stop.set()
    return "".join(out)


def bt_status() -> dict:
    """Controller power state and the paired devices with their connection state."""
    text = _bluetoothctl(["show", "devices Paired"], timeout=10)
    powered = "Powered: yes" in text
    devices = []
    for line in text.splitlines():
        if line.startswith("Device "):
            _, mac, name = line.split(" ", 2)
            info = _bluetoothctl([f"info {mac}"], timeout=10)
            devices.append({"mac": mac, "name": name.strip(), "connected": "Connected: yes" in info})
    return {"powered": powered, "devices": devices}


def bt_scan(seconds: float = 8.0) -> list[dict]:
    """Scan for discoverable devices; returns those with a name (unnamed ones are not keyboards in pairing mode)."""
    text = _bluetoothctl(["power on", "agent KeyboardOnly", "default-agent", "scan on", f"sleep {seconds}", "scan off", "devices"], timeout=seconds + 15)
    found = {}
    for line in text.splitlines():
        line = line.strip().lstrip("[NEW] [CHG] ")
        if line.startswith("Device "):
            _, mac, name = line.split(" ", 2)
            name = name.strip()
            if name and name.replace("-", ":").upper() != mac:
                found[mac] = name
    return [{"mac": m, "name": n} for m, n in found.items()]


def bt_pair(mac: str, pin: str = "1234") -> str:
    """Pair, trust (auto-reconnect) and connect. Returns bluetoothctl's transcript tail for the page."""
    text = _bluetoothctl(["power on", "agent KeyboardOnly", "default-agent", f"pair {mac}", "sleep 6", f"trust {mac}", f"connect {mac}", "sleep 3"], timeout=30, answer_pin=pin)
    if "Pairing successful" not in text and "Connection successful" not in text and "already exists" not in text:
        raise RuntimeError("pairing failed:\n" + "\n".join(text.strip().splitlines()[-8:]))
    return "\n".join(text.strip().splitlines()[-6:])


def bt_forget(mac: str) -> None:
    _bluetoothctl([f"remove {mac}"], timeout=10)
