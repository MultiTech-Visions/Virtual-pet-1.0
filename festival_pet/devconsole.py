"""A real terminal on the robot, in the robot's own web page: the Dev tab.

The point is to stop debugging this thing by description. A Claude Code session running ON the robot
can read the camera, the IMU, the trace and the mind API, drive the pet through its own HTTP API, edit
the code and restart the app — all the things that, from a chat window, have to be guessed at.

This is the plumbing: a pseudo-terminal running a shell (or ``claude`` directly) on the robot, wired
to xterm.js in the browser over a WebSocket. It is the real CLI, so the chat, the tool calls, the
diffs and the permission prompts are all exactly as they are in a terminal.

Two things it deliberately does NOT do:

- It does not work without a network. Claude Code talks to Anthropic's API; at a campsite with no
  signal there is nothing to talk to. This is for the bench, the van, a phone hotspot — and the app
  itself stays fully offline, as it must.
- It does not run unless somebody switches it on and sets a passphrase. The pet's page has no login:
  anybody on the same network can open it. That is fine for a page of sliders and a bad idea for a
  shell, so the console is off until enabled, the WebSocket refuses a wrong or missing passphrase,
  and both are re-checked on every connection rather than trusted from the page.

The API key is written to a 0600 file in the robot's data directory, is never logged, never appears
in the trace, and is only ever reported back as "set, ending ...abcd".
"""

from __future__ import annotations

import fcntl
import logging
import os
import pty
import secrets
import shutil
import signal
import struct
import subprocess
import termios
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SHELL = os.environ.get("SHELL", "/bin/bash")
# npm's global prefix on a Debian/apt node is /usr/local, which the pollen user cannot write: without
# sudo this dies with EACCES every time. `-i` so sudo asks on this terminal if it wants a password.
INSTALL_CMD = "sudo -i npm install -g @anthropic-ai/claude-code && claude --version"
KEY_ENV = "ANTHROPIC_API_KEY"
MIN_PASSPHRASE = 6
READ_SIZE = 65536


class DevConsole:
    """One pseudo-terminal, plus the key and the gate in front of it."""

    def __init__(self, data_dir: Path, workdir: Path | None = None) -> None:
        self.key_file = Path(data_dir) / "anthropic_key"
        self.workdir = Path(workdir) if workdir is not None else Path.cwd()
        self.enabled = False
        self.passphrase = ""
        self._fd: int | None = None
        self._pid: int | None = None
        self._cmd = ""
        self._status: int | None = None  # how the last session ended; None while one is up

    # ------------------------------------------------------------------ the gate
    def enable(self, passphrase: str) -> None:
        """Switch the console on. A passphrase is required, not optional: this is a shell."""
        if len(passphrase or "") < MIN_PASSPHRASE:
            raise ValueError(f"the dev console needs a passphrase of at least {MIN_PASSPHRASE} characters: it is a shell on the robot")
        self.passphrase = passphrase
        self.enabled = True
        logger.warning("dev console ENABLED: anybody on this network with the passphrase has a terminal on this robot")

    def disable(self) -> None:
        self.enabled = False
        self.passphrase = ""
        self.stop()
        logger.info("dev console disabled")

    def allows(self, token: str | None) -> bool:
        """Constant-time check, and never true while the console is off or unset."""
        if not self.enabled or not self.passphrase:
            return False
        return secrets.compare_digest(str(token or ""), self.passphrase)

    # ------------------------------------------------------------------ the key
    def set_key(self, key: str) -> None:
        key = (key or "").strip()
        if not key:
            raise ValueError("no API key given")
        if not key.startswith("sk-"):
            raise ValueError("an Anthropic API key starts with 'sk-'; that does not look like one")
        self.key_file.parent.mkdir(parents=True, exist_ok=True)
        self.key_file.touch(mode=0o600, exist_ok=True)
        os.chmod(self.key_file, 0o600)  # ...and if it already existed with looser permissions
        self.key_file.write_text(key)

    def clear_key(self) -> None:
        self.key_file.unlink(missing_ok=True)

    def key(self) -> str | None:
        if not self.key_file.exists():
            return None
        return self.key_file.read_text().strip() or None

    # ------------------------------------------------------------------ what is installed
    def status(self) -> dict:
        """Everything the Dev tab needs to tell somebody what is still missing."""
        key = self.key()
        node = shutil.which("node")
        claude = shutil.which("claude")
        return {
            "enabled": self.enabled, "running": self.running, "cmd": self._cmd,
            "key_set": key is not None, "key_tail": None if key is None else key[-4:],
            "node": node, "node_version": _version(node, "--version"),
            "claude": claude, "claude_version": _version(claude, "--version"),
            "install_cmd": INSTALL_CMD, "workdir": str(self.workdir),
            "online": _online(),
        }

    # ------------------------------------------------------------------ the terminal
    @property
    def running(self) -> bool:
        if self._pid is None:
            return False
        try:
            gone, status = os.waitpid(self._pid, os.WNOHANG)
        except ChildProcessError:
            gone, status = self._pid, 0
        if gone:
            self._reap(status)
            return False
        return True

    def ended(self) -> str | None:
        """Why the last session finished, in a line for the terminal — or None if one is still up.

        A shell that dies on its first line (no such command, sudo refused, npm's EACCES) used to leave
        the Dev tab showing nothing at all but "[session closed]", which is the least useful thing a
        terminal can say. The exit status is always worth printing.
        """
        if self.running or self._status is None:
            return None
        code, sig = self._status >> 8, self._status & 0x7F
        if sig:
            return f"[the session was killed by signal {sig}]"
        if code:
            return f"[the session exited with status {code} — the last lines above say why]"
        return "[the session finished]"

    def start(self, cmd: str = "", cols: int = 100, rows: int = 30) -> None:
        """Fork a pty running ``cmd`` (a login shell by default). Replaces any session already up."""
        self.stop()
        argv = [DEFAULT_SHELL, "-l"] if not cmd else [DEFAULT_SHELL, "-lc", cmd]
        env = dict(os.environ)
        env.update({"TERM": "xterm-256color", "COLUMNS": str(cols), "LINES": str(rows),
                    "PAGER": "cat", "GIT_PAGER": "cat"})
        key = self.key()
        if key:
            env[KEY_ENV] = key
        pid, fd = pty.fork()
        if pid == 0:  # the child: this call never returns
            try:
                os.chdir(self.workdir)
                os.execvpe(argv[0], argv, env)
            except Exception as e:  # pragma: no cover - runs in the forked child
                # The pty is this child's stderr, so say it out loud rather than dying silently: a bad
                # $SHELL is otherwise an empty black box with no clue in it.
                os.write(2, f"\r\ncould not start {argv[0]}: {e}\r\n".encode())
                os._exit(127)
        self._pid, self._fd, self._cmd, self._status = pid, fd, cmd or "shell", None
        self.resize(cols, rows)
        logger.info("dev console started: %s (pid %d)", self._cmd, pid)

    def _reap(self, status: int | None = None) -> None:
        self._status = status
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
        self._fd, self._pid, self._cmd = None, None, ""

    def stop(self) -> None:
        """Ask, then insist. A shell that ignores SIGHUP must not be able to wedge the control loop."""
        import time as _time

        if self._pid is None:
            return
        pid = self._pid
        for sig in (signal.SIGHUP, signal.SIGKILL):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                break
            end = _time.monotonic() + 2.0
            while _time.monotonic() < end:
                try:
                    gone, status = os.waitpid(pid, os.WNOHANG)
                    if gone:
                        self._reap(status)
                        return
                except ChildProcessError:
                    self._reap(0)
                    return
                _time.sleep(0.02)
        self._reap()

    @property
    def fd(self) -> int | None:
        return self._fd

    def write(self, data: bytes) -> None:
        if self._fd is not None:
            os.write(self._fd, data)

    def read(self) -> bytes:
        """Whatever the terminal has said. b"" means it has finished."""
        if self._fd is None:
            return b""
        try:
            return os.read(self._fd, READ_SIZE)
        except OSError:
            return b""

    def resize(self, cols: int, rows: int) -> None:
        if self._fd is None:
            return
        cols, rows = max(20, min(500, int(cols))), max(5, min(200, int(rows)))
        fcntl.ioctl(self._fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


def _version(path: str | None, flag: str) -> str | None:
    if not path:
        return None
    try:
        out = subprocess.run([path, flag], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr).strip() else None


def _online(host: str = "api.anthropic.com", port: int = 443, timeout: float = 2.0) -> bool:
    """Can it reach Anthropic at all? At a festival the honest answer is no, and the page should say so."""
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False
