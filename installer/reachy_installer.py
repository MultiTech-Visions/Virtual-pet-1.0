"""Festival Pet installer / updater for Reachy Mini Wireless.

A small wizard window (tkinter, no extra runtime needed) that:
  1. asks for the robot's address and SSH password,
  2. checks it can reach the robot over SSH and that the robot has internet,
  3. uploads this app to the robot and runs scripts/setup_offline.sh there,
  4. makes it the app that starts when you touch an antenna, and starts it now.

Run it again any time to update: the setup script is idempotent (models and
libraries are only downloaded when missing, the app is upgraded in place).

Run from source:      python installer/reachy_installer.py
Built executables:    see .github/workflows/installer.yml (Windows + macOS)
"""

from __future__ import annotations

import io
import json
import os
import posixpath
import queue
import socket
import stat
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

import paramiko

APP_NAME = "festival_pet"
DEFAULT_HOST = "reachy-mini.local"
DEFAULT_USER = "pollen"
DEFAULT_PASSWORD = "root"  # factory default on the wireless unit; change it if you changed yours
REMOTE_DIR = f"/home/{DEFAULT_USER}/festival_pet"
DAEMON_PORT = 8000
GITHUB_ZIP = "https://github.com/MultiTech-Visions/Virtual-pet-1.0/archive/refs/heads/{branch}.tar.gz"
DEFAULT_BRANCH = "claude/reachy-festival-robot-xqb2ao"

# Files worth shipping; everything else in the repo (tests, fixtures, git) stays home.
SHIP = ("festival_pet", "dashboard", "scripts", "pyproject.toml", "README.md")


def source_root() -> Path | None:
    """The app source next to this installer (repo checkout or PyInstaller bundle), if any."""
    candidates = [Path(__file__).resolve().parents[1]]
    if getattr(sys, "_MEIPASS", None):
        candidates.insert(0, Path(sys._MEIPASS) / "app")  # type: ignore[attr-defined]
    for c in candidates:
        if (c / "pyproject.toml").exists() and (c / "festival_pet" / "main.py").exists():
            return c
    return None


def source_version(root: Path) -> str:
    text = (root / "pyproject.toml").read_text()
    for line in text.splitlines():
        if line.strip().startswith("version"):
            return line.split("=", 1)[1].strip().strip('"')
    raise ValueError("no version in pyproject.toml")


TEXT_SUFFIXES = (".sh", ".py", ".toml", ".md", ".html", ".txt", ".js", ".css")


def _add_file(tar: tarfile.TarFile, path: Path, arcname: str) -> None:
    """Add one file, converting CRLF to LF for text files (a Windows checkout would otherwise break bash on the robot)."""
    data = path.read_bytes()
    if path.suffix in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")
    info = tarfile.TarInfo(arcname)
    info.size = len(data)
    info.mtime = int(path.stat().st_mtime)
    info.mode = 0o755 if path.suffix == ".sh" else 0o644
    tar.addfile(info, io.BytesIO(data))


def git_commit(root: Path) -> str | None:
    """Short commit of the checkout, if this is one and git is around."""
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def build_stamp(root: Path) -> bytes:
    """festival_pet/_build.py: what exactly got uploaded, shown on the Mind page and in the app log."""
    built = time.strftime("%Y-%m-%d %H:%M")
    return f'"""Written by the installer at upload time."""\n\nCOMMIT = {git_commit(root)!r}\nBUILT = {built!r}\n'.encode()


def make_tarball(root: Path) -> bytes:
    """Tar the shippable parts of the app into memory, plus a build stamp."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        stamp = build_stamp(root)
        info = tarfile.TarInfo("festival_pet/_build.py")
        info.size, info.mtime, info.mode = len(stamp), int(time.time()), 0o644
        tar.addfile(info, io.BytesIO(stamp))
        for name in SHIP:
            path = root / name
            if not path.exists():
                raise FileNotFoundError(path)
            files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
            for f in files:
                rel = f.relative_to(root).as_posix()
                if "__pycache__" in rel or rel.endswith(".pyc") or rel == "festival_pet/_build.py":
                    continue
                _add_file(tar, f, rel)
    return buf.getvalue()


def download_source(branch: str, log) -> Path:
    """Fetch the latest app from GitHub into a temp folder (for updating without a checkout)."""
    import tempfile

    url = GITHUB_ZIP.format(branch=branch)
    log(f"downloading {url}")
    data = urllib.request.urlopen(url, timeout=60).read()
    tmp = Path(tempfile.mkdtemp(prefix="festival_pet_src_"))
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(tmp, filter="data")
    (root,) = [p for p in tmp.iterdir() if p.is_dir()]
    return root


class Robot:
    """SSH + REST access to one Reachy Mini Wireless."""

    def __init__(self, host: str, user: str, password: str, log, ssh_port: int = 22, daemon_port: int = DAEMON_PORT) -> None:
        self.host, self.user, self.password, self.log = host, user, password, log
        self.ssh_port, self.daemon_port = ssh_port, daemon_port
        self.ssh: paramiko.SSHClient | None = None

    # ------------------------------------------------------------ checks
    def resolve(self) -> str:
        return socket.gethostbyname(self.host)

    def connect(self) -> None:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(self.host, port=self.ssh_port, username=self.user, password=self.password, timeout=10, banner_timeout=15, look_for_keys=False, allow_agent=False)
        self.ssh = c

    def run(self, cmd: str, check: bool = True, stream: bool = True) -> tuple[int, str]:
        assert self.ssh is not None
        chan = self.ssh.get_transport().open_session()  # type: ignore[union-attr]
        chan.set_combine_stderr(True)
        chan.exec_command(cmd)
        out = []
        while True:
            data = chan.recv(4096)
            if not data:
                break
            text = data.decode("utf-8", "replace")
            out.append(text)
            if stream:
                for line in text.splitlines():
                    self.log("  " + line)
        code = chan.recv_exit_status()
        if check and code != 0:
            tail = "".join(out).strip().splitlines()[-8:]
            raise RuntimeError(f"command failed ({code}): {cmd}" + ("\n  " + "\n  ".join(tail) if tail else ""))
        return code, "".join(out)

    def daemon_status(self) -> dict:
        with urllib.request.urlopen(f"http://{self.host}:{self.daemon_port}/api/daemon/status", timeout=5) as r:
            return json.load(r)

    def robot_has_internet(self) -> bool:
        code, _ = self.run("curl -sS -m 8 -o /dev/null -w '%{http_code}' https://huggingface.co >/dev/null", check=False, stream=False)
        return code == 0

    def installed_version(self) -> str | None:
        code, out = self.run(f"/venvs/apps_venv/bin/python -m pip show {APP_NAME} 2>/dev/null | grep -i ^version", check=False, stream=False)
        if code != 0 or "Version:" not in out:
            return None
        return out.split(":", 1)[1].strip()

    def models_present(self) -> bool:
        code, _ = self.run(
            "test -f ~/.local/share/festival_pet/models/face_recognition_sface_2021dec.onnx && test -d ~/.local/share/festival_pet/models/vosk-model-small-en-us-0.15",
            check=False,
            stream=False,
        )
        return code == 0

    # ------------------------------------------------------------ install
    def upload(self, tarball: bytes) -> None:
        assert self.ssh is not None
        sftp = self.ssh.open_sftp()
        try:
            with sftp.file("/tmp/festival_pet.tgz", "wb") as f:
                f.set_pipelined(True)
                f.write(tarball)
        finally:
            sftp.close()
        self.run(f"rm -rf {REMOTE_DIR} && mkdir -p {REMOTE_DIR} && tar -xzf /tmp/festival_pet.tgz -C {REMOTE_DIR} && rm /tmp/festival_pet.tgz", stream=False)

    def setup(self) -> None:
        self.run(f"bash {REMOTE_DIR}/scripts/setup_offline.sh")

    def restore_dashboard(self) -> None:
        """Put the web dashboard back on port 8000; the script is root-only, so the SSH password goes to sudo on stdin."""
        import shlex

        self.run(f"printf '%s\\n' {shlex.quote(self.password)} | sudo -S -p '' bash {REMOTE_DIR}/scripts/restore_dashboard.sh")

    def _api(self, method: str, path: str, body: dict | None = None) -> dict | list | None:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(f"http://{self.host}:{self.daemon_port}/api{path}", data=data, method=method, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read()
            return json.loads(raw) if raw else None

    def set_startup_app(self) -> None:
        self._api("PUT", "/apps/startup-app", {"startup_app": APP_NAME})

    def start_app(self) -> None:
        self._api("POST", f"/apps/start-app/{APP_NAME}")

    def restart_daemon(self) -> None:
        self.run("sudo -n systemctl restart reachy-mini-daemon", check=True, stream=False)
        for _ in range(40):
            time.sleep(1.5)
            try:
                if self.daemon_status().get("state") == "running":
                    return
            except Exception:
                pass
        raise RuntimeError("daemon did not come back after restart")


class Steps:
    """The whole procedure as a list of (title, function) so the GUI can tick them off."""

    def __init__(self, robot: Robot, source: Path | None, branch: str, log, set_startup: bool, start_now: bool) -> None:
        self.robot, self.source, self.branch, self.log = robot, source, branch, log
        self.set_startup, self.start_now = set_startup, start_now
        self.notes: dict[str, str] = {}

    def check_reachable(self) -> str:
        ip = self.robot.resolve()
        self.robot.connect()
        st = self.robot.daemon_status()
        if not st.get("wireless_version"):
            raise RuntimeError("This daemon does not report itself as the wireless version. Is this a Reachy Mini Wireless?")
        return f"{ip}, daemon {st.get('state')}"

    def check_internet(self) -> str:
        has = self.robot.robot_has_internet()
        models = self.robot.models_present()
        if not has and not models:
            raise RuntimeError("The robot cannot reach the internet and has no models cached yet. Connect it to Wi‑Fi with internet for the first install.")
        if not has:
            return "offline, but models already cached (update-only mode)"
        return "online"

    def gather_source(self) -> str:
        if self.source is None:
            self.source = download_source(self.branch, self.log)
            return f"downloaded ({source_version(self.source)})"
        return f"local copy v{source_version(self.source)}"

    def upload(self) -> str:
        assert self.source is not None
        tar = make_tarball(self.source)
        self.robot.upload(tar)
        commit = git_commit(self.source)
        return f"v{source_version(self.source)}" + (f" ({commit})" if commit else "") + f", {len(tar) // 1024} KB sent"

    def run_setup(self) -> str:
        before = self.robot.installed_version()
        self.robot.setup()
        after = self.robot.installed_version()
        if after is None:
            raise RuntimeError("setup finished but the app is not installed in the apps venv")
        return f"v{before or '—'} → v{after}" + (" (same version number: the Mind page header shows the upload time)" if before == after else "")

    def restore_dashboard(self) -> str:
        self.robot.restore_dashboard()
        return f"http://{self.robot.host}:{self.robot.daemon_port}"

    def startup_and_start(self) -> str:
        msgs = []
        if self.set_startup:
            self.robot.set_startup_app()
            msgs.append("starts on antenna touch")
        if self.start_now:
            self.robot.start_app()
            msgs.append("running now")
        return ", ".join(msgs) or "skipped"

    def all(self):
        return [
            ("Reach the robot", self.check_reachable),
            ("Robot internet access", self.check_internet),
            ("Gather the app files", self.gather_source),
            ("Upload to the robot", self.upload),
            ("Install on the robot (can take a few minutes)", self.run_setup),
            ("Restore the port-8000 web dashboard & restart the daemon", self.restore_dashboard),
            ("Set as start-up app & start", self.startup_and_start),
        ]


# ============================================================================ GUI


def run_gui() -> None:
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Festival Pet installer for Reachy Mini")
    root.geometry("640x620")
    root.minsize(560, 520)

    frm = ttk.Frame(root, padding=14)
    frm.pack(fill="both", expand=True)
    ttk.Label(frm, text="Festival Pet → Reachy Mini Wireless", font=("", 15, "bold")).pack(anchor="w")
    ttk.Label(frm, text="Installs or updates the pet on the robot. Your computer and the robot must be on the same Wi‑Fi; the robot needs internet for the first install (model downloads).", wraplength=600).pack(anchor="w", pady=(2, 10))

    form = ttk.Frame(frm)
    form.pack(fill="x")
    vals = {"host": tk.StringVar(value=DEFAULT_HOST), "user": tk.StringVar(value=DEFAULT_USER), "password": tk.StringVar(value=DEFAULT_PASSWORD), "branch": tk.StringVar(value=DEFAULT_BRANCH)}
    for i, (label, key, show) in enumerate([("Robot address", "host", None), ("SSH user", "user", None), ("SSH password", "password", "•"), ("GitHub branch (for 'latest')", "branch", None)]):
        ttk.Label(form, text=label).grid(row=i, column=0, sticky="w", padx=(0, 8), pady=2)
        ttk.Entry(form, textvariable=vals[key], show=show, width=44).grid(row=i, column=1, sticky="ew", pady=2)
    form.columnconfigure(1, weight=1)

    src = source_root()
    use_local = tk.BooleanVar(value=src is not None)
    set_startup = tk.BooleanVar(value=True)
    start_now = tk.BooleanVar(value=True)
    opts = ttk.Frame(frm)
    opts.pack(fill="x", pady=(8, 4))
    ttk.Checkbutton(opts, text=f"Use the app files bundled with this installer{' (v' + source_version(src) + ')' if src else ' (none found)'}", variable=use_local, state=("normal" if src else "disabled")).pack(anchor="w")
    ttk.Label(opts, text="   unchecked = download the latest from GitHub (needs internet on this computer)", foreground="#666").pack(anchor="w")
    ttk.Checkbutton(opts, text="Make it the app that starts when an antenna is touched", variable=set_startup).pack(anchor="w")
    ttk.Checkbutton(opts, text="Start the pet right after installing", variable=start_now).pack(anchor="w")

    steps_frame = ttk.LabelFrame(frm, text="Progress", padding=8)
    steps_frame.pack(fill="x", pady=(8, 4))
    step_labels: list[tk.StringVar] = []

    log_box = tk.Text(frm, height=10, font=("Menlo" if sys.platform == "darwin" else "Consolas", 9), state="disabled", wrap="word")
    log_box.pack(fill="both", expand=True, pady=(4, 6))
    q: queue.Queue = queue.Queue()

    def log(msg: str) -> None:
        q.put(("log", msg))

    def pump() -> None:
        while True:
            try:
                kind, payload = q.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                log_box.configure(state="normal")
                log_box.insert("end", payload + "\n")
                log_box.see("end")
                log_box.configure(state="disabled")
            elif kind == "step":
                i, text = payload
                step_labels[i].set(text)
            elif kind == "done":
                go.configure(state="normal")
                status.set(payload)
        root.after(100, pump)

    status = tk.StringVar(value="")
    bottom = ttk.Frame(frm)
    bottom.pack(fill="x")
    ttk.Label(bottom, textvariable=status, foreground="#2a7").pack(side="left")
    go = ttk.Button(bottom, text="Install / Update")
    go.pack(side="right")

    def worker() -> None:
        for w in steps_frame.winfo_children():
            w.destroy()
        step_labels.clear()
        robot = Robot(vals["host"].get().strip(), vals["user"].get().strip(), vals["password"].get(), log)
        steps = Steps(robot, src if use_local.get() else None, vals["branch"].get().strip(), log, set_startup.get(), start_now.get())
        plan = steps.all()
        for title, _ in plan:
            var = tk.StringVar(value=f"○  {title}")
            step_labels.append(var)
            ttk.Label(steps_frame, textvariable=var).pack(anchor="w")
        try:
            for i, (title, fn) in enumerate(plan):
                q.put(("step", (i, f"◔  {title} …")))
                log(f"== {title}")
                note = fn()
                q.put(("step", (i, f"●  {title} — {note}")))
            q.put(("done", "Done. Open http://%s:8042 on your phone for Reachy's Mind." % vals["host"].get().strip()))
            log("All done.")
        except Exception as e:  # shown to the user, never swallowed
            q.put(("step", (i, f"✕  {title} — {e}")))
            log(f"FAILED: {e}")
            q.put(("done", "Failed, see the log above."))
        finally:
            if robot.ssh is not None:
                robot.ssh.close()

    def start() -> None:
        go.configure(state="disabled")
        status.set("Working…")
        threading.Thread(target=worker, daemon=True).start()

    go.configure(command=start)
    root.after(100, pump)
    root.mainloop()


def run_cli(argv: list[str]) -> int:
    """Same steps without a window (for automation and for testing the installer itself)."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--user", default=DEFAULT_USER)
    ap.add_argument("--password", default=DEFAULT_PASSWORD)
    ap.add_argument("--branch", default=DEFAULT_BRANCH)
    ap.add_argument("--latest", action="store_true", help="download from GitHub instead of using the bundled/local copy")
    ap.add_argument("--no-startup", action="store_true")
    ap.add_argument("--no-start", action="store_true")
    ap.add_argument("--ssh-port", type=int, default=22)
    ap.add_argument("--daemon-port", type=int, default=DAEMON_PORT)
    a = ap.parse_args(argv)
    robot = Robot(a.host, a.user, a.password, print, a.ssh_port, a.daemon_port)
    steps = Steps(robot, None if a.latest else source_root(), a.branch, print, not a.no_startup, not a.no_start)
    try:
        for title, fn in steps.all():
            print(f"== {title}")
            print(f"   {fn()}")
    finally:
        if robot.ssh is not None:
            robot.ssh.close()
    print("Done.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        sys.exit(run_cli(sys.argv[2:]))
    run_gui()
