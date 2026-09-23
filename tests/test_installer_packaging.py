import io
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "installer"))
import reachy_installer as ri  # noqa: E402


def test_tarball_normalizes_crlf_and_keeps_layout(tmp_path):
    root = tmp_path
    (root / "festival_pet").mkdir()
    (root / "scripts").mkdir()
    (root / "dashboard" / "reachy_dashboard" / "static" / "css").mkdir(parents=True)
    (root / "dashboard" / "reachy_dashboard" / "static" / "css" / "app.css").write_bytes(b"body{}\r\n")
    (root / "festival_pet" / "main.py").write_bytes(b"print('hi')\r\n")
    (root / "festival_pet" / "__pycache__").mkdir()
    (root / "festival_pet" / "__pycache__" / "x.pyc").write_bytes(b"junk")
    (root / "scripts" / "setup_offline.sh").write_bytes(b"#!/usr/bin/env bash\r\nset -euo pipefail\r\n")
    (root / "pyproject.toml").write_bytes(b'version = "9.9.9"\r\n')
    (root / "README.md").write_bytes(b"# x\r\n")
    tar = tarfile.open(fileobj=io.BytesIO(ri.make_tarball(root)), mode="r:gz")
    names = tar.getnames()
    assert "scripts/setup_offline.sh" in names and "festival_pet/main.py" in names
    assert "dashboard/reachy_dashboard/static/css/app.css" in names
    stamp = tar.extractfile("festival_pet/_build.py").read().decode()
    assert "BUILT = '" in stamp and "COMMIT = " in stamp
    assert not any("__pycache__" in n for n in names)
    sh = tar.extractfile("scripts/setup_offline.sh").read()
    assert b"\r" not in sh and sh.endswith(b"pipefail\n")
    assert tar.getmember("scripts/setup_offline.sh").mode & 0o111
    assert ri.source_version(root) == "9.9.9"
