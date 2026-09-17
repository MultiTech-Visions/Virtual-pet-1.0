"""The restored port-8000 dashboard, mounted onto a stand-in for the daemon's FastAPI app."""

import sys
import types
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
import reachy_dashboard  # noqa: E402


@dataclass
class Args:
    wireless_version: bool = True


def deprecated_daemon_app() -> FastAPI:
    """What reachy-mini >= 1.9.0 builds: API routes plus 'download the desktop app' pages."""
    app = FastAPI()

    @app.get("/api/daemon/status")
    def status():
        return {"state": "running", "wireless_version": True}

    for path in ("/", "/settings", "/logs"):
        app.get(path)(lambda: HTMLResponse("<h1>Web Dashboard Deprecated</h1>"))
    return app


def test_dashboard_replaces_deprecation_pages():
    app = reachy_dashboard.restore(deprecated_daemon_app(), Args())
    c = TestClient(app)
    home = c.get("/")
    assert home.status_code == 200 and "Reachy Mini dashboard" in home.text and "Deprecated" not in home.text
    assert "deprecat" not in home.text.lower()  # 1.8.4 carried a "deprecated soon" banner; gone
    settings = c.get("/settings")
    assert settings.status_code == 200 and "wifi" in settings.text.lower() and "deprecat" not in settings.text.lower()
    assert "daemonLogs" in c.get("/logs").text
    assert c.get("/static/css/app.css").status_code == 200
    assert c.get("/static/js/apps.js").status_code == 200
    assert c.get("/api/daemon/status").json()["state"] == "running"  # API untouched
    assert [r.path for r in app.router.routes].count("/") == 1


def test_lite_daemon_has_no_wireless_pages():
    app = reachy_dashboard.restore(deprecated_daemon_app(), Args(wireless_version=False))
    c = TestClient(app)
    assert "Reachy Mini dashboard" in c.get("/").text
    assert c.get("/settings").status_code == 404


def test_restore_on_daemon_that_still_has_its_dashboard():
    """reachy-mini <= 1.8.4 mounts /static itself; a second mount must not be stacked."""
    app = deprecated_daemon_app()
    app.mount("/static", StaticFiles(directory=reachy_dashboard.STATIC_DIR), name="static")
    reachy_dashboard.restore(app, Args())
    assert [r.path for r in app.router.routes].count("/static") == 1
    assert TestClient(app).get("/static/css/app.css").status_code == 200


def test_wrap_create_app_patches_the_daemon_module():
    fake = types.ModuleType("fake_daemon_main")
    fake.create_app = lambda args, health_check_event=None: deprecated_daemon_app()
    reachy_dashboard.wrap_create_app(fake)
    app = fake.create_app(Args(), None)
    assert "Reachy Mini dashboard" in TestClient(app).get("/").text


def test_every_dashboard_file_is_a_package_datafile():
    """Every JS/CSS/font/svg the templates need is inside the package (nothing left in the SDK)."""
    for rel in ("css/app.css", "js/apps.js", "js/appstore.js", "js/daemon.js", "js/wifi.js", "js/update.js", "js/logs.js",
                "js/volume_control.js", "js/move_player.js", "js/hf_auth.js", "js/health_check.js", "js/notification.js"):
        assert (reachy_dashboard.STATIC_DIR / rel).is_file(), rel
    for rel in ("base.html", "index.html", "settings.html", "logs.html"):
        assert (reachy_dashboard.TEMPLATES_DIR / rel).is_file(), rel
