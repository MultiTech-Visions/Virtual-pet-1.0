"""The Reachy Mini web dashboard (port 8000), as shipped up to reachy-mini 1.8.4.

reachy-mini 1.9.0 deleted ``reachy_mini/daemon/app/dashboard`` and replaced ``/``,
``/settings`` and ``/logs`` with a "download the desktop app" page. The REST and
WebSocket API the dashboard talks to (``/api/...``, ``/wifi/...``, ``/update/...``,
``/cache/...``, ``/logs/ws/daemon``) is unchanged, so the dashboard files are shipped
here verbatim and mounted back onto the daemon's own FastAPI app.

``python -m reachy_dashboard <daemon args>`` runs the daemon with the dashboard back in
place; ``restore(app, args)`` does the mounting on an already-built app.
"""

from pathlib import Path

from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

DASHBOARD_DIR = Path(__file__).parent
STATIC_DIR = DASHBOARD_DIR / "static"
TEMPLATES_DIR = DASHBOARD_DIR / "templates"

# Routes the daemon registers that the dashboard takes over.
REPLACED_PATHS = frozenset({"/", "/settings", "/logs", "/static"})


def restore(app, args):
    """Mount the dashboard onto the daemon's FastAPI ``app``.

    ``args`` is the daemon's ``Args``; the templates only read ``args.wireless_version``.
    Any existing ``/``, ``/settings``, ``/logs`` routes and ``/static`` mount are removed
    first, so this works both on a daemon that serves the deprecation page and on one
    that still ships its own dashboard.
    """
    app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", None) not in REPLACED_PATHS]

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    @app.get("/")
    async def dashboard(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "index.html", {"args": args})

    if args.wireless_version:

        @app.get("/settings")
        async def settings(request: Request) -> HTMLResponse:
            return templates.TemplateResponse(request, "settings.html", {})

        @app.get("/logs")
        async def logs_page(request: Request) -> HTMLResponse:
            return templates.TemplateResponse(request, "logs.html", {})

    return app


def wrap_create_app(daemon_main):
    """Replace ``daemon_main.create_app`` with one that mounts the dashboard on its result.

    ``run_app`` looks ``create_app`` up as a module global, so patching the attribute is
    enough for ``daemon_main.main()`` to build a daemon with the dashboard.
    """
    original = daemon_main.create_app

    def create_app(args, *a, **kw):
        return restore(original(args, *a, **kw), args)

    daemon_main.create_app = create_app
    return create_app
