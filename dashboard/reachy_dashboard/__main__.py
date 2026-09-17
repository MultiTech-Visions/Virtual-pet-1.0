"""Run the Reachy Mini daemon with the web dashboard restored on port 8000.

Drop-in for ``python -m reachy_mini.daemon.app.main``: same arguments, same daemon,
plus ``/``, ``/settings``, ``/logs`` and ``/static`` served from the bundled dashboard.
"""

import reachy_mini.daemon.app.main as daemon_main

from reachy_dashboard import wrap_create_app

wrap_create_app(daemon_main)
daemon_main.main()
