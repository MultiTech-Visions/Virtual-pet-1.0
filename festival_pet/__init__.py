"""Festival Pet: an offline, non-verbal virtual-pet behavior app for Reachy Mini Wireless."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("festival_pet")
except PackageNotFoundError:  # running from a checkout that was never pip-installed
    __version__ = "0+unpackaged"


def build_info() -> dict:
    """Version plus the stamp the installer writes at upload time (git commit, upload time)."""
    info = {"version": __version__, "commit": None, "built": None}
    try:
        from festival_pet import _build  # written by the installer; absent in a plain checkout
    except ImportError:
        return info
    info["commit"], info["built"] = _build.COMMIT, _build.BUILT
    return info
