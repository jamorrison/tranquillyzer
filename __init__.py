# __init__.py
"""Main script package for Tranquillyzer."""


def _resolve_version():
    """Determine package version with fallback chain."""
    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version("tranquillyzer")
    except Exception:
        try:
            from _version import __version__ as v

            return v
        except Exception:
            return "0.0.0"


__version__ = _resolve_version()
