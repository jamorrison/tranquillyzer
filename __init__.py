# __init__.py
"""Main script package for Tranquillyzer."""

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("tranquillyzer")
except Exception:
    try:
        from _version import __version__
    except Exception:
        __version__ = "0.0.0"
